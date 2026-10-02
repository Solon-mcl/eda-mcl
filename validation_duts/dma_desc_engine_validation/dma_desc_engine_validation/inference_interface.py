#!/usr/bin/env python3
"""Random and reachability baselines for the held-out descriptor DMA engine."""

from collections import deque
import json
import os
import numpy as np

MODE_BLOCK = 0
MODE_HANDSHAKE = 1
MODE_STRIDE = 3
BASE = 0x100
STRIDE = 0x10


class InferenceInterface:
    DIMS = 16
    action_dims = 16
    BOUNDS = np.asarray([2, 2, 2, 4, 256, 256, 256, 256,
                         256, 256, 256, 256, 2, 2, 2, 1], dtype=np.float32)

    def __init__(self, dut_spec_path=None, covergroup_path=None,
                 policy="random", seed=7):
        self.dut_spec_path = dut_spec_path
        self.covergroup_path = covergroup_path
        self.policy = policy
        self.rng = np.random.RandomState(seed)
        self._program = deque()
        if policy == "greedy":
            self._build_program()

    @property
    def total_bins(self):
        if self.covergroup_path:
            path = os.path.join(os.path.dirname(self.covergroup_path),
                                "coverage_meta.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as handle:
                    return int(json.load(handle)["total_bins"])
        return 86

    @staticmethod
    def _bytes(value):
        return [(int(value) >> (8 * i)) & 0xFF for i in range(4)]

    def _action(self, start=0, fetch_valid=0, word_we=0, word_idx=0,
                desc_ptr=0, data=0, ack=0, abort=0, reset=1, clear=0):
        return np.asarray(
            [start, fetch_valid, word_we, word_idx,
             *self._bytes(desc_ptr), *self._bytes(data),
             ack, abort, reset, clear], dtype=np.float32)

    def _put(self, action, cycles=1):
        for _ in range(max(1, int(cycles))):
            self._program.append(action.copy())

    def _head(self, index):
        return BASE + index * STRIDE

    # ------------------------------------------------------------------
    def _arm(self, head=None):
        """Start a chain.  The start request is an IDLE-state event, so the
        engine latches it on this cycle and only enters FETCH on the next one;
        the following fill cycle absorbs that transition."""
        self._put(self._action(start=1, desc_ptr=BASE if head is None else head))
        self._put(self._action())

    def _fetch(self, ptr, word_we=0, word_idx=0, data=0):
        self._put(self._action(fetch_valid=1, desc_ptr=ptr,
                               word_we=word_we, word_idx=word_idx, data=data))

    def _patch(self, ptr, word_idx, value):
        """Latch the descriptor with a single word overwritten in the same
        cycle, then latch it again so the engine validates the edited copy."""
        self._fetch(ptr, word_we=1, word_idx=word_idx, data=value)
        self._fetch(ptr)

    def _drain(self, mode, last=False):
        """Drive one descriptor to retirement.

        The engine latches a fetch on one cycle and only advances to
        VALIDATE/ISSUE on the next, so every descriptor gets one settling cycle
        before its transfer phase starts.  A handshake descriptor needs an
        acknowledgement once it is actually parked in WAIT_ACK, so the pulse is
        widened instead of being trusted to land on the exact cycle; the extra
        cycles are harmless because a retired descriptor ignores them.
        """
        self._put(self._action())
        if mode == MODE_HANDSHAKE:
            self._put(self._action(ack=1), 3)
        else:
            self._put(self._action(), 4)
        if last:
            # The end of the chain parks in DONE, whose completion handshake
            # also needs an acknowledgement before the engine returns to IDLE.
            self._put(self._action(ack=1), 3)
        self._put(self._action(), 2)

    def _walk(self, limit=6):
        """Walk a descriptor chain without relying on the hidden chain limit.

        The engine stops on whichever comes first -- the end-of-list marker or
        the programmed chain limit -- so the stimulus walks past the largest
        supported limit and lets the engine park in DONE on its own.  Each
        descriptor is driven according to the mode its index receives.
        """
        for index in range(limit + 1):
            mode = [MODE_BLOCK, MODE_HANDSHAKE, MODE_STRIDE,
                    MODE_HANDSHAKE, MODE_BLOCK, MODE_STRIDE][index % 6]
            self._fetch(self._head(index))
            self._drain(mode, last=index == limit)
        # Release the completion handshake whether or not the limit was reached.
        self._put(self._action(ack=1), 3)
        self._put(self._action(), 2)

    def _anchor(self):
        """Force a known starting point between sections.

        A reset clears the chain, the retry budget and any parked state, so the
        section that follows is independent of how long the previous chain ran.
        """
        self._put(self._action(reset=0), 2)
        self._put(self._action(), 2)

    def _build_program(self):
        # ---- chain 1: clean walk over the whole chain, stopping at the limit.
        self._arm()
        self._walk()
        self._put(self._action(), 3)      # consume DONE, return to IDLE

        # ---- chain 2: descriptor word writes, then a fetch miss.
        self._anchor()
        self._arm()
        for word_idx in range(4):
            self._patch(self._head(0), word_idx, 0x80)
        self._put(self._action(), 3)
        self._anchor()
        self._arm()
        self._fetch(0xDEAD)
        self._put(self._action(), 3)

        # ---- chain 3: mid-burst abort from ISSUE, then from WAIT_ACK.
        self._anchor()
        self._arm()
        self._fetch(self._head(0))
        self._put(self._action())
        self._put(self._action(abort=1))
        self._put(self._action(), 3)
        self._anchor()
        self._arm()
        self._fetch(self._head(1))
        self._put(self._action(ack=0))
        self._put(self._action(abort=1))
        self._put(self._action(), 3)

        # ---- chain 4: acknowledgement withheld past the retry budget, then
        # the error is cleared back to DONE.
        self._anchor()
        self._arm()
        self._fetch(self._head(1))
        self._put(self._action(ack=0), 12)
        self._put(self._action(clear=1))
        self._put(self._action(), 3)

        # ---- chain 5: every descriptor failure class.  Each case starts a
        # fresh chain so the accumulated word edits of one case cannot shadow
        # the next one's edited field.
        broken = [
            (0, 0x00000000),   # zero source
            (0, 0x00000008),   # source below the minimum
            (0, 0x00FFF000),   # source plus length past the window
            (2, 0x00000000),   # zero destination
            (2, 0x00000008),   # destination below the minimum
            (2, 0x00FFF000),   # destination plus length past the window
            (1, 0x00000000),   # zero length
            (1, 0x00002000),   # length above the burst maximum
            (0, 0x00000241),   # unaligned source
            (3, 0x00000002),   # unsupported mode
            (3, 0x80000003),   # supported mode, unsupported address flag
        ]
        for case, (word_idx, value) in enumerate(broken):
            head = (BASE + 0x200 + case * 0x40) & 0xFFFFFFFF
            # Anchor each case with a reset: the previous case may have left the
            # engine in ERROR or parked in DONE, and a reset makes the next arm
            # independent of whatever the previous case did.
            self._put(self._action(reset=0), 2)
            self._put(self._action(), 2)
            self._arm(head)
            # Latch with the word overwritten, then latch again so the engine
            # validates the edited descriptor and enters ERROR.
            self._fetch(head, word_we=1, word_idx=word_idx, data=value)
            self._fetch(head)
            self._put(self._action(), 2)
        self._put(self._action(clear=1))
        self._put(self._action(), 2)

        # ---- chain 6: clean exit with a full walk, again ending in DONE.
        self._arm()
        self._walk()
        self._put(self._action(), 3)

        # ---- chain 7: asynchronous reset in the middle of a walk, then a
        # fresh chain proves the engine recovers.
        self._arm()
        self._fetch(self._head(0))
        self._drain(MODE_BLOCK)
        self._put(self._action(reset=0), 2)
        self._put(self._action(), 2)
        self._arm()
        self._fetch(self._head(0))
        self._drain(MODE_BLOCK)
        self._put(self._action(), 3)

    def reset(self):
        self._program.clear()
        if self.policy == "greedy":
            self._build_program()

    def predict(self, coverage_state, step, max_steps):
        if self.policy == "greedy":
            if not self._program:
                self._build_program()
            return self._program.popleft()
        action = (self.rng.uniform(0, 1, self.DIMS) * self.BOUNDS).astype(np.float32)
        action[14] = 0.0 if step < 2 else 1.0
        return action


if __name__ == "__main__":
    agent = InferenceInterface(policy="greedy")
    assert agent.predict(np.zeros(86), 0, 100).shape == (16,)
    print("dma_desc_engine_validation inference interface OK")
