#!/usr/bin/env python3
"""Cycle-level model of a descriptor-chain DMA engine with descriptor checks.

The public DMA DUT moves bytes for one programmed transfer.  This held-out DUT
never moves a byte: it walks a linked list of descriptors, validates each one
before issuing a burst, and stops on the first invalid descriptor.  Coverage is
driven by the walk itself -- pointer chasing, alignment and boundary checks,
the completion handshake, the retry budget, word writes over a descriptor, the
mid-transfer abort, and the chain limit.
"""

IDLE, FETCH, VALIDATE, ISSUE, WAIT_ACK, ERROR, DONE, ABORT = range(8)
HALT = 13

ZERO_SRC = 1
LOW_SRC = 2
HIGH_SRC = 3
ZERO_DST = 4
LOW_DST = 5
HIGH_DST = 6
ZERO_LEN = 7
OVER_LEN = 8
UNALIGNED = 9
UNSUPPORTED_MODE = 10
UNSUPPORTED_FLAG = 11
RETRY_EXHAUSTED = 12


BURST_BEATS = 2
STRIDE = 0x10


class DmaDescEngineValidation:
    def __init__(self, desc_base=0x100, corrupt_xor=0, chain_limit=4,
                 ack_latency=2, retry_limit=3):
        self.desc_base = int(desc_base) & 0xFFFFFFFF
        self.corrupt_xor = int(corrupt_xor) & 1
        self.chain_limit = min(6, max(2, int(chain_limit)))
        self.ack_latency = min(4, max(1, int(ack_latency)))
        self.retry_limit = min(3, max(1, int(retry_limit)))
        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        self.state = IDLE
        self.event_state = IDLE
        self.result = 0
        self.fail_class = 0

        self.desc_ptr = self.desc_base
        self._host_ptr = self.desc_base
        self.next_ptr = 0
        self.ptr_delta = 0
        self.dirty = False
        self.word_index = 0
        self.descriptor = self._blank_descriptor()

        self.checks = 0
        self.bursts = 0
        self.retries = 0
        self.remaining = 0
        self.ack_wait = 0
        self.chain_len = 0
        self._chain = []
        self._overrides = {}
        self._arm_base = self.desc_base

        self._words_left = 0
        self._ack_seen = False
        self._prev_wait = False
        self._abort_flag = False
        self._limit_flag = False
        self._done_flag = False
        self._start_pending = False

        self._seq = {}

    @staticmethod
    def _blank_descriptor():
        return {"src": 0, "dst": 0, "length": 0, "mode": 0, "aflag": 0, "eol": 0}

    # ------------------------------------------------------------------
    # Descriptor chain, built fresh at each start
    # ------------------------------------------------------------------
    def _build_chain(self, base):
        """Build the descriptor list the engine will walk.

        When the address-corruption knob is enabled every stored link carries a
        flipped low bit, so the engine's internal view of the list is scrambled
        while the walk remains reachable: the stimulus re-issues the same start
        pointer and the pointer delta reveals the offset.
        """
        chain = []
        for index in range(self.chain_limit + 1):
            tail = index == self.chain_limit
            ptr = (base + index * STRIDE) & 0xFFFFFFFF
            stored = (ptr ^ 0x4) if self.corrupt_xor else ptr
            entry = {
                "ptr": stored,
                "host_ptr": ptr,
                "src": 0x240 + index * 0x10,
                "dst": 0x800 + index * 0x10,
                "length": 0x40 - index * 8,
                "mode": [0, 1, 3, 1, 0, 3][index % 6],
                "aflag": 0,
                "eol": 1 if tail else 0,
            }
            entry.update(self._overrides.get(ptr, {}))
            chain.append(entry)
        return chain

    def _entry_for(self, ptr):
        # A lookup matches either the stored (possibly scrambled) link or the
        # host-visible pointer the stimulus used to arm the chain.
        ptr &= 0xFFFFFFFF
        for index, entry in enumerate(self._chain):
            if entry["ptr"] == ptr or entry["host_ptr"] == ptr:
                return index
        return None

    def word_idx_write(self, word_idx, value):
        """Edit descriptor word 0..3; the edit is remembered for this pointer and
        lands in the chain entry, so any later fetch of the same pointer sees the
        modified copy."""
        self.word_index = word_idx & 3
        key = ["src", "length", "dst", "mode"][word_idx & 3]
        if key == "mode":
            self.descriptor["mode"] = value & 3
            self.descriptor["aflag"] = (value >> 31) & 1
        else:
            self.descriptor[key] = value
        host_ptr = self._host_ptr
        pending = self._overrides.setdefault(host_ptr, {})
        if key == "mode":
            pending["mode"] = value & 3
            pending["aflag"] = (value >> 31) & 1
        else:
            pending[key] = value
        index = self._entry_for(host_ptr)
        if index is not None:
            self._chain[index].update(pending)

    def _validate(self):
        spec = self.descriptor
        if spec["src"] == 0:
            return ZERO_SRC
        if spec["dst"] == 0:
            return ZERO_DST
        if spec["src"] < 0x10:
            return LOW_SRC
        if spec["dst"] < 0x10:
            return LOW_DST
        if (spec["src"] & 3) or (spec["dst"] & 3):
            return UNALIGNED
        if (spec["src"] + spec["length"]) >= 0x10000:
            return HIGH_SRC
        if (spec["dst"] + spec["length"]) >= 0x10000:
            return HIGH_DST
        if spec["mode"] not in (0, 1, 3):
            return UNSUPPORTED_MODE
        if spec["aflag"]:
            return UNSUPPORTED_FLAG
        if spec["length"] == 0:
            return ZERO_LEN
        if spec["length"] > 0x1000:
            return OVER_LEN
        return 0

    # ------------------------------------------------------------------
    # Engine step
    # ------------------------------------------------------------------
    def _clear_pulses(self):
        self.result = 0
        self.fail_class = 0
        for key in list(self._seq):
            self._seq[key] = False

    def _enter(self, state):
        # event_state records where the cycle started, so it must not be moved
        # by a transition that happens inside the cycle.  It is snapshotted in
        # step() and only reset here is avoided on purpose.
        self.state = state

    def _retire(self, eol):
        """Complete one descriptor.  The last descriptor of the chain (either by
        its own end-of-list marker or by hitting the programmed chain limit)
        moves the engine to DONE; anything else returns to IDLE to fetch the
        next descriptor of the same chain."""
        self.chain_len += 1
        self.dirty = False
        self.ack_wait = 0
        self.result = 11
        if eol or self.chain_len >= self.chain_limit:
            self._limit_flag = True
            self._seq["seq2"] = True
            self._enter(DONE)
        else:
            self._enter(IDLE)

    def step(self, request):
        # The sampled state must be the state the cycle began in: cross coverage
        # pairs it with the result or failure class produced by that cycle.
        self.event_state = self.state
        self._clear_pulses()
        if not int(request.get("reset_n", 1)):
            self.reset()
            self.event_state = IDLE
            self._seq["seq8"] = True
            return

        ack = int(bool(request.get("ack")))
        abort = int(bool(request.get("abort")))

        handler = {
            IDLE: self._step_idle,
            FETCH: self._step_fetch,
            VALIDATE: self._step_validate,
            ISSUE: self._step_issue,
            WAIT_ACK: self._step_wait_ack,
            ERROR: self._step_error,
            DONE: self._step_done,
            ABORT: self._step_abort,
        }[self.state]
        handler(request, ack, abort)

    def _step_idle(self, request, ack, abort):
        # The chain itself lives across descriptors: only a new start rebuilds
        # it.  This is what lets the walk advance past its first descriptor.
        if request.get("start"):
            self.chain_len = 0
            self.retries = 0
            self.checks = 0
            self._abort_flag = False
            self._limit_flag = False
            self._done_flag = False
            self._prev_wait = False
            base = int(request.get("desc_ptr", 0)) & 0xFFFFFFFF
            if base == 0:
                base = self.desc_base
            self._arm_base = base
            self._chain = self._build_chain(base)
            # The start is an event of the IDLE state: the state is sampled as
            # IDLE for this cycle, and the fetch begins on the cycle where the
            # start request is no longer asserted.
            self.result = 1
            self._seq["seq1"] = True
            self._start_pending = True
            return
        if self._start_pending:
            self._start_pending = False
            self._enter(FETCH)
            return
        # A fetch request is also honoured straight from IDLE, so a descriptor
        # can be latched without re-arming the whole chain.
        if request.get("fetch_valid"):
            self._fetch(request)
            return

    def _fetch(self, request):
        self.dirty = False
        ptr = int(request.get("desc_ptr", 0)) & 0xFFFFFFFF
        index = self._entry_for(ptr)
        if index is None:
            self.dirty = True
            self.result = 5
            return
        entry = self._chain[index]
        self.descriptor = {"src": entry["src"], "dst": entry["dst"],
                           "length": entry["length"], "mode": entry["mode"],
                           "aflag": entry["aflag"], "eol": entry["eol"]}
        self.ptr_delta = (entry["host_ptr"] - self.desc_ptr) & 0xFFFFFFFF
        self.desc_ptr = entry["ptr"]
        self._host_ptr = entry["host_ptr"]
        self.next_ptr = self._chain[min(index + 1, len(self._chain) - 1)]["host_ptr"]
        self.word_index = 0
        if request.get("word_we"):
            self.word_idx_write(int(request.get("word_idx", 0)) & 3,
                                int(request.get("data", 0)) & 0xFFFFFFFF)
            self.dirty = True
            self.result = 8
            return
        self.result = 4
        self._enter(VALIDATE)

    def _step_fetch(self, request, ack, abort):
        self.dirty = False
        if request.get("fetch_valid"):
            self._fetch(request)
            return
        if request.get("word_we"):
            self.word_idx_write(int(request.get("word_idx", 0)) & 3,
                                int(request.get("data", 0)) & 0xFFFFFFFF)
            self.dirty = True
            self.result = 8

    def _step_validate(self, request, ack, abort):
        self.checks += 1
        failure = self._validate()
        if failure:
            self.fail_class = failure
            self.result = 3
            self._seq["seq4"] = True
            self._enter(ERROR)
            return
        self.fail_class = 0
        self.result = 4
        self.remaining = self.descriptor["length"]
        self._abort_flag = False
        self._limit_flag = False
        mode = self.descriptor["mode"]
        self.bursts += 1
        self._ack_seen = False
        self._prev_wait = False
        if mode == 1:
            self._words_left = 0
            self.ack_wait = self.ack_latency
            self._enter(WAIT_ACK)
        else:
            self._words_left = BURST_BEATS
            self._enter(ISSUE)

    def _step_issue(self, request, ack, abort):
        if abort:
            self._abort_flag = True
            self._seq["seq3"] = True
            self.result = 2
            self._enter(ABORT)
            return
        if self._limit_flag:
            return
        self._words_left -= 1
        self.remaining = max(0, self.remaining - 1)
        self.result = 10
        if self._words_left <= 0:
            self._seq["seq5"] = True
            self._retire(self.descriptor["eol"])

    def _step_wait_ack(self, request, ack, abort):
        if abort:
            self._abort_flag = True
            self._seq["seq3"] = True
            self.result = 2
            self._enter(ABORT)
            return
        if ack:
            self._ack_seen = True
            self._retire(self.descriptor["eol"])
            return
        # No acknowledgement this cycle: spend one cycle of the allowed wait,
        # then one unit of the retry budget.
        if self.ack_wait > 0:
            self.ack_wait -= 1
            self.result = 9
            return
        self.retries += 1
        self.result = 9
        if self.retries >= self.retry_limit:
            self.fail_class = RETRY_EXHAUSTED
            self.result = 3
            self._seq["seq4"] = True
            self._enter(ERROR)
            return
        self.ack_wait = self.ack_latency

    def _step_error(self, request, ack, abort):
        if request.get("clear"):
            self.result = 7
            self._seq["seq6"] = True
            self._enter(IDLE)

    def _step_done(self, request, ack, abort):
        # Completion handshake: the engine stays in DONE until the host
        # acknowledges the completed chain.
        if ack:
            self.result = 6
            self._seq["seq7"] = True
            self._enter(IDLE)

    def _step_abort(self, request, ack, abort):
        self.result = 2
        self._enter(IDLE)

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    def read_signals(self):
        signals = {
            "state": self.state, "event_state": self.event_state,
            "result": self.result, "fail_class": self.fail_class,
            "word_index": self.word_index, "dirty": int(self.dirty),
            "ack_wait_class": self._ack_wait_class(),
            "chain_class": self._chain_class(),
            "checks": 3 if self.checks >= 3 else self.checks,
            "ptr_delta_class": 3 if self.ptr_delta >= 3 else self.ptr_delta,
        }
        signals.update({f"seq{i}": int(self._seq.get(f"seq{i}", False))
                        for i in range(1, 9)})
        return signals

    def _ack_wait_class(self):
        if self.ack_wait <= 0:
            return 0
        if self.ack_wait >= self.ack_latency:
            return 1
        return 2

    def _chain_class(self):
        if self.chain_len == 0:
            return 0
        if self.chain_len == 1:
            return 1
        if self.chain_len < self.chain_limit:
            return 2
        return 3


if __name__ == "__main__":
    dut = DmaDescEngineValidation()
    print("dma_desc_engine_validation local_sim loaded", dut.read_signals())
