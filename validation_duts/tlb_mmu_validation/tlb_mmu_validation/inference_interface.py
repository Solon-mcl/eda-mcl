#!/usr/bin/env python3
"""Random and reachability baselines for the held-out TLB/MMU DUT."""

from collections import deque
import json
import os
import numpy as np

LOAD, STORE, FETCH, ATOMIC = range(4)


class InferenceInterface:
    DIMS = 16
    action_dims = 16
    BOUNDS = np.asarray([2, 256, 256, 256, 256, 4, 2, 4, 2, 2, 2, 2, 2, 2, 1, 1],
                        dtype=np.float32)

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
            path = os.path.join(os.path.dirname(self.covergroup_path), "coverage_meta.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as handle:
                    return int(json.load(handle)["total_bins"])
        return 78

    @staticmethod
    def _bytes(value):
        return [(int(value) >> (8 * i)) & 0xFF for i in range(4)]

    def _action(self, valid=0, vpn=0, op=0, priv=1, asid=0, glob=0, fence=0,
                flush=0, satp=0, stall=0, reset=1):
        return np.asarray(
            [valid, *self._bytes(vpn), op, priv, asid, glob, fence, flush,
             satp, stall, reset, 0, 0], dtype=np.float32)

    def _access(self, vpn, op, priv, asid=0, glob=0, fence=0, flush=0, satp=0):
        self._program.append(self._action(1, vpn, op, priv, asid, glob, fence,
                                          flush, satp))

    def _idle(self, stall=0, reset=1):
        self._program.append(self._action(valid=0, stall=stall, reset=reset))

    def _build_program(self):
        # DUT-only reset, then front-end backpressure before the first
        # translation so the stall-recovery sequence has a witness.
        self._idle(reset=0)
        self._idle(stall=1)

        # Cold walk, repeat hit, invalid PTE, and recovery from the cached
        # entry: walk_ok -> refill -> hit, plus page-fault recovery.
        self._access(18, LOAD, 1, asid=0)
        self._access(18, LOAD, 1, asid=0)
        self._access(2, LOAD, 1)
        self._access(18, LOAD, 1)

        # User/supervisor permission split on a supervisor-only page, all four
        # access classes, and the A/D bit lifecycle of one page.
        self._access(0, LOAD, 0, asid=0)
        self._access(0, LOAD, 1)
        self._access(0, STORE, 1)
        self._access(0, STORE, 1)
        self._access(0, LOAD, 1)
        self._access(0, FETCH, 1)

        # Atomic and fetch on a fully permitted page exercise the remaining
        # access classes and leave clean/dirty witnesses for each of them.
        self._access(19, FETCH, 1)
        self._access(19, FETCH, 1)
        self._access(19, ATOMIC, 1)
        self._access(19, ATOMIC, 1)
        self._access(3, ATOMIC, 1)

        # Write- and execute-denied pages: store/atomic protection faults,
        # then a legal load that still fills the entry.
        self._access(4, STORE, 1)
        self._access(4, ATOMIC, 1)
        self._access(4, LOAD, 1)

        # Write to an unmapped page, then a locked page that walks but never
        # fills, then enough set pressure to force a victim replacement.
        self._access(5, STORE, 1)
        self._access(8, LOAD, 1)
        self._access(20, LOAD, 1)

        # ASID aliasing: same VPN under two address-space identifiers, in both
        # privilege modes, each forcing a second translation of one page.
        self._access(1, LOAD, 1, asid=0)
        self._access(1, LOAD, 1, asid=1)
        self._access(18, LOAD, 0, asid=1)
        self._access(21, LOAD, 0, asid=0)
        self._access(5, LOAD, 0)

        # A global page must outlive an address-space fence and keep hitting;
        # a local page in the same fence must walk again.  The store is also
        # the first touch of this page, so it witnesses a store miss.
        self._access(17, STORE, 1, asid=0, glob=1)
        self._access(17, LOAD, 1, asid=0, fence=1)
        self._access(0, LOAD, 1, asid=0, fence=1)

        # Full flush, then a page-table-root change: both invalidate the whole
        # TLB, and the second must also change the walked permission bits.
        self._access(18, LOAD, 1, asid=0, flush=1)
        self._access(18, LOAD, 1, asid=0, satp=1)

        # Post-change traffic re-warms the TLB under the new page table.
        self._access(19, LOAD, 1, asid=0)
        self._access(0, STORE, 1)

        # Generic pressure tail: mixed accesses over a small page window keep
        # replacement, A/D, and the functional crosses reachable when the
        # hidden set count, associativity, salt, or ASID width differ from the
        # published defaults.
        for repeat in range(3):
            for index in range(24):
                op = (index + repeat) % 4
                self._access(0x40 + ((index * 3 + repeat) & 0x3F), op, 1,
                             asid=(index + repeat) & 3,
                             glob=1 if index % 7 == 0 else 0)

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
        action[13] = 0.0 if step < 2 else 1.0
        return action


if __name__ == "__main__":
    agent = InferenceInterface(policy="greedy")
    assert agent.predict(np.zeros(78), 0, 100).shape == (16,)
    print("tlb_mmu_validation inference interface OK")
