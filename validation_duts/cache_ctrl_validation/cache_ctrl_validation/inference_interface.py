#!/usr/bin/env python3
"""Reachability and random baselines for the held-out cache DUT.

Do not import this policy into a candidate model: it is validation-set ground
truth used to establish that the bins are reachable.
"""

from collections import deque
import json
import os

import numpy as np


class InferenceInterface:
    DIMS = 16
    BOUNDS = np.asarray([2, 4, 256, 256, 256, 256, 256, 256, 256, 256,
                         2, 2, 1, 1, 1, 1], dtype=np.float32)

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
            meta = os.path.join(os.path.dirname(self.covergroup_path), "coverage_meta.json")
            if os.path.exists(meta):
                with open(meta, encoding="utf-8") as handle:
                    return int(json.load(handle)["total_bins"])
        return 75

    @staticmethod
    def _bytes(value):
        return [(int(value) >> (8 * i)) & 0xFF for i in range(4)]

    def _action(self, valid=0, op=0, addr=0, data=0, ready=1, reset=1):
        return np.asarray([valid, op, *self._bytes(addr), *self._bytes(data),
                           ready, reset, 0, 0, 0, 0], dtype=np.float32)

    def _put(self, action, cycles=1):
        for _ in range(max(1, int(cycles))):
            self._program.append(action.copy())

    def _txn(self, op, addr=0, data=0, stall=False, wait=10):
        self._put(self._action(1, op, addr, data))
        if stall:
            self._put(self._action(ready=0), 3)
        self._put(self._action(), wait)

    def _build_program(self):
        self._put(self._action(reset=0), 2)
        # A cold write establishes the write-miss cross independently, then a
        # DUT-only reset restores a clean starting point (coverage persists).
        self._txn(1, 0x110, 0x12345678)
        self._put(self._action(reset=0), 1)
        self._put(self._action(), 1)
        # Every set, offset boundary, both ways, hits, and data classes.
        for set_idx in range(4):
            base = set_idx << 4
            self._txn(0, base + (0, 4, 8, 15)[set_idx], stall=(set_idx == 0))
            self._txn(0, base)
            self._txn(1, base, (0, 0xFFFFFFFF, 0xAAAAAAAA, 0x12345678)[set_idx])
            self._txn(0, base)
            self._txn(0, base + 0x40)  # same set, second way

        # Third tags force clean and dirty conflict evictions in all sets.
        for set_idx in range(4):
            self._txn(0, (set_idx << 4) + 0x80)
            self._txn(1, (set_idx << 4) + 0xC0, 0xAAAAAAAA)

        # Invalidate hit, miss, then refill of the invalidated line.
        self._txn(2, 0xC0)
        self._txn(2, 0xC0)
        self._txn(0, 0xC0)

        # Hidden poison class (default validation secret).
        self._txn(0, 0x2A << 6)

        # Fill every slot dirty so dirty-count half/full boundaries are seen.
        for set_idx in range(4):
            self._txn(1, (0x30 << 6) | (set_idx << 4), 0xFFFFFFFF)
            self._txn(1, (0x31 << 6) | (set_idx << 4), 0)
        # Read a third tag through a dirty victim and hold mem_ready low long
        # enough to exercise WRITEBACK backpressure and recovery.
        self._txn(0, 0x32 << 6, stall=True)
        # Full flush, including a deliberate memory backpressure interval.
        self._txn(3, stall=True, wait=30)

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
        action[11] = 0.0 if step < 2 else 1.0
        return action


if __name__ == "__main__":
    agent = InferenceInterface(policy="greedy")
    assert agent.predict(np.zeros(75), 0, 100).shape == (16,)
    print("cache_ctrl_validation inference interface OK")
