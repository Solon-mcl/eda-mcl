#!/usr/bin/env python3
"""Random and reachability baselines for the held-out ECC memory."""

from collections import deque
import json
import os
import numpy as np


class InferenceInterface:
    DIMS = 16
    BOUNDS = np.asarray([2, 2, 8, 256, 256, 256, 256, 2, 33, 33,
                         2, 2, 2, 1, 1, 1], dtype=np.float32)

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
        return 72

    @staticmethod
    def _bytes(value):
        return [(int(value) >> (8 * i)) & 0xFF for i in range(4)]

    def _action(self, write=0, read=0, addr=0, data=0, inject=0,
                bit_a=0, bit_b=0, scrub=0, stall=0, reset=1):
        return np.asarray([write, read, addr, *self._bytes(data), inject,
                           bit_a, bit_b, scrub, stall, reset, 0, 0, 0],
                          dtype=np.float32)

    def _put(self, action, cycles=1):
        for _ in range(max(1, int(cycles))):
            self._program.append(action.copy())

    def _write(self, addr, data=0):
        self._put(self._action(write=1, addr=addr, data=data))

    def _inject(self, addr, a, b=None):
        self._put(self._action(inject=1, addr=addr, bit_a=a,
                               bit_b=a if b is None else b))

    def _build_program(self):
        self._put(self._action(reset=0))
        values = (0, 0xFFFFFFFF, 0xAAAAAAAA, 0x12345678)
        for addr in range(8):
            self._write(addr, values[addr & 3])
        self._put(self._action(read=1, addr=0))

        # Single-error delayed correction with explicit backpressure.
        self._inject(0, 4)
        self._put(self._action(read=1, addr=0))
        self._put(self._action(stall=1))
        self._put(self._action(), 4)

        # Address/error crosses and DBE detection.
        self._inject(0, 4, 7)
        self._put(self._action(read=1, addr=0))
        self._write(0, 0)
        self._inject(1, 4)
        self._inject(1, 4, 7)
        self._write(1, 0xFFFFFFFF)
        self._inject(7, 4)
        self._inject(7, 4, 7)
        self._put(self._action(read=1, addr=7))

        # Hit every syndrome class with a single remapped bit, clearing each
        # word afterward so the injection remains single-error.
        for syndrome in range(4):
            physical = (syndrome - 5) & 3
            requested = physical ^ 5
            self._inject(2, requested)
            self._write(2, 0xAAAAAAAA)

        # Poison classification at the hidden default poison word.
        self._inject(6, 4, 7)
        self._put(self._action(read=1, addr=6))
        self._write(6, 0x12345678)

        # Default salted scrub order: cursor 0 -> address 5, cursor 1 -> 0.
        self._inject(5, 4)
        self._put(self._action(scrub=1, stall=1))
        self._put(self._action(scrub=1))
        self._inject(0, 4, 7)
        self._put(self._action(scrub=1))
        self._put(self._action(scrub=1), 6)

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
        action[12] = 0.0 if step < 2 else 1.0
        return action


if __name__ == "__main__":
    agent = InferenceInterface(policy="greedy")
    assert agent.predict(np.zeros(72), 0, 100).shape == (16,)
    print("ecc_memory_validation inference interface OK")
