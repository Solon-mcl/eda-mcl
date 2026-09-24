#!/usr/bin/env python3
"""Random and reachability baselines for the held-out branch predictor."""

from collections import deque
import json
import os
import numpy as np


class InferenceInterface:
    DIMS = 16
    BOUNDS = np.asarray([2, 256, 256, 256, 256, 4, 2, 256, 256, 256, 256,
                         2, 2, 2, 1, 1], dtype=np.float32)

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
        return 66

    @staticmethod
    def _bytes(value):
        return [(int(value) >> (8 * i)) & 0xFF for i in range(4)]

    def _action(self, valid=0, pc=0, kind=0, taken=0, target=0,
                stall=0, flush=0, reset=1):
        return np.asarray([valid, *self._bytes(pc), kind, taken,
                           *self._bytes(target), stall, flush, reset, 0, 0],
                          dtype=np.float32)

    def _branch(self, pc, kind, taken, target):
        self._program.append(self._action(1, pc, kind, taken, target))

    def _build_program(self):
        self._program.append(self._action(reset=0))
        # Backpressure and flush each followed by an accepted branch.
        self._program.append(self._action(stall=1))
        self._branch(0x00, 0, 0, 0x100)
        self._program.append(self._action(flush=1))
        self._branch(0x04, 0, 1, 0x104)

        # Broad conditional training covers all counter/actual combinations,
        # history classes, aliases, direction errors, and BTB target changes.
        patterns = (0, 0, 1, 1, 1, 0, 1, 0)
        for round_idx in range(6):
            for i in range(32):
                pc = i * 4
                taken = patterns[(i + round_idx) % len(patterns)]
                target = 0x400 + i * 8 + (4 if round_idx == 5 else 0)
                self._branch(pc, 0, taken, target)

        # Jump miss -> hit/correct -> changed-target mispredict.
        self._program.append(self._action(reset=0))
        self._branch(0x200, 1, 1, 0x900)
        self._branch(0x200, 1, 1, 0x900)
        self._branch(0x200, 1, 1, 0x904)

        # Three same-set calls force BTB replacement. Five nested calls cover
        # half/full stack levels and hidden-depth overflow under defaults.
        for pc in tuple(0x300 + 8 * i for i in range(10)):
            self._branch(pc, 2, 1, 0xA00 + pc)
        self._branch(0x348, 2, 1, 0xA00 + 0x348)

        # Empty-return direction miss (after reset), then matched call/return;
        # repeating the return PC supplies both BTB miss and hit cases.
        self._program.append(self._action(reset=0))
        self._branch(0x500, 3, 1, 0x1234)
        self._branch(0x600, 2, 1, 0x604)
        self._branch(0x500, 3, 1, 0x604)
        self._branch(0x604, 2, 1, 0x608)
        self._branch(0x500, 3, 1, 0x608)

        # Extra all-taken/all-not-taken runs make saturation robust to the
        # hidden history length and salt rather than only the public defaults.
        for taken in (1, 0):
            for repeat in range(96):
                pc = 0x800 + ((repeat * 4) & 0x7C)
                self._branch(pc, 0, taken, 0x1800 + pc)
        for repeat in range(16):
            pc = 0xA00 + ((repeat * 4) & 0x3C)
            self._branch(pc, 0, repeat & 1, 0x2000 + pc)
        # Once all-taken history saturates, a fixed PC maps repeatedly to the
        # same salted PHT entry and BTB entry for every supported history size.
        for _ in range(16):
            self._branch(0xB00, 0, 1, 0x2B00)

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
    assert agent.predict(np.zeros(66), 0, 100).shape == (16,)
    print("branch_predictor_validation inference interface OK")
