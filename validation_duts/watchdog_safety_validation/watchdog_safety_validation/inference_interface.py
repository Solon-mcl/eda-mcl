#!/usr/bin/env python3
"""Random and reachability baselines for the held-out safety watchdog."""

from collections import deque
import json
import os
import numpy as np


class InferenceInterface:
    DIMS = 16
    BOUNDS = np.asarray([2, 4, 256, 256, 256, 256, 2, 2, 2, 2,
                         1, 1, 1, 1, 1, 1], dtype=np.float32)

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
        return 59

    @staticmethod
    def _bytes(value):
        return [(int(value) >> (8 * i)) & 0xFF for i in range(4)]

    def _action(self, we=0, addr=0, data=0, tick=0, service=0,
                fault=0, reset=1):
        return np.asarray([we, addr, *self._bytes(data), tick, service, fault,
                           reset, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    def _put(self, action, cycles=1):
        for _ in range(max(1, int(cycles))):
            self._program.append(action.copy())

    def _key_pair(self, a=0xA5, b=0x5A):
        self._put(self._action(data=a, service=1))
        self._put(self._action(data=b, service=1))

    def _build_program(self):
        self._put(self._action(reset=0))
        self._put(self._action(we=1, addr=1, data=3))
        self._put(self._action(we=1, addr=2, data=8))
        self._put(self._action(we=1, addr=0, data=1))
        self._put(self._action(we=1, addr=1, data=4))  # rejected while active

        self._put(self._action(tick=1))
        self._key_pair()                               # early service, fault 1
        self._put(self._action(tick=1), 2)             # enter open window
        self._key_pair()                               # accepted, reset counter

        self._put(self._action(tick=1), 3)
        self._put(self._action(data=0x11, service=1))  # bad key, fault 2
        self._put(self._action(tick=1), 3)             # enter pretimeout
        self._key_pair()                               # legal but late service

        self._put(self._action(tick=1), 8)             # timeout -> pending reset
        self._put(self._action(), 3)                   # pending -> disabled

        self._put(self._action(we=1, addr=0, data=3))  # enable + config lock
        self._put(self._action(we=1, addr=1, data=5))  # closed-state reject
        self._put(self._action(fault=1))               # threshold -> safety lock
        self._put(self._action(we=1, addr=2, data=9))  # locked-state reject
        self._put(self._action(reset=0))               # external recovery

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
        action[9] = 0.0 if step < 2 else 1.0
        return action


if __name__ == "__main__":
    agent = InferenceInterface(policy="greedy")
    assert agent.predict(np.zeros(59), 0, 100).shape == (16,)
    print("watchdog_safety_validation inference interface OK")
