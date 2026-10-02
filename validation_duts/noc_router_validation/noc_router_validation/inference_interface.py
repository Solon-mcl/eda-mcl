#!/usr/bin/env python3
"""Random and reachability baselines for the held-out NoC router."""

from collections import deque
import json
import os
import numpy as np


class InferenceInterface:
    DIMS = 16
    BOUNDS = np.asarray([2, 5, 3, 3, 2, 4, 256, 256, 256, 256,
                         32, 32, 2, 2, 1, 1], dtype=np.float32)

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
        return 75

    @staticmethod
    def _bytes(value):
        return [(int(value) >> (8 * i)) & 0xFF for i in range(4)]

    def _action(self, valid=0, port=0, x=1, y=1, vc=0, flit=3,
                payload=0, credits=31, congestion=0, stall=0, reset=1):
        return np.asarray([valid, port, x, y, vc, flit, *self._bytes(payload),
                           credits, congestion, stall, reset, 0, 0],
                          dtype=np.float32)

    def _put(self, action, cycles=1):
        for _ in range(max(1, int(cycles))):
            self._program.append(action.copy())

    def _send(self, port, x, y, vc=0, flit=3, congestion=0, credits=31):
        self._put(self._action(1, port, x, y, vc, flit,
                               0xA5000000 | port, credits, congestion))
        self._put(self._action(credits=credits, congestion=congestion))

    def _build_program(self):
        self._put(self._action(reset=0))
        self._put(self._action())
        # Directed port matrix and all flit types.
        cases = [(0,1,2,0), (0,2,1,1), (0,1,0,2), (0,0,1,3),
                 (1,2,1,3), (2,1,2,3), (3,0,1,3), (4,1,0,3),
                 (1,1,1,3), (0,1,1,3)]
        for port, x, y, flit in cases:
            self._send(port, x, y, vc=0, flit=flit)
        self._send(0, 1, 1, vc=1, flit=3)
        self._send(0, 2, 1, vc=1, flit=3)
        self._put(self._action(congestion=(1 << 2)))

        # Exercise both VC roles so this remains valid when the hidden escape VC flips.
        heavy_congestion = (1 << 1) | (1 << 2) | (1 << 3)
        self._send(0, 2, 2, vc=0, flit=3, congestion=heavy_congestion)
        self._send(0, 2, 2, vc=1, flit=3, congestion=heavy_congestion)
        self._send(0, 2, 2, vc=0, flit=3, congestion=0)
        self._send(0, 2, 2, vc=1, flit=3, congestion=0)

        # Credit loss and later recovery.
        self._put(self._action(1, 0, 2, 1, 0, 3, credits=0))
        self._put(self._action(credits=0), 32)
        self._put(self._action(credits=31))
        self._put(self._action(credits=(1 << 2) | (1 << 0)))

        # All five inputs request EAST while draining is frozen, then all resolve.
        self._put(self._action(stall=1))
        self._put(self._action(1, 0, 2, 1, 0, 3, stall=1))
        self._put(self._action(1, 1, 2, 1, 0, 3, stall=1))
        self._put(self._action(1, 2, 2, 1, 0, 3, stall=1))
        self._put(self._action(1, 3, 2, 1, 0, 3, stall=1))
        self._put(self._action(1, 4, 2, 1, 0, 3, stall=1))
        self._put(self._action(), 5)

        # Fill one input FIFO and force overflow at the hidden default depth.
        for _ in range(6):
            self._put(self._action(1, 4, 1, 0, 0, 3, stall=1))
        self._put(self._action(), 6)

        # Ordered multi-flit packet while stalled, then forward each flit.
        self._put(self._action(1, 3, 0, 1, 0, 0, stall=1))
        self._put(self._action(1, 3, 0, 1, 0, 1, stall=1))
        self._put(self._action(1, 3, 0, 1, 0, 2, stall=1))
        self._put(self._action(), 2)
        self._put(self._action(credits=0), 8)
        self._put(self._action(), 2)

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
    assert agent.predict(np.zeros(75), 0, 100).shape == (16,)
    print("noc_router_validation inference interface OK")
