#!/usr/bin/env python3
"""Black-box evaluation loop for cache_ctrl_validation."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def default_secrets():
    return {"replacement_xor": 1, "memory_latency": 3, "poison_tag": 0x2A}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CacheCtrlHarness:
    DIMS = 16

    def __init__(self, secrets=None, backend="local"):
        if backend != "local":
            raise ValueError("cache_ctrl_validation ships only the local backend")
        self.backend = backend
        self.secrets = secrets or default_secrets()
        self._ls = _load_module("cache_validation_ls", HERE / "local_sim.py")
        cov = _load_module("cache_validation_cov", HERE / "coverage_simulator.py")
        self._cs = cov.CoverageSimulator(HERE / "dut" / "coverage_meta.json")
        self._dut = self._make_dut()

    def _make_dut(self):
        return self._ls.CacheCtrlValidation(**self.secrets)

    def reset(self):
        self._dut = self._make_dut()
        self._cs.reset()
        return self._state()

    def _state(self):
        return np.asarray(self._cs.get_bin_vector(), dtype=np.float32)

    @property
    def total_bins(self):
        return self._cs.total_bins

    @property
    def coverage(self):
        return self._cs.coverage

    @staticmethod
    def _word(lanes):
        values = [int(value) & 0xFF for value in lanes]
        return sum(values[i] << (8 * i) for i in range(4))

    def step(self, stimulus):
        action = np.asarray(stimulus, dtype=np.float32).reshape(-1)
        if action.size < self.DIMS:
            action = np.pad(action, (0, self.DIMS - action.size))
        a = [int(value) for value in action]
        self._dut.step({
            "req_valid": a[0] & 1,
            "opcode": a[1] & 3,
            "addr": self._word(a[2:6]),
            "wdata": self._word(a[6:10]),
            "mem_ready": a[10] & 1,
            "reset_n": a[11] & 1,
        })
        self._cs.step(self._dut.read_signals())
        state = self._state()
        return state, float(np.mean(state)), False, {
            "total_coverage": 100.0 * self.coverage,
            "fsm_state": self._dut.state,
        }


def run_episode(agent, secrets=None, max_steps=4000, interval=200, backend="local"):
    harness = CacheCtrlHarness(secrets=secrets, backend=backend)
    state = harness.reset()
    curve = []
    for step in range(max_steps):
        action = agent.predict(state, step, max_steps)
        state, _, _, _ = harness.step(action)
        if step % interval == 0:
            curve.append((step, harness.coverage))
    curve.append((max_steps - 1, harness.coverage))
    return curve


if __name__ == "__main__":
    interface = _load_module("cache_validation_agent", HERE / "inference_interface.py")
    agent = interface.InferenceInterface(policy="greedy")
    print(run_episode(agent, max_steps=2000, interval=200))
