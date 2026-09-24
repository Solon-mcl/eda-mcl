#!/usr/bin/env python3
"""Black-box evaluation loop for branch_predictor_validation."""

import importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def default_secrets():
    return {"history_bits": 4, "index_salt": 9,
            "replacement_xor": 1, "ras_depth": 4}


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BranchPredictorHarness:
    DIMS = 16

    def __init__(self, secrets=None, backend="local"):
        if backend != "local":
            raise ValueError("branch_predictor_validation ships only the local backend")
        self.backend = backend
        self.secrets = secrets or default_secrets()
        self._ls = _module("branch_validation_ls", HERE / "local_sim.py")
        cov = _module("branch_validation_cov", HERE / "coverage_simulator.py")
        self._cs = cov.CoverageSimulator(HERE / "dut" / "coverage_meta.json")
        self._dut = self._make_dut()

    def _make_dut(self):
        return self._ls.BranchPredictorValidation(**self.secrets)

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
        return sum((int(value) & 0xFF) << (8 * i) for i, value in enumerate(lanes))

    def step(self, stimulus):
        action = np.asarray(stimulus, dtype=np.float32).reshape(-1)
        if action.size < self.DIMS:
            action = np.pad(action, (0, self.DIMS - action.size))
        a = [int(value) for value in action]
        self._dut.step({
            "valid": a[0] & 1, "pc": self._word(a[1:5]), "kind": a[5] & 3,
            "actual_taken": a[6] & 1, "target": self._word(a[7:11]),
            "stall": a[11] & 1, "flush": a[12] & 1, "reset_n": a[13] & 1,
        })
        self._cs.step(self._dut.read_signals())
        state = self._state()
        return state, float(np.mean(state)), False, {
            "total_coverage": self.coverage * 100.0,
            "outcome": self._dut.outcome,
        }


def run_episode(agent, secrets=None, max_steps=2000, interval=100, backend="local"):
    harness = BranchPredictorHarness(secrets=secrets, backend=backend)
    state = harness.reset()
    curve = []
    for step in range(max_steps):
        state, _, _, _ = harness.step(agent.predict(state, step, max_steps))
        if step % interval == 0:
            curve.append((step, harness.coverage))
    curve.append((max_steps - 1, harness.coverage))
    return curve


if __name__ == "__main__":
    mod = _module("branch_validation_agent", HERE / "inference_interface.py")
    print(run_episode(mod.InferenceInterface(policy="greedy"), max_steps=1000))
