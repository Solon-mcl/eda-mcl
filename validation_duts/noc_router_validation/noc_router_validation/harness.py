#!/usr/bin/env python3
"""Black-box evaluation loop for noc_router_validation."""

import importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def default_secrets():
    return {"xy_order": 1, "priority_seed": 2, "escape_vc": 1,
            "congestion_threshold": 1, "buffer_depth": 3}


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NocRouterHarness:
    DIMS = 16

    def __init__(self, secrets=None, backend="local"):
        if backend != "local":
            raise ValueError("noc_router_validation ships only the local backend")
        self.backend = backend
        self.secrets = secrets or default_secrets()
        self._ls = _module("noc_router_ls", HERE / "local_sim.py")
        cov = _module("noc_router_cov", HERE / "coverage_simulator.py")
        self._cs = cov.CoverageSimulator(HERE / "dut" / "coverage_meta.json")
        self._dut = self._make_dut()

    def _make_dut(self):
        return self._ls.NocRouterValidation(**self.secrets)

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
            "valid": a[0] & 1, "in_port": a[1] % 5,
            "dest_x": a[2] % 3, "dest_y": a[3] % 3,
            "vc": a[4] & 1, "flit_type": a[5] & 3,
            "payload": self._word(a[6:10]), "credit_mask": a[10] & 0x1F,
            "congestion_mask": a[11] & 0x1F,
            "router_stall": a[12] & 1, "reset_n": a[13] & 1,
        })
        self._cs.step(self._dut.read_signals())
        state = self._state()
        return state, float(np.mean(state)), False, {
            "total_coverage": self.coverage * 100.0,
        }


def run_episode(agent, secrets=None, max_steps=1000, interval=50, backend="local"):
    harness = NocRouterHarness(secrets=secrets, backend=backend)
    state = harness.reset()
    curve = []
    for step in range(max_steps):
        state, _, _, _ = harness.step(agent.predict(state, step, max_steps))
        if step % interval == 0:
            curve.append((step, harness.coverage))
    curve.append((max_steps - 1, harness.coverage))
    return curve


if __name__ == "__main__":
    mod = _module("noc_router_agent", HERE / "inference_interface.py")
    print(run_episode(mod.InferenceInterface(policy="greedy"), max_steps=500))
