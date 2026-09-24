#!/usr/bin/env python3
"""Black-box evaluation loop for watchdog_safety_validation."""

import importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def default_secrets():
    return {"service_key_a": 0xA5, "service_key_b": 0x5A,
            "key_gap": 3, "escalation_limit": 3, "reset_hold": 2}


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WatchdogSafetyHarness:
    DIMS = 16

    def __init__(self, secrets=None, backend="local"):
        if backend != "local":
            raise ValueError("watchdog_safety_validation ships only the local backend")
        self.backend = backend
        self.secrets = secrets or default_secrets()
        self._ls = _module("watchdog_validation_ls", HERE / "local_sim.py")
        cov = _module("watchdog_validation_cov", HERE / "coverage_simulator.py")
        self._cs = cov.CoverageSimulator(HERE / "dut" / "coverage_meta.json")
        self._dut = self._make_dut()

    def _make_dut(self):
        return self._ls.WatchdogSafetyValidation(**self.secrets)

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
            "reg_we": a[0] & 1, "reg_addr": a[1] & 3,
            "data": self._word(a[2:6]), "tick": a[6] & 1,
            "service": a[7] & 1, "fault_inject": a[8] & 1,
            "reset_n": a[9] & 1,
        })
        self._cs.step(self._dut.read_signals())
        state = self._state()
        return state, float(np.mean(state)), False, {
            "total_coverage": self.coverage * 100.0, "fsm_state": self._dut.state,
        }


def run_episode(agent, secrets=None, max_steps=1000, interval=50, backend="local"):
    harness = WatchdogSafetyHarness(secrets=secrets, backend=backend)
    state = harness.reset()
    curve = []
    for step in range(max_steps):
        state, _, _, _ = harness.step(agent.predict(state, step, max_steps))
        if step % interval == 0:
            curve.append((step, harness.coverage))
    curve.append((max_steps - 1, harness.coverage))
    return curve


if __name__ == "__main__":
    mod = _module("watchdog_validation_agent", HERE / "inference_interface.py")
    print(run_episode(mod.InferenceInterface(policy="greedy"), max_steps=300))
