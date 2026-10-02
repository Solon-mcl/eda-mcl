#!/usr/bin/env python3
"""Black-box evaluation loop for dma_desc_engine_validation."""

import importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def default_secrets():
    return {"desc_base": 0x100, "corrupt_xor": 0, "chain_limit": 4,
            "ack_latency": 2, "retry_limit": 3}


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DmaDescEngineHarness:
    DIMS = 16

    def __init__(self, secrets=None, backend="local"):
        if backend != "local":
            raise ValueError(
                "dma_desc_engine_validation ships only the local backend")
        self.backend = backend
        self.secrets = secrets or default_secrets()
        self._ls = _module("dma_desc_validation_ls", HERE / "local_sim.py")
        cov = _module("dma_desc_validation_cov", HERE / "coverage_simulator.py")
        self._cs = cov.CoverageSimulator(HERE / "dut" / "coverage_meta.json")
        self._dut = self._make_dut()

    def _make_dut(self):
        return self._ls.DmaDescEngineValidation(**self.secrets)

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
        ptr = self._word(a[4:8])
        self._dut.step({
            "start": a[0] & 1,
            "fetch_valid": a[1] & 1,
            "word_we": a[2] & 1,
            "word_idx": a[3] & 3,
            "desc_ptr": ptr,
            "data": self._word(a[8:12]),
            "ack": a[12] & 1,
            "abort": a[13] & 1,
            "reset_n": a[14] & 1,
            "clear": a[15] & 1,
        })
        self._cs.step(self._dut.read_signals())
        state = self._state()
        return state, float(np.mean(state)), False, {
            "total_coverage": 100.0 * self.coverage,
            "fsm_state": self._dut.state,
            "fail_class": self._dut.fail_class,
            "chain_heads": [entry["ptr"] for entry in self._dut._chain],
        }


def run_episode(agent, secrets=None, max_steps=4000, interval=200,
                backend="local"):
    harness = DmaDescEngineHarness(secrets=secrets, backend=backend)
    state = harness.reset()
    curve = []
    for step in range(max_steps):
        state, _, _, _ = harness.step(agent.predict(state, step, max_steps))
        if step % interval == 0:
            curve.append((step, harness.coverage))
    curve.append((max_steps - 1, harness.coverage))
    return curve


if __name__ == "__main__":
    interface = _module("dma_desc_validation_agent", HERE / "inference_interface.py")
    agent = interface.InferenceInterface(policy="greedy")
    print(run_episode(agent, max_steps=1200, interval=150))
