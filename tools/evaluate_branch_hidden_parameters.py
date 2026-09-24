#!/usr/bin/env python3
"""Frozen hidden-parameter evaluation for the branch validation DUT."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tools"))

from inference import InferenceInterface  # noqa: E402
from run_experiments import load_harness  # noqa: E402


CASES = [
    ("default", {"history_bits": 4, "index_salt": 9,
                 "replacement_xor": 1, "ras_depth": 4}),
    ("h4_s0_r0_d8", {"history_bits": 4, "index_salt": 0,
                     "replacement_xor": 0, "ras_depth": 8}),
    ("h4_s15_r1_d6", {"history_bits": 4, "index_salt": 15,
                      "replacement_xor": 1, "ras_depth": 6}),
    ("h5_s3_r0_d4", {"history_bits": 5, "index_salt": 3,
                     "replacement_xor": 0, "ras_depth": 4}),
    ("h5_s12_r1_d8", {"history_bits": 5, "index_salt": 12,
                      "replacement_xor": 1, "ras_depth": 8}),
    ("h6_s0_r1_d4", {"history_bits": 6, "index_salt": 0,
                     "replacement_xor": 1, "ras_depth": 4}),
    ("h6_s7_r0_d6", {"history_bits": 6, "index_salt": 7,
                     "replacement_xor": 0, "ras_depth": 6}),
    ("h6_s15_r1_d8", {"history_bits": 6, "index_salt": 15,
                      "replacement_xor": 1, "ras_depth": 8}),
]


class SchemaRandom:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
        self.bounds = np.asarray(
            [2, *([256] * 4), 4, 2, *([256] * 4), 2, 2, 2, 1, 1],
            dtype=np.float32)

    def predict(self, state, step, max_steps):
        action = (self.rng.uniform(size=16) * self.bounds).astype(np.float32)
        action[13] = 0 if step < 2 else 1
        return action


def run_case(name, secrets, steps, interval, kind, seed):
    _, harness_cls, base = load_harness("branch_predictor_validation", "local")
    harness = harness_cls(secrets=secrets, backend="local")
    state = harness.reset()
    agent = (InferenceInterface(str(base / "dut/dut_spec.md"),
                                str(base / "dut/covergroup.svh"))
             if kind == "full" else SchemaRandom(seed))
    curve = []
    timings = []
    started = time.perf_counter()
    for step in range(steps):
        tick = time.perf_counter_ns()
        action = agent.predict(state, step, steps)
        timings.append(time.perf_counter_ns() - tick)
        state, _, _, _ = harness.step(action)
        if step % interval == 0:
            curve.append([step, float(np.mean(state))])
    curve.append([steps - 1, float(np.mean(state))])
    missing = []
    flat = 0
    for cp in harness._cs.coverpoints:
        for item in cp.get("bins", []):
            if state[flat] == 0:
                missing.append(f"{cp['name']}.{item['name']}")
            flat += 1
    auc = float(np.trapz([p[1] for p in curve], [p[0] for p in curve]) /
                max(1, steps - 1))
    return {
        "case": name, "agent": kind, "secrets": secrets, "steps": steps,
        "covered_bins": int(np.sum(state)), "total_bins": int(state.size),
        "coverage": float(np.mean(state)), "auc": auc,
        "missing_bins": missing, "curve": curve,
        "elapsed_seconds": time.perf_counter() - started,
        "predict_mean_ms": float(np.mean(timings) / 1e6),
        "predict_p99_ms": float(np.percentile(timings, 99) / 1e6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260923)
    parser.add_argument("--output",
                        default="results/branch_hidden_parameter_sweep.json")
    args = parser.parse_args()
    records = []
    for index, (name, secrets) in enumerate(CASES):
        for kind in ("full", "random"):
            records.append(run_case(name, secrets, args.steps, args.interval,
                                    kind, args.seed + index))
    summary = {}
    for kind in ("full", "random"):
        subset = [item for item in records if item["agent"] == kind]
        coverage = np.asarray([item["coverage"] for item in subset])
        auc = np.asarray([item["auc"] for item in subset])
        summary[kind] = {
            "mean_coverage": float(np.mean(coverage)),
            "std_coverage": float(np.std(coverage)),
            "min_coverage": float(np.min(coverage)),
            "max_coverage": float(np.max(coverage)),
            "mean_auc": float(np.mean(auc)),
        }
    report = {"cases": records, "summary": summary}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
