#!/usr/bin/env python3
"""Evaluate a frozen candidate and random baseline across hidden parameters.

This script is evaluation-only: it does not import the validation oracle and
does not update candidate weights, policies, prompts, or model artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# numpy>=2.0 renamed trapz to trapezoid; keep both working.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tools"))

from inference import InferenceInterface  # noqa: E402
from run_experiments import load_harness  # noqa: E402


CASES = [
    ("default", {"replacement_xor": 1, "memory_latency": 3,
                 "poison_tag": 0x2A}),
    ("xor0_fast_low", {"replacement_xor": 0, "memory_latency": 1,
                       "poison_tag": 0}),
    ("xor1_fast_high", {"replacement_xor": 1, "memory_latency": 1,
                        "poison_tag": 0x3FFFFFF}),
    ("xor0_mid_a", {"replacement_xor": 0, "memory_latency": 4,
                    "poison_tag": 0x1555555}),
    ("xor1_mid_b", {"replacement_xor": 1, "memory_latency": 5,
                    "poison_tag": 0x2AAAAAA}),
    ("xor0_slow_low", {"replacement_xor": 0, "memory_latency": 8,
                       "poison_tag": 1}),
    ("xor1_slow_mid", {"replacement_xor": 1, "memory_latency": 8,
                       "poison_tag": 0x123456}),
    ("xor0_slow_high", {"replacement_xor": 0, "memory_latency": 8,
                        "poison_tag": 0x3FFFFFE}),
]


class SchemaRandom:
    """Fair random baseline derived only from the published action schema."""

    def __init__(self, seed: int):
        self.rng = np.random.RandomState(seed)
        self.bounds = np.asarray(
            [2, 4, *([256] * 8), 2, 2, 1, 1, 1, 1], dtype=np.float32)

    def predict(self, coverage_state, step, max_steps):
        action = (self.rng.uniform(size=16) * self.bounds).astype(np.float32)
        action[11] = 0.0 if step < 2 else 1.0
        return action


def bin_names(harness, state):
    missing = []
    index = 0
    for coverpoint in harness._cs.coverpoints:
        for item in coverpoint.get("bins", []):
            if state[index] == 0:
                missing.append(f"{coverpoint['name']}.{item['name']}")
            index += 1
    return missing


def run_case(name, secrets, steps, interval, agent_kind, seed):
    _, harness_cls, base = load_harness("cache_ctrl_validation", "local")
    harness = harness_cls(secrets=secrets, backend="local")
    state = harness.reset()
    if agent_kind == "full":
        agent = InferenceInterface(str(base / "dut/dut_spec.md"),
                                   str(base / "dut/covergroup.svh"))
    else:
        agent = SchemaRandom(seed)
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
    auc = float(_trapz([point[1] for point in curve],
                         [point[0] for point in curve]) / max(1, steps - 1))
    return {
        "case": name,
        "agent": agent_kind,
        "secrets": secrets,
        "steps": steps,
        "covered_bins": int(np.sum(state)),
        "total_bins": int(state.size),
        "coverage": float(np.mean(state)),
        "auc": auc,
        "missing_bins": bin_names(harness, state),
        "curve": curve,
        "elapsed_seconds": time.perf_counter() - started,
        "predict_mean_ms": float(np.mean(timings) / 1e6),
        "predict_p99_ms": float(np.percentile(timings, 99) / 1e6),
        "llm_status": getattr(agent, "llm_status", None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260923)
    parser.add_argument("--output",
                        default="results/cache_hidden_parameter_sweep.json")
    args = parser.parse_args()
    records = []
    for index, (name, secrets) in enumerate(CASES):
        for agent_kind in ("full", "random"):
            records.append(run_case(name, secrets, args.steps, args.interval,
                                    agent_kind, args.seed + index))
    summary = {}
    for agent_kind in ("full", "random"):
        subset = [item for item in records if item["agent"] == agent_kind]
        values = np.asarray([item["coverage"] for item in subset])
        aucs = np.asarray([item["auc"] for item in subset])
        summary[agent_kind] = {
            "mean_coverage": float(np.mean(values)),
            "std_coverage": float(np.std(values)),
            "min_coverage": float(np.min(values)),
            "max_coverage": float(np.max(values)),
            "mean_auc": float(np.mean(aucs)),
        }
    report = {"cases": records, "summary": summary}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
