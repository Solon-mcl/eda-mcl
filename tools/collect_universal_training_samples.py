#!/usr/bin/env python3
"""Collect option/outcome samples from public DUTs using UniversalPolicy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from inference.universal_training import samples_from_policy  # noqa: E402
from tools.run_experiments import PUBLIC_DUTS, load_harness, make_agent  # noqa: E402


def collect(dut, steps, seed, backend):
    _, harness_cls, base = load_harness(dut, backend)
    harness = harness_cls(backend=backend)
    state = harness.reset()
    agent = make_agent("full", base, seed)
    for step in range(steps):
        action = agent.predict(state, step, steps)
        state, _, _, _ = harness.step(action)
    samples = samples_from_policy(agent._policy, dut, int(state.size))
    if hasattr(harness._dut, "close"):
        harness._dut.close()
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--backend", choices=("local", "verilator"),
                        default="verilator")
    parser.add_argument("--output", default="training/universal_public_v1.jsonl")
    args = parser.parse_args()
    samples = []
    for seed_offset in range(max(1, args.seeds)):
        for dut in PUBLIC_DUTS:
            samples.extend(collect(dut, args.steps, 260923 + seed_offset,
                                   args.backend))
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sample.to_json() for sample in samples) + "\n",
                    encoding="utf-8")
    positives = sum(sample.direct_reward > 0 for sample in samples)
    print(f"samples={len(samples)} positive={positives} output={path}")


if __name__ == "__main__":
    main()
