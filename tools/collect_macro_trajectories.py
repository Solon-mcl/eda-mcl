#!/usr/bin/env python3
"""Collect macro-level transitions from the real public RTL simulators."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tools"))

from inference import InferenceInterface  # noqa: E402
from run_experiments import load_harness  # noqa: E402

DUTS = ("spi_master_public", "spi_xfer_public")


def selected_orders(limit: int):
    """Deterministic, position-balanced subset of all 24 permutations."""
    all_orders = list(itertools.permutations(range(4)))
    preferred = [
        (0, 3, 2, 1), (3, 0, 1, 2), (2, 1, 3, 0), (1, 2, 0, 3),
        (0, 1, 2, 3), (3, 2, 1, 0), (2, 3, 0, 1), (1, 0, 3, 2),
        (0, 2, 1, 3), (1, 3, 2, 0), (2, 0, 3, 1), (3, 1, 0, 2),
    ]
    orders = preferred + [order for order in all_orders if order not in preferred]
    return orders[:max(1, min(int(limit), len(orders)))]


def collect_one(dut: str, order, steps: int, backend: str):
    _, harness_cls, base = load_harness(dut, backend)
    harness = harness_cls(backend=backend)
    state = harness.reset()
    agent = InferenceInterface(str(base / "dut/dut_spec.md"),
                               str(base / "dut/covergroup.svh"),
                               macro_order=order)
    policy = agent._policy
    transitions = []
    active = None
    seen_events = 0
    for step in range(steps):
        state_before = state.copy()
        action = agent.predict(state, step, steps)
        if len(policy.macro_trace) > seen_events:
            if active is not None:
                active["next_state"] = state_before.astype(int).tolist()
                active["reward_bins"] = int(np.sum(state_before) -
                                             np.sum(active.pop("_state_array")))
                active["duration"] = int(step - active["step"])
                active["done"] = False
                transitions.append(active)
            active = {
                "dut": dut,
                "order": list(order),
                "action": int(policy.macro_trace[-1]),
                "step": int(step),
                "state": state_before.astype(int).tolist(),
                "_state_array": state_before.copy(),
            }
            seen_events = len(policy.macro_trace)
        state, _, _, _ = harness.step(action)
    if active is not None:
        active["next_state"] = state.astype(int).tolist()
        active["reward_bins"] = int(np.sum(state) -
                                     np.sum(active.pop("_state_array")))
        active["duration"] = int(steps - active["step"])
        active["done"] = True
        transitions.append(active)
    record = {
        "dut": dut,
        "order": list(order),
        "steps": steps,
        "covered_bins": int(np.sum(state)),
        "total_bins": int(state.size),
        "transitions": transitions,
    }
    if hasattr(harness._dut, "close"):
        harness._dut.close()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("local", "verilator"),
                        default="verilator")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--orders", type=int, default=8)
    parser.add_argument("--output", default="results/macro_trajectories.json")
    args = parser.parse_args()
    records = []
    for dut in DUTS:
        for index, order in enumerate(selected_orders(args.orders), 1):
            record = collect_one(dut, order, args.steps, args.backend)
            records.append(record)
            print(f"{dut} {index}/{args.orders} order={order} "
                  f"coverage={record['covered_bins']}/{record['total_bins']}",
                  flush=True)
    payload = {
        "backend": args.backend,
        "steps": args.steps,
        "orders_per_dut": args.orders,
        "records": records,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output} with "
          f"{sum(len(r['transitions']) for r in records)} transitions")


if __name__ == "__main__":
    main()
