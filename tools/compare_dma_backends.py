#!/usr/bin/env python3
"""Replay one fixed M5 action stream on DMA local and Verilator backends.

The action stream is generated once against the local backend, then replayed
unchanged from reset on both implementations.  The report keeps compact
windows around the first signal and coverage divergences instead of dumping
every cycle.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DMA = ROOT / "public_duts" / "dma_xfer_public" / "dma_xfer_public"
sys.path.insert(0, str(ROOT / "app"))

from inference import InferenceInterface  # noqa: E402


SIGNALS = (
    "ch0_state", "ch1_state", "ch2_state", "ch3_state",
    "arb_winner", "arb_valid", "arb_conflict", "active_ch",
    "dir_dir", "cfg_burst", "cfg_saddr", "cfg_daddr", "cfg_len",
    "hold", "done_ok_valid", "done_ok",
    "seq1", "seq2", "seq3", "seq4", "seq5", "seq6",
    "seq7", "seq8", "seq9", "seq10", "seq11", "seq12",
)


def load_harness_class():
    spec = importlib.util.spec_from_file_location("dma_compare_harness",
                                                  DMA / "harness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DmaXferHarness


def bin_names(harness):
    return [f"{cp['name']}.{item['name']}"
            for cp in harness._cs.coverpoints for item in cp.get("bins", [])]


def signal_snapshot(harness):
    raw = harness._dut.read_signals() or {}
    return {name: int(raw.get(name, 0)) for name in SIGNALS}


def generate_actions(harness_cls, steps, seed):
    os.environ["EDA_STIMULUS_SEED"] = str(int(seed))
    harness = harness_cls(backend="local")
    state = np.asarray(harness.reset(), dtype=np.float32)
    agent = InferenceInterface(str(DMA / "dut" / "dut_spec.md"),
                               str(DMA / "dut" / "covergroup.svh"))
    actions = []
    for step in range(steps):
        action = np.asarray(agent.predict(state, step, steps),
                            dtype=np.float32).reshape(-1)
        actions.append(action.copy())
        state, _, _, _ = harness.step(action)
        state = np.asarray(state, dtype=np.float32)
    if hasattr(harness._dut, "close"):
        harness._dut.close()
    return actions


def compact_cycle(record):
    return {
        "cycle": record["cycle"],
        "action": record["action"],
        "differing_signals": record["differing_signals"],
        "local_signals": record["local_signals"],
        "verilator_signals": record["verilator_signals"],
        "local_new_bins": record["local_new_bins"],
        "verilator_new_bins": record["verilator_new_bins"],
        "coverage_only_local": record["coverage_only_local"],
        "coverage_only_verilator": record["coverage_only_verilator"],
    }


def window(records, center, radius):
    if center is None:
        return []
    lo = max(0, center - 1 - radius)
    hi = min(len(records), center + radius)
    return [compact_cycle(item) for item in records[lo:hi]]


def compare(steps, seed, radius):
    harness_cls = load_harness_class()
    actions = generate_actions(harness_cls, steps, seed)
    local = harness_cls(backend="local")
    verilator = harness_cls(backend="verilator")
    local_state = np.asarray(local.reset(), dtype=np.float32)
    verilator_state = np.asarray(verilator.reset(), dtype=np.float32)
    names = bin_names(local)

    records = []
    first_signal = None
    first_coverage = None
    first_new_bin = None
    mismatch_cycles = Counter()
    first_signal_cycle = {}
    previous_local = local_state.copy()
    previous_verilator = verilator_state.copy()

    try:
        for index, action in enumerate(actions):
            cycle = index + 1
            local_state, _, _, _ = local.step(action)
            verilator_state, _, _, _ = verilator.step(action)
            local_state = np.asarray(local_state, dtype=np.float32)
            verilator_state = np.asarray(verilator_state, dtype=np.float32)
            local_signals = signal_snapshot(local)
            verilator_signals = signal_snapshot(verilator)
            differing = [name for name in SIGNALS
                          if local_signals[name] != verilator_signals[name]]
            for name in differing:
                mismatch_cycles[name] += 1
                first_signal_cycle.setdefault(name, cycle)
            if differing and first_signal is None:
                first_signal = cycle

            local_new_idx = np.flatnonzero(
                (local_state > 0.5) & (previous_local <= 0.5)).tolist()
            verilator_new_idx = np.flatnonzero(
                (verilator_state > 0.5) & (previous_verilator <= 0.5)).tolist()
            coverage_diff = np.flatnonzero(
                (local_state > 0.5) != (verilator_state > 0.5)).tolist()
            if coverage_diff and first_coverage is None:
                first_coverage = cycle
            if set(local_new_idx) != set(verilator_new_idx) and first_new_bin is None:
                first_new_bin = cycle

            records.append({
                "cycle": cycle,
                "action": [int(value) for value in action],
                "differing_signals": differing,
                "local_signals": local_signals,
                "verilator_signals": verilator_signals,
                "local_new_bins": [names[i] for i in local_new_idx],
                "verilator_new_bins": [names[i] for i in verilator_new_idx],
                "coverage_only_local": [names[i] for i in coverage_diff
                                        if local_state[i] > 0.5],
                "coverage_only_verilator": [names[i] for i in coverage_diff
                                            if verilator_state[i] > 0.5],
            })
            previous_local = local_state.copy()
            previous_verilator = verilator_state.copy()
    finally:
        for harness in (local, verilator):
            if hasattr(harness._dut, "close"):
                harness._dut.close()

    local_hit = set(np.flatnonzero(local_state > 0.5).tolist())
    verilator_hit = set(np.flatnonzero(verilator_state > 0.5).tolist())
    return {
        "steps": steps,
        "seed": seed,
        "action_source": ("current policy driven by local coverage, then "
                          "fixed replay on both backends"),
        "first_signal_divergence_cycle": first_signal,
        "first_new_bin_divergence_cycle": first_new_bin,
        "first_accumulated_coverage_divergence_cycle": first_coverage,
        "local_covered_bins": len(local_hit),
        "verilator_covered_bins": len(verilator_hit),
        "coverage_only_local": [names[i] for i in sorted(local_hit - verilator_hit)],
        "coverage_only_verilator": [names[i] for i in sorted(verilator_hit - local_hit)],
        "signal_mismatch_summary": [
            {"signal": name, "first_cycle": first_signal_cycle[name],
             "mismatch_cycles": count}
            for name, count in sorted(mismatch_cycles.items(),
                                      key=lambda item: (-item[1], item[0]))
        ],
        "key_signal_first_windows": {
            name: window(records, first_signal_cycle.get(name), radius)
            for name in ("cfg_len", "ch0_state", "hold", "done_ok_valid",
                         "cfg_burst", "dir_dir")
            if name in first_signal_cycle
        },
        "first_signal_divergence_window": window(records, first_signal, radius),
        "first_new_bin_divergence_window": window(records, first_new_bin, radius),
        "first_coverage_divergence_window": window(records, first_coverage, radius),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=260923)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--output",
                        default="results/dma_backend_differential_5k.json")
    args = parser.parse_args()
    if args.steps < 1 or args.window < 0:
        parser.error("--steps must be positive and --window non-negative")
    report = compare(args.steps, args.seed, args.window)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if not key.endswith("_window")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
