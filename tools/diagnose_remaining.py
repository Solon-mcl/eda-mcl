#!/usr/bin/env python3
"""Summarize raw RTL events related to the remaining public coverage holes."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ["DEEPSEEK_ENABLED"] = "0"

from inference import InferenceInterface  # noqa: E402
from run_experiments import load_harness  # noqa: E402


def run(dut, steps=30000):
    _, harness_cls, base = load_harness(dut, "verilator")
    harness = harness_cls(backend="verilator")
    state = harness.reset()
    agent = InferenceInterface(str(base / "dut/dut_spec.md"),
                               str(base / "dut/covergroup.svh"))
    summary = {"dut": dut, "raw_events": {}, "samples": []}
    counters = Counter()
    last_trace = 0
    last_ss = None
    for step in range(steps):
        action = agent.predict(state, step, steps)
        state, _, _, _ = harness.step(action)
        signals = harness._dut.read_signals()
        trace = getattr(agent._policy, "macro_trace", [])
        if len(trace) != last_trace:
            summary["samples"].append({"step": step, "macro": int(trace[-1])})
            last_trace = len(trace)
        if dut == "spi_master_public":
            risr = int(signals.get("cov_risr", 0))
            s0 = int(signals.get("cov_s0", 0))
            state_id = int(signals.get("cov_fsm_state_id", 0))
            ss = int(action[5])
            if ss != last_ss and step < 7000 and len(summary["samples"]) < 80:
                summary["samples"].append({
                    "step": step, "ss": ss, "fsm": state_id,
                    "protocol": int(signals.get("protocol", -1)),
                    "busy": int(signals.get("cov_fsm_busy", 0)),
                })
            last_ss = ss
            if signals.get("cov_rx_underflow", 0):
                counters[f"rxu_risr_{risr}"] += 1
            if s0:
                counters[f"s0_state_{state_id}"] += 1
                if len(summary["samples"]) < 80:
                    summary["samples"].append({
                        "step": step, "s0": 1, "fsm": state_id,
                        "frame": int(signals.get("cov_frame_cnt", 0)),
                        "hold": int(signals.get("cov_hold_ss_cnt", 0)),
                        "last": int(signals.get("cov_last_frame", 0)),
                    })
        else:
            if signals.get("done_ok_valid", 0):
                cls = int(signals.get("done_ok", -1))
                counters[f"done_class_{cls}"] += 1
                if len(summary["samples"]) < 80:
                    summary["samples"].append({
                        "step": step, "done_class": cls,
                        "active_ch": int(signals.get("active_ch", -1)),
                        "probe_mode": int(getattr(agent._policy, "_probe_mode", -1)),
                        "hold": int(signals.get("hold", 0)),
                    })
            if signals.get("arb_valid", 0):
                counters[f"winner_{int(signals.get('arb_winner', -1))}"] += 1
    summary["raw_events"] = dict(counters)
    summary["covered_bins"] = int(np.sum(state))
    summary["total_bins"] = int(state.size)
    if hasattr(harness._dut, "close"):
        harness._dut.close()
    return summary


def main():
    result = [run("dma_xfer_public"), run("spi_master_public")]
    output = ROOT / "results/remaining_signal_diagnostics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
