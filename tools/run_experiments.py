#!/usr/bin/env python3
"""Run repeatable public-DUT experiments and emit JSON/CSV summaries."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from inference import InferenceInterface  # noqa: E402

DUTS = {
    "dma_xfer_public": ("dma_xfer_public/dma_xfer_public", "DmaXferHarness"),
    "spi_master_public": ("spi_master_public/spi_master_public", "SpiMasterHarness"),
    "spi_xfer_public": ("spi_xfer_public/spi_xfer_public", "SpiXferHarness"),
    # Kept outside public_duts as a development-validation distribution. The
    # relative path preserves the existing loader without special-case imports.
    "cache_ctrl_validation": ("../validation_duts/cache_ctrl_validation/cache_ctrl_validation",
                              "CacheCtrlHarness"),
    "branch_predictor_validation": (
        "../validation_duts/branch_predictor_validation/branch_predictor_validation",
        "BranchPredictorHarness"),
    "watchdog_safety_validation": (
        "../validation_duts/watchdog_safety_validation/watchdog_safety_validation",
        "WatchdogSafetyHarness"),
}
PUBLIC_DUTS = ("dma_xfer_public", "spi_master_public", "spi_xfer_public")


def load_harness(dut, backend):
    # Each public package ships a same-named ``run_verilator`` module.  Purge
    # the previous DUT's module before switching packages in one process.
    sys.modules.pop("run_verilator", None)
    rel, klass = DUTS[dut]
    base = ROOT / "public_duts" / rel
    spec = importlib.util.spec_from_file_location(f"harness_{dut}", base / "harness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, getattr(module, klass), base


def make_agent(kind, base, seed, controller_model=None):
    if kind == "full":
        os.environ["EDA_STIMULUS_SEED"] = str(int(seed))
        return InferenceInterface(str(base / "dut/dut_spec.md"),
                                  str(base / "dut/covergroup.svh"),
                                  controller_path=controller_model)
    spec = importlib.util.spec_from_file_location(
        f"baseline_{base.name}_{kind}", base / "inference_interface.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InferenceInterface(
        str(base / "dut/dut_spec.md"), str(base / "dut/covergroup.svh"),
        policy=kind, seed=seed)


def run(dut, steps, interval, backend, agent_kind="full", seed=260923,
        controller_model=None):
    module, harness_cls, base = load_harness(dut, backend)
    harness = harness_cls(backend=backend)
    state = harness.reset()
    agent = make_agent(agent_kind, base, seed, controller_model)
    curve = []
    timings = []
    started = time.perf_counter()
    for step in range(steps):
        tick = time.perf_counter_ns()
        action = agent.predict(state, step, steps)
        timings.append(time.perf_counter_ns() - tick)
        state, _, _, _ = harness.step(action)
        if step % interval == 0:
            curve.append([step, float(harness.coverage)])
    curve.append([steps - 1, float(harness.coverage)])
    elapsed = time.perf_counter() - started
    missing = []
    idx = 0
    for cp in harness._cs.coverpoints:
        for item in cp.get("bins", []):
            if state[idx] == 0:
                missing.append(f"{cp['name']}.{item['name']}")
            idx += 1
    result = {
        "dut": dut, "backend": backend, "agent": agent_kind, "seed": seed,
        "steps": steps, "interval": interval,
        "curve": curve, "covered_bins": int(np.sum(state)),
        "total_bins": int(state.size), "coverage": float(np.mean(state)),
        "missing_bins": missing, "elapsed_seconds": elapsed,
        "predict_mean_ms": float(np.mean(timings) / 1e6),
        "predict_p99_ms": float(np.percentile(timings, 99) / 1e6),
    }
    if agent_kind == "full":
        result["neural_route"] = getattr(agent, "neural_route", None)
        result["neural_confidence"] = float(getattr(agent, "neural_confidence", 0.0))
        result["llm_status"] = getattr(agent, "llm_status", None)
        trace = getattr(getattr(agent, "_policy", None), "macro_trace", None)
        if trace is not None:
            result["neural_macro_trace"] = [int(value) for value in trace]
    if hasattr(harness._dut, "close"):
        harness._dut.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dut", choices=[*DUTS, "all"], default="all")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--backend", choices=["local", "verilator"], default="local")
    parser.add_argument("--agent", choices=["full", "greedy", "random"], default="full")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=260923)
    parser.add_argument("--output", default="results/latest.json")
    parser.add_argument("--controller-model", default=None)
    args = parser.parse_args()
    # "all" retains its historical meaning (the three public image DUTs).
    # Held-out validation DUTs must be selected explicitly.
    duts = list(PUBLIC_DUTS) if args.dut == "all" else [args.dut]
    records = []
    for repeat in range(max(1, args.repeats)):
        for dut in duts:
            records.append(run(dut, args.steps, args.interval, args.backend,
                               args.agent, args.seed + repeat,
                               args.controller_model))
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
