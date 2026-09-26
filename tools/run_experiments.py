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
    "tlb_mmu_validation": (
        "../validation_duts/tlb_mmu_validation/tlb_mmu_validation",
        "TlbMmuHarness"),
}
PUBLIC_DUTS = ("dma_xfer_public", "spi_master_public", "spi_xfer_public")


def joint_candidates_enabled():
    """Return the effective M3b switch used by the submission policy."""
    return os.environ.get("EDA_JOINT_CANDIDATES", "1").lower() not in (
        "0", "false", "no")


def adaptive_joint_ranking_enabled():
    return os.environ.get("EDA_ADAPTIVE_JOINT_RANKING", "1").lower() not in (
        "0", "false", "no")


def field_transaction_templates_enabled():
    return os.environ.get("EDA_FIELD_TRANSACTION_TEMPLATES", "1").lower() not in (
        "0", "false", "no")


def robust_field_writes_enabled():
    return os.environ.get("EDA_ROBUST_FIELD_WRITES", "1").lower() not in (
        "0", "false", "no")


def stateful_sequence_templates_enabled():
    return os.environ.get("EDA_STATEFUL_SEQUENCE_TEMPLATES", "0").lower() not in (
        "0", "false", "no")


def generic_sequence_search_enabled():
    return os.environ.get("EDA_GENERIC_SEQUENCE_SEARCH", "1").lower() not in (
        "0", "false", "no")


def generic_trace_learning_enabled():
    return os.environ.get("EDA_GENERIC_TRACE_LEARNING", "1").lower() not in (
        "0", "false", "no")


def algorithm_variant(agent_kind):
    if agent_kind != "full":
        return f"baseline_{agent_kind}"
    if stateful_sequence_templates_enabled():
        return "m7_stateful_sequence_templates"
    if generic_sequence_search_enabled():
        return ("m8_1_trace_learning" if generic_trace_learning_enabled()
                else "m8_generic_sequence_search")
    if not joint_candidates_enabled():
        return "m3_legal_transactions"
    if field_transaction_templates_enabled():
        return ("m6_backend_robust_transactions"
                if robust_field_writes_enabled() else
                "m5_field_selected_transactions")
    if adaptive_joint_ranking_enabled():
        return "m4_adaptive_joint_ranking"
    return "m3b_joint_candidates"


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
        controller_model=None, trace_actions=False, secrets=None):
    module, harness_cls, base = load_harness(dut, backend)
    harness = harness_cls(backend=backend, secrets=secrets)
    state = np.asarray(harness.reset(), dtype=np.float32).reshape(-1)
    init_started = time.perf_counter()
    agent = make_agent(agent_kind, base, seed, controller_model)
    init_elapsed = time.perf_counter() - init_started
    curve = [[0, float(np.mean(state))]]
    timings = []
    first_hit_cycles = {int(index): 0 for index in np.flatnonzero(state > 0.5)}
    invalid_actions = 0
    started = time.perf_counter()
    for step in range(steps):
        tick = time.perf_counter_ns()
        action = np.asarray(agent.predict(state, step, steps),
                            dtype=np.float32).reshape(-1)
        timings.append(time.perf_counter_ns() - tick)
        expected_dim = int(getattr(agent, "action_dims", action.size))
        if action.size != expected_dim or not np.all(np.isfinite(action)):
            invalid_actions += 1
        state, _, _, _ = harness.step(action)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        cycle = step + 1
        for index in np.flatnonzero(state > 0.5):
            first_hit_cycles.setdefault(int(index), cycle)
        if cycle % interval == 0:
            curve.append([cycle, float(harness.coverage)])
    if curve[-1][0] != steps:
        curve.append([steps, float(harness.coverage)])
    elapsed = time.perf_counter() - started
    auc = float(np.trapz([point[1] for point in curve],
                         [point[0] for point in curve]))
    normalized_auc = auc / max(1, steps)
    missing = []
    idx = 0
    for cp in harness._cs.coverpoints:
        for item in cp.get("bins", []):
            if state[idx] == 0:
                missing.append(f"{cp['name']}.{item['name']}")
            idx += 1
    result = {
        "dut": dut, "backend": backend, "agent": agent_kind,
        "algorithm_variant": algorithm_variant(agent_kind), "seed": seed,
        "steps": steps, "interval": interval,
        "curve": curve, "covered_bins": int(np.sum(state)),
        "total_bins": int(state.size), "coverage": float(np.mean(state)),
        "missing_bins": missing, "elapsed_seconds": elapsed,
        "initialization_seconds": init_elapsed,
        "auc": auc, "normalized_auc": normalized_auc,
        "invalid_actions": invalid_actions,
        "secret_override": secrets is not None,
        "first_hit_cycles": {str(index): cycle for index, cycle in
                             sorted(first_hit_cycles.items())},
        "predict_mean_ms": float(np.mean(timings) / 1e6),
        "predict_p99_ms": float(np.percentile(timings, 99) / 1e6),
        "algorithm_parameters": {
            "joint_candidates": joint_candidates_enabled(),
            "adaptive_joint_ranking": adaptive_joint_ranking_enabled(),
            "joint_max_failed_attempts": (
                int(os.environ["EDA_JOINT_MAX_FAILED_ATTEMPTS"])
                if "EDA_JOINT_MAX_FAILED_ATTEMPTS" in os.environ else "auto"),
            "adaptive_joint_max_candidates": int(os.environ.get(
                "EDA_ADAPTIVE_JOINT_MAX_CANDIDATES", "8")),
            "field_transaction_templates": field_transaction_templates_enabled(),
            "robust_field_writes": robust_field_writes_enabled(),
            "field_write_repeats": int(os.environ.get(
                "EDA_FIELD_WRITE_REPEATS", "16")),
            "stateful_sequence_templates": stateful_sequence_templates_enabled(),
            "generic_sequence_search": generic_sequence_search_enabled(),
            "generic_trace_learning": generic_trace_learning_enabled(),
        },
    }
    if agent_kind == "full":
        result["neural_route"] = getattr(agent, "neural_route", None)
        result["neural_confidence"] = float(getattr(agent, "neural_confidence", 0.0))
        result["llm_status"] = getattr(agent, "llm_status", None)
        trace = getattr(getattr(agent, "_policy", None), "macro_trace", None)
        if trace is not None:
            result["neural_macro_trace"] = [
                int(value) if isinstance(value, (int, np.integer)) else str(value)
                for value in trace
            ]
        policy = getattr(agent, "_policy", None)
        sequence_families = [
            name for name, enabled in (
                ("branch", getattr(policy, "branch_sequence_interface", False)),
                ("memory", getattr(policy, "memory_sequence_interface", False)),
                ("watchdog", getattr(policy, "watchdog_sequence_interface", False)),
            ) if enabled]
        result["algorithm_parameters"]["stateful_sequence_applied"] = bool(
            sequence_families)
        result["algorithm_parameters"]["stateful_sequence_families"] = (
            sequence_families)
        result["algorithm_parameters"]["watchdog_key_a_discovered"] = (
            getattr(policy, "_watchdog_key_a", None) is not None)
        result["algorithm_parameters"]["watchdog_key_b_discovered"] = (
            getattr(policy, "_watchdog_key_b", None) is not None)
        generic_search = getattr(policy, "_generic_sequence_search", None)
        result["algorithm_parameters"]["generic_sequence_candidate_count"] = (
            len(generic_search.candidates) if generic_search is not None else 0)
        result["algorithm_parameters"]["generic_sequence_applicable"] = bool(
            getattr(policy, "generic_sequence_search_applicable", False))
        result["algorithm_parameters"]["generic_sequence_explicit_targets"] = bool(
            getattr(policy, "generic_sequence_has_explicit_targets", False))
        result["algorithm_parameters"]["generic_sequence_native_candidates"] = int(
            getattr(policy, "generic_sequence_native_candidate_count", 0))
        result["algorithm_parameters"]["generic_sequence_immediate"] = bool(
            getattr(policy, "generic_sequence_immediate", False))
        result["algorithm_parameters"]["generic_sequence_started"] = bool(
            getattr(policy, "_generic_sequence_started", False))
        result["algorithm_parameters"]["generic_trace_seed_count"] = int(
            getattr(policy, "_generic_trace_seed_count", 0))
        result["algorithm_parameters"]["generic_trace_candidate_count"] = sum(
            1 for item in (generic_search.candidates
                           if generic_search is not None else ())
            if item.metadata.get("learned_trace"))
        result["algorithm_parameters"]["generic_trace_successes"] = int(
            getattr(policy, "_generic_trace_successes", 0))
        result["algorithm_parameters"]["generic_trace_bootstrap_gains"] = int(
            getattr(policy, "_generic_trace_bootstrap_gains", 0))
        result["algorithm_parameters"]["generic_trace_failures"] = int(
            getattr(policy, "_generic_trace_failures", 0))
        result["algorithm_parameters"]["generic_trace_exploration_allowed"] = bool(
            getattr(policy, "_trace_exploration_allowed", lambda: False)())
        result["algorithm_parameters"]["effective_joint_retry_budget"] = (
            int(getattr(policy, "joint_retry_budget", 0)))
        result["algorithm_parameters"]["adaptive_joint_ranking_applied"] = (
            bool(getattr(policy, "adaptive_joint_ranking_applied", False)))
        joint_candidates = getattr(policy, "joint_candidates", None)
        if joint_candidates is not None:
            result["joint_candidate_count"] = len(joint_candidates)
            result["joint_candidate_names"] = [
                item.target_name for item in joint_candidates]
            history = getattr(getattr(policy, "_macro_scheduler", None),
                              "history", [])
            result["joint_programs_completed"] = sum(
                1 for item in history
                if item.get("context", {}).get("joint_candidate"))
            result["joint_program_new_bins"] = sum(
                int(item.get("new_bins", 0)) for item in history
                if item.get("context", {}).get("joint_candidate"))
            result["generic_sequence_programs_completed"] = sum(
                1 for item in history
                if item.get("context", {}).get("generic_sequence_search"))
            result["generic_sequence_program_new_bins"] = sum(
                int(item.get("new_bins", 0)) for item in history
                if item.get("context", {}).get("generic_sequence_search"))
            result["generic_trace_programs_completed"] = sum(
                1 for item in history
                if (item.get("context", {}).get("generic_sequence_search") and
                    str(item.get("context", {}).get(
                        "sequence_family", "")).startswith("learned_trace")))
            result["generic_trace_program_new_bins"] = sum(
                int(item.get("new_bins", 0)) for item in history
                if (item.get("context", {}).get("generic_sequence_search") and
                    str(item.get("context", {}).get(
                        "sequence_family", "")).startswith("learned_trace")))
            result["program_trace"] = [{
                "macro": item.get("macro"),
                "cycles": int(item.get("cycles", 0)),
                "new_bins": int(item.get("new_bins", 0)),
                "new_bin_indices": list(item.get("new_bin_indices", ())),
                "context": dict(item.get("context", {})),
            } for item in history]
            if trace_actions:
                idle = (policy._base_action().astype(float).tolist()
                        if hasattr(policy, "_base_action") else
                        [0.0] * int(getattr(agent, "action_dims", 0)))
                result["replay_idle_action"] = idle
                result["replay_trace"] = [{
                    "macro": item.get("macro"),
                    "cycles": int(item.get("cycles", 0)),
                    "new_bins": int(item.get("new_bins", 0)),
                    "new_bin_indices": list(item.get("new_bin_indices", ())),
                    "context": dict(item.get("context", {})),
                    "action_sequence": list(item.get("action_sequence", ())),
                } for item in history]
            ranker = getattr(policy, "_joint_ranker", None)
            result["joint_candidate_stats"] = (
                ranker.snapshot() if ranker is not None else {})
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
    parser.add_argument(
        "--secrets-json", default=None,
        help="development-only JSON object or JSON file with harness secrets")
    parser.add_argument(
        "--joint-candidates", choices=("default", "on", "off"),
        default="default",
        help=("select M3b joint candidates explicitly; 'default' preserves "
              "EDA_JOINT_CANDIDATES or the submission default (on)"))
    parser.add_argument(
        "--adaptive-joint-ranking", choices=("default", "on", "off"),
        default="default",
        help=("enable M4 outcome/cost ranking; default preserves "
              "EDA_ADAPTIVE_JOINT_RANKING or the submission default (on)"))
    parser.add_argument(
        "--joint-max-failed-attempts", type=int, default=None,
        help="M4 retry cap per still-missing joint target (default: 2)")
    parser.add_argument(
        "--field-transaction-templates", choices=("default", "on", "off"),
        default="default",
        help="enable M5 selector-addressed complete transactions (default: on)")
    parser.add_argument(
        "--robust-field-writes", choices=("default", "on", "off"),
        default="default",
        help=("enable M6 bounded retries for selector-addressed writes "
              "(default: on)"))
    parser.add_argument(
        "--field-write-repeats", type=int, default=None,
        help="M6 retry cycles per selector-addressed write (default: 16)")
    parser.add_argument(
        "--stateful-sequence-templates", choices=("default", "on", "off"),
        default="default",
        help="enable M7 expert history/timing templates (default: off)")
    parser.add_argument(
        "--generic-sequence-search", choices=("default", "on", "off"),
        default="default",
        help="enable M8 generic coverage-directed program search (default: on)")
    parser.add_argument(
        "--generic-trace-learning", choices=("default", "on", "off"),
        default="default",
        help="enable M8.1 rewarded-transaction trace learning (default: on)")
    parser.add_argument(
        "--trace-actions", action="store_true",
        help="include exact compressed macro actions for offline replay")
    args = parser.parse_args()
    secrets = None
    if args.secrets_json:
        raw = args.secrets_json.strip()
        candidate = Path(raw) if not raw.startswith("{") else None
        text = (candidate.read_text(encoding="utf-8")
                if candidate is not None and candidate.exists() else raw)
        try:
            secrets = json.loads(text)
        except json.JSONDecodeError as exc:
            parser.error(f"invalid --secrets-json: {exc}")
        if not isinstance(secrets, dict):
            parser.error("--secrets-json must decode to an object")
    if args.joint_candidates != "default":
        os.environ["EDA_JOINT_CANDIDATES"] = (
            "1" if args.joint_candidates == "on" else "0")
    if args.adaptive_joint_ranking != "default":
        os.environ["EDA_ADAPTIVE_JOINT_RANKING"] = (
            "1" if args.adaptive_joint_ranking == "on" else "0")
    if args.joint_max_failed_attempts is not None:
        if args.joint_max_failed_attempts < 1:
            parser.error("--joint-max-failed-attempts must be >= 1")
        os.environ["EDA_JOINT_MAX_FAILED_ATTEMPTS"] = str(
            args.joint_max_failed_attempts)
    if args.field_transaction_templates != "default":
        os.environ["EDA_FIELD_TRANSACTION_TEMPLATES"] = (
            "1" if args.field_transaction_templates == "on" else "0")
    if args.robust_field_writes != "default":
        os.environ["EDA_ROBUST_FIELD_WRITES"] = (
            "1" if args.robust_field_writes == "on" else "0")
    if args.field_write_repeats is not None:
        if args.field_write_repeats < 1:
            parser.error("--field-write-repeats must be >= 1")
        os.environ["EDA_FIELD_WRITE_REPEATS"] = str(args.field_write_repeats)
    if args.stateful_sequence_templates != "default":
        os.environ["EDA_STATEFUL_SEQUENCE_TEMPLATES"] = (
            "1" if args.stateful_sequence_templates == "on" else "0")
    if args.generic_sequence_search != "default":
        os.environ["EDA_GENERIC_SEQUENCE_SEARCH"] = (
            "1" if args.generic_sequence_search == "on" else "0")
    if args.generic_trace_learning != "default":
        os.environ["EDA_GENERIC_TRACE_LEARNING"] = (
            "1" if args.generic_trace_learning == "on" else "0")
    # "all" retains its historical meaning (the three public image DUTs).
    # Held-out validation DUTs must be selected explicitly.
    duts = list(PUBLIC_DUTS) if args.dut == "all" else [args.dut]
    records = []
    for repeat in range(max(1, args.repeats)):
        for dut in duts:
            records.append(run(dut, args.steps, args.interval, args.backend,
                               args.agent, args.seed + repeat,
                               args.controller_model, args.trace_actions,
                               secrets))
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
