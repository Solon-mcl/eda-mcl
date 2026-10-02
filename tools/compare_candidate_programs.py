#!/usr/bin/env python3
"""Compare one replacement program against an exact, fixed 30k action trace.

The prefix and suffix actions are identical in both branches. The replacement
must occupy exactly the same number of cycles as the selected macro, so every
reported bin difference is caused by those changed actions and their DUT state
effects, without a changed online scheduler confounding the comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from run_experiments import load_harness, make_agent


def _expand(sequence, dims):
    actions = []
    for chunk in sequence:
        action = np.asarray(chunk["action"], dtype=np.float32).reshape(-1)
        cycles = int(chunk["cycles"])
        if action.size != dims or cycles < 1 or not np.all(np.isfinite(action)):
            raise ValueError("invalid action dimension, value, or cycle count")
        actions.extend([action] * cycles)
    return np.stack(actions) if actions else np.empty((0, dims), np.float32)


def _bin_names(base):
    meta = json.loads((Path(base) / "dut" / "coverage_meta.json").read_text(
        encoding="utf-8"))
    return [f"{point['name']}.{item['name']}"
            for point in meta["coverpoints"] for item in point["bins"]]


def _play(dut, backend, actions, interval, checkpoint, secrets=None):
    _, harness_cls, _ = load_harness(dut, backend)
    harness = harness_cls(backend=backend, secrets=secrets)
    state = np.asarray(harness.reset(), dtype=np.float32).reshape(-1)
    curve = [[0, float(harness.coverage)]]
    first_hit = {int(index): 0 for index in np.flatnonzero(state > 0.5)}
    prefix_bins = None
    try:
        for step, action in enumerate(actions):
            if step == checkpoint:
                prefix_bins = set(np.flatnonzero(state > 0.5).tolist())
            state, _, _, _ = harness.step(action)
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            for index in np.flatnonzero(state > 0.5):
                first_hit.setdefault(int(index), step + 1)
            if (step + 1) % interval == 0:
                curve.append([step + 1, float(harness.coverage)])
        if curve[-1][0] != len(actions):
            curve.append([len(actions), float(harness.coverage)])
        integrate = getattr(np, "trapezoid", None) or np.trapz
        auc = float(integrate([point[1] for point in curve],
                              [point[0] for point in curve]) / len(actions))
        return {"covered": set(np.flatnonzero(state > 0.5).tolist()),
                "prefix_bins": prefix_bins, "auc": auc,
                "curve": curve, "first_hit": first_hit}
    finally:
        if hasattr(harness._dut, "close"):
            harness._dut.close()


def _play_adaptive(source, original_actions, checkpoint, original_program,
                   replacement, target_name, secrets=None):
    dut, backend = source["dut"], source["backend"]
    _, harness_cls, base = load_harness(dut, backend)
    harness = harness_cls(backend=backend, secrets=secrets)
    state = np.asarray(harness.reset(), dtype=np.float32).reshape(-1)
    parameters = source.get("algorithm_parameters", {})
    keys = {
        "EDA_TARGET_TASK_SCHEDULER": bool(parameters.get(
            "target_task_scheduler_requested", False)),
        "EDA_TARGET_TASK_PROGRAM_FIX": bool(parameters.get(
            "target_task_program_fix_requested", False)),
        "EDA_TARGET_TASK_PERIOD": int(parameters.get("target_task_period", 12)),
        "EDA_CREDIT_PACKET_PROGRAMS": bool(parameters.get(
            "credit_packet_programs", False)),
    }
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key, value in keys.items():
            os.environ[key] = (str(int(value)) if isinstance(value, bool)
                               else str(value))
        agent = make_agent("full", base, int(source["seed"]))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    curve = [[0, float(harness.coverage)]]
    first_hit = {int(index): 0 for index in np.flatnonzero(state > 0.5)}
    prefix_bins = None
    try:
        for step in range(int(source["steps"])):
            if step == checkpoint:
                prefix_bins = set(np.flatnonzero(state > 0.5).tolist())
            action = np.asarray(agent.predict(state, step, int(source["steps"])),
                                dtype=np.float32).reshape(-1)
            if step < checkpoint and not np.array_equal(action, original_actions[step]):
                raise ValueError(f"online prefix diverged at cycle {step}")
            if step == checkpoint:
                policy = agent._policy
                scheduler = policy._macro_scheduler
                if scheduler.active_context.get("target_name") != target_name:
                    raise ValueError("online policy selected a different target")
                planned = _expand(scheduler.active_plan, action.size)
                if not np.array_equal(planned, original_program):
                    raise ValueError("online policy generated a different original program")
                policy.queue.clear()
                for chunk in replacement:
                    policy.put(chunk["action"], int(chunk["cycles"]))
                scheduler.attach_plan(list(replacement))
                action = policy.take()
            state, _, _, _ = harness.step(action)
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            for index in np.flatnonzero(state > 0.5):
                first_hit.setdefault(int(index), step + 1)
            if (step + 1) % int(source["interval"]) == 0:
                curve.append([step + 1, float(harness.coverage)])
        if curve[-1][0] != int(source["steps"]):
            curve.append([int(source["steps"]), float(harness.coverage)])
        integrate = getattr(np, "trapezoid", None) or np.trapz
        auc = float(integrate([point[1] for point in curve],
                              [point[0] for point in curve]) / int(source["steps"]))
        return {"covered": set(np.flatnonzero(state > 0.5).tolist()),
                "prefix_bins": prefix_bins, "auc": auc,
                "first_hit": first_hit}
    finally:
        if hasattr(harness._dut, "close"):
            harness._dut.close()


def compare(source, target_name, occurrence, replacement, secrets=None,
            adaptive=False):
    if source.get("secret_override") and secrets is None:
        raise ValueError("source used custom secrets; pass --secrets-json")
    dut, backend = source["dut"], source["backend"]
    _, harness_cls, base = load_harness(dut, backend)
    dims = int(harness_cls.DIMS)
    actions = _expand(source.get("exact_action_trace", ()), dims)
    if len(actions) != int(source["steps"]):
        raise ValueError("source needs a complete exact_action_trace; rerun with --trace-actions")
    matches = [item for item in source.get("replay_trace", ())
               if item.get("context", {}).get("target_name") == target_name]
    if not 1 <= occurrence <= len(matches):
        raise ValueError(f"found {len(matches)} matching programs")
    chosen = matches[occurrence - 1]
    checkpoint = int(chosen["context"]["step"])
    original = _expand(chosen["action_sequence"], dims)
    alternative = _expand(replacement, dims)
    if len(original) != int(chosen["cycles"]):
        raise ValueError("source macro duration does not match its actions")
    if len(alternative) != len(original):
        raise ValueError("replacement must use the same number of cycles")
    stop = checkpoint + len(original)
    if stop > len(actions) or not np.array_equal(actions[checkpoint:stop], original):
        raise ValueError("macro actions do not match the exact trace at the checkpoint")
    variant_actions = actions.copy()
    variant_actions[checkpoint:stop] = alternative
    baseline = _play(dut, backend, actions, int(source["interval"]),
                     checkpoint, secrets)
    variant = _play(dut, backend, variant_actions, int(source["interval"]),
                    checkpoint, secrets)
    if len(baseline["covered"]) != int(source["covered_bins"]):
        raise ValueError("exact trace does not reproduce source coverage")
    if abs(baseline["auc"] - float(source["normalized_auc"])) > 1e-8:
        raise ValueError("exact trace does not reproduce source AUC: "
                         f"{baseline['auc']} vs {source['normalized_auc']}")
    if len(baseline["curve"]) != len(source["curve"]) or any(
            first[0] != second[0] or abs(first[1] - second[1]) > 1e-8
            for first, second in zip(baseline["curve"], source["curve"])):
        raise ValueError("exact trace does not reproduce source coverage curve")
    if baseline["prefix_bins"] != variant["prefix_bins"]:
        raise AssertionError("counterfactual branches diverged before checkpoint")
    names = _bin_names(base)
    label = lambda indices: [names[index] for index in sorted(indices)]
    target_index = next((index for index, name in enumerate(names)
                         if name == target_name), None)
    report = {
        "dut": dut, "backend": backend, "steps": len(actions),
        "target_name": target_name, "occurrence": occurrence,
        "checkpoint_cycle": checkpoint, "program_cycles": len(original),
        "prefix_sha256": hashlib.sha256(
            actions[:checkpoint].tobytes()).hexdigest(),
        "prefix_covered_bins": len(baseline["prefix_bins"]),
        "baseline_covered_bins": len(baseline["covered"]),
        "variant_covered_bins": len(variant["covered"]),
        "coverage_delta": len(variant["covered"]) - len(baseline["covered"]),
        "baseline_auc": baseline["auc"], "variant_auc": variant["auc"],
        "auc_delta": variant["auc"] - baseline["auc"],
        "gained_bins": label(variant["covered"] - baseline["covered"]),
        "lost_bins": label(baseline["covered"] - variant["covered"]),
        "target_first_hit_baseline": baseline["first_hit"].get(target_index),
        "target_first_hit_variant": variant["first_hit"].get(target_index),
        "continuation": "fixed original actions after the replacement program",
    }
    if adaptive:
        online = _play_adaptive(source, actions, checkpoint, original,
                                replacement, target_name, secrets)
        if online["prefix_bins"] != baseline["prefix_bins"]:
            raise AssertionError("adaptive branch diverged before checkpoint")
        report["adaptive_continuation"] = {
            "covered_bins": len(online["covered"]),
            "coverage_delta": len(online["covered"]) - len(baseline["covered"]),
            "auc": online["auc"],
            "auc_delta": online["auc"] - baseline["auc"],
            "gained_bins": label(online["covered"] - baseline["covered"]),
            "lost_bins": label(baseline["covered"] - online["covered"]),
            "target_first_hit": online["first_hit"].get(target_index),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--occurrence", type=int, default=1)
    parser.add_argument("--replacement", required=True)
    parser.add_argument("--secrets-json", default=None)
    parser.add_argument("--adaptive", action="store_true",
                        help="also continue the online policy from the same prefix")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    source = data[0] if isinstance(data, list) else data
    replacement = json.loads(Path(args.replacement).read_text(encoding="utf-8"))
    if isinstance(replacement, dict):
        replacement = replacement["action_sequence"]
    secrets = (json.loads(Path(args.secrets_json).read_text(encoding="utf-8"))
               if args.secrets_json else None)
    report = compare(source, args.target_name, args.occurrence,
                     replacement, secrets, args.adaptive)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
