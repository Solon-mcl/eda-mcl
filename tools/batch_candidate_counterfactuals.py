#!/usr/bin/env python3
"""Generate equal-budget program variants from a declared register interface.

The interface JSON is supplied by the experimenter from the DUT spec. No DUT
name is used to choose a transform. Labels are offline development evidence,
not observations available to the inference policy at run time.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np

from compare_candidate_programs import compare


def _payload(target):
    match = re.search(r"(?:0x([0-9a-fA-F]+)|all_ones(\d+))$", target)
    if not match:
        raise ValueError(f"target has no numeric payload: {target}")
    if match.group(1):
        return int(match.group(1), 16)
    return (1 << int(match.group(2))) - 1


def _writes(program, bus, addresses):
    return [(i, chunk) for i, chunk in enumerate(program)
            if int(chunk["action"][bus["write_enable"]]) == 1
            and int(chunk["action"][bus["address"]]) in addresses]


def generate(original, target, interface, transforms):
    """Return an edited macro only when every requested change is applicable."""
    program = copy.deepcopy(original)
    bus = interface["bus"]
    regs = interface["registers"]
    data_addresses = set(regs["data_port"])
    data_writes = _writes(program, bus, data_addresses)
    enable_writes = _writes(program, bus, {regs["enable"]})
    if not data_writes or not enable_writes:
        raise ValueError("program lacks data-port writes or enable write")
    enable_index = next((i for i, c in enable_writes
                         if int(c["action"][bus["data"]]) != 0), None)
    if enable_index is None:
        raise ValueError("program never enables the controller")
    after_enable = [(i, c) for i, c in data_writes if i > enable_index]
    if not after_enable:
        raise ValueError("program has no payload writes after enable")
    applied = []
    for transform in transforms:
        if transform == "align_payload":
            value = _payload(target)
            if value >= (1 << int(interface["payload_bits"])):
                raise ValueError("target payload exceeds declared data width")
            if int(np.float32(value)) != value:
                raise ValueError("target payload is not exactly representable by float32 actions")
            # Reuse a declared alias; the goal is a consistent address and
            # payload across setup and enabled writes, not a chosen DUT name.
            address = int(data_writes[0][1]["action"][bus["address"]])
            for _, chunk in data_writes:
                chunk["action"][bus["address"]] = address
                chunk["action"][bus["data"]] = value
            applied.append(transform)
        elif transform == "start_threshold":
            writes = _writes(program, bus, {regs["start_threshold"]})
            if not writes:
                raise ValueError("program has no start threshold write")
            # Spec declares the strict comparison: queued frames > threshold.
            count = len(after_enable)
            if interface.get("strict_start_threshold") is not True:
                raise ValueError("strict start threshold must be declared")
            for _, chunk in writes:
                chunk["action"][bus["data"]] = min(
                    int(chunk["action"][bus["data"]]), count - 1)
            applied.append(transform)
        elif transform == "fast_clock":
            writes = _writes(program, bus, {regs["divider"]})
            if not writes:
                raise ValueError("program has no divider write")
            divider = int(interface["minimum_valid_divider"])
            for _, chunk in writes:
                chunk["action"][bus["data"]] = divider
            applied.append(transform)
        elif transform == "longer_observation":
            # Move one idle cycle from each multi-cycle setup gap to the
            # final idle block. The total macro duration stays unchanged.
            if program[-1]["action"][bus["write_enable"]] != 0:
                raise ValueError("program has no final idle block")
            shifted = 0
            for chunk in program[:-1]:
                if (chunk["action"][bus["write_enable"]] == 0
                        and int(chunk["cycles"]) > 1):
                    chunk["cycles"] -= 1
                    shifted += 1
            if not shifted:
                raise ValueError("no idle cycles can be moved")
            program[-1]["cycles"] += shifted
            applied.append(transform)
        else:
            raise ValueError(f"unknown transform: {transform}")
    if sum(x["cycles"] for x in program) != sum(x["cycles"] for x in original):
        raise AssertionError("program duration changed")
    if program == original:
        raise ValueError("transforms made no change")
    return program, applied


def program_features(program, context, source, interface):
    """Features observable when the original program is selected online."""
    bus = interface["bus"]
    regs = interface["registers"]
    enables = _writes(program, bus, {regs["enable"]})
    enabled_at = next((i for i, chunk in enables
                       if int(chunk["action"][bus["data"]]) != 0), len(program))
    data = _writes(program, bus, set(regs["data_port"]))
    after_enable = [(i, chunk) for i, chunk in data if i > enabled_at]
    thresholds = _writes(program, bus, {regs["start_threshold"]})
    dividers = _writes(program, bus, {regs["divider"]})
    target_index = context.get("target_index")
    first_hit = source.get("first_hit_cycles", {}).get(str(target_index))
    step = int(context["step"])
    threshold = (int(thresholds[-1][1]["action"][bus["data"]])
                 if thresholds else None)
    return {
        "budget_fraction": step / int(source["steps"]),
        "covered_bins_before": int(context["covered_bins"]),
        "target_covered_before": first_hit is not None and int(first_hit) <= step,
        "program_cycles": sum(int(x["cycles"]) for x in program),
        "payload_writes_after_enable": len(after_enable),
        "data_port_alias_count": len({int(x["action"][bus["address"]])
                                      for _, x in data}),
        "payload_value_count_after_enable": len({int(x["action"][bus["data"]])
                                                 for _, x in after_enable}),
        "start_threshold": threshold,
        "start_margin": len(after_enable) - threshold if threshold is not None else None,
        "divider": (int(dividers[-1][1]["action"][bus["data"]])
                    if dividers else None),
        "final_observation_cycles": int(program[-1]["cycles"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--occurrences", default="1")
    parser.add_argument("--variants", default=(
        "align_payload,start_threshold,fast_clock,longer_observation,"
        "align_payload+start_threshold+fast_clock"))
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--secrets-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    source = raw[0] if isinstance(raw, list) else raw
    secrets = (json.loads(Path(args.secrets_json).read_text(encoding="utf-8"))
               if args.secrets_json else None)
    interface = json.loads(Path(args.interface).read_text(encoding="utf-8"))
    matches = [item for item in source["replay_trace"]
               if item.get("context", {}).get("target_name") == args.target_name]
    reports = []
    for occurrence in [int(x) for x in args.occurrences.split(",")]:
        if not 1 <= occurrence <= len(matches):
            raise ValueError(f"only {len(matches)} matching programs")
        original = matches[occurrence - 1]["action_sequence"]
        context = matches[occurrence - 1]["context"]
        for name in args.variants.split(","):
            transforms = name.split("+")
            try:
                program, applied = generate(original, args.target_name,
                                            interface, transforms)
                report = compare(source, args.target_name, occurrence,
                                 program, secrets=secrets,
                                 adaptive=args.adaptive)
                report["variant"] = name
                report["transforms"] = applied
                report["features_before"] = program_features(
                    original, context, source, interface)
                report["features_after"] = program_features(
                    program, context, source, interface)
                report["replacement_program"] = program
                reports.append(report)
                fixed = report["coverage_delta"]
                adaptive = report.get("adaptive_continuation", {}).get(
                    "coverage_delta")
                print(f"{occurrence} {name}: fixed {fixed:+d}, adaptive {adaptive}",
                      flush=True)
            except ValueError as exc:
                reports.append({"occurrence": occurrence, "variant": name,
                                "error": str(exc)})
                print(f"{occurrence} {name}: skipped ({exc})", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"source": args.input, "interface": args.interface,
                                  "secrets_json": args.secrets_json,
                                  "target_name": args.target_name,
                                  "reports": reports}, ensure_ascii=False,
                                 indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
