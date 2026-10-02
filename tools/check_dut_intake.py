#!/usr/bin/env python3
"""Intake check for a new DUT package (generic path vs its own reference).

Every package ships a ``random`` and (usually) a ``greedy`` baseline.  Those are
a free oracle: they say what the interface *can* reach, so the gap between the
generic policy and the stronger baseline localises the failure before any
manual diagnosis.

Reports, for one DUT:
  * the three-way baseline table (random / greedy / generic),
  * which coverpoints the generic path misses that the reference reaches,
  * the interface-capability vector that the gates act on.

The capability vector is the important part: a package whose write path is
gated off for a structural reason shows up here even when the missing bins look
unrelated (this is how the descriptor-DMA blind spot was found).

Run:
    python3 tools/check_dut_intake.py --dut dma_desc_engine_validation
    python3 tools/check_dut_intake.py --dut all --steps 2000
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
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

from inference import InferenceInterface                          # noqa: E402
from tools.run_experiments import DUTS, load_harness              # noqa: E402

# Capability keys whose absence silently disables whole stimulus paths.
CAPABILITY_KEYS = (
    "instance_select", "register_address", "write_enable", "request", "payload",
)


def _owner_table(base: Path):
    meta = json.loads((base / "dut" / "coverage_meta.json").read_text(
        encoding="utf-8"))
    owners = []
    for coverpoint in meta["coverpoints"]:
        for entry in coverpoint["bins"]:
            owners.append((coverpoint["name"], entry.get("name")))
    return owners, int(meta["total_bins"])


def _run(agent, harness_cls, steps):
    harness = harness_cls(backend="local")
    state = np.asarray(harness.reset(), dtype=np.float32).reshape(-1)
    for step in range(steps):
        action = np.asarray(agent.predict(state, step, steps),
                            dtype=np.float32).reshape(-1)
        state = np.asarray(harness.step(action)[0], dtype=np.float32).reshape(-1)
    return {int(i) for i in np.flatnonzero(state > 0.5)}


def _capability_vector(base: Path):
    spec = base / "dut" / "dut_spec.md"
    cover = base / "dut" / "covergroup.svh"
    policy = InferenceInterface(str(spec), str(cover))._policy
    readable = {key: bool(getattr(policy, "interface_capability", {}).get(key))
                for key in CAPABILITY_KEYS}
    unclassified = [field.name for field in policy.semantic_ir.fields
                    if field.role == "scalar"]
    return {
        "capability": readable,
        "score": sum(readable.values()),
        "field_selected_interface": bool(
            getattr(policy, "field_selected_interface", False)),
        "scalar_fields": unclassified,
        "native_sequence_candidates": int(getattr(
            policy, "generic_sequence_native_candidate_count", 0)),
    }


def check(dut: str, steps: int) -> dict:
    module, harness_cls, base = load_harness(dut, "local")
    owners, total = _owner_table(base)

    def build(kind):
        if kind == "full":
            os.environ["EDA_STIMULUS_SEED"] = "260923"
            return InferenceInterface(str(base / "dut" / "dut_spec.md"),
                                      str(base / "dut" / "covergroup.svh"))
        spec = importlib.util.spec_from_file_location(
            f"baseline_{base.name}_{kind}", base / "inference_interface.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.InferenceInterface(str(base / "dut" / "dut_spec.md"),
                                      str(base / "dut" / "covergroup.svh"),
                                      policy=kind, seed=260923)

    hits = {}
    for kind in ("random", "greedy", "full"):
        try:
            hits[kind] = _run(build(kind), harness_cls, steps)
        except Exception as exc:                       # noqa: BLE001
            hits[kind] = set()
            print(f"  [{kind}] unavailable: {exc}")

    reference = hits["greedy"] or hits["random"]
    missing = sorted(reference - hits["full"])
    by_cp = Counter(owners[i][0] for i in missing)

    return {
        "dut": dut,
        "steps": steps,
        "total_bins": total,
        "covered": {k: len(v) for k, v in hits.items()},
        "gap_vs_reference": len(missing),
        "missing_by_coverpoint": dict(by_cp),
        "interface": _capability_vector(base),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dut", default="all",
                        help="a key from run_experiments.DUTS, or 'all'")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    names = list(DUTS) if args.dut == "all" else [args.dut]
    rows = []
    for name in names:
        print(f"=== {name} ({args.steps} steps) ===")
        row = check(name, args.steps)
        rows.append(row)
        covered = row["covered"]
        print(f"  random={covered.get('random')} "
              f"greedy={covered.get('greedy')} "
              f"generic={covered.get('full')} / {row['total_bins']}")
        caps = row["interface"]["capability"]
        print("  capability: " + " ".join(
            f"{k}={'Y' if v else 'n'}" for k, v in caps.items()))
        if row["interface"]["scalar_fields"]:
            print("  scalar (unclassified) fields: "
                  + ", ".join(row["interface"]["scalar_fields"]))
        if row["gap_vs_reference"]:
            print(f"  GAP vs reference: {row['gap_vs_reference']} bins")
            for cp, count in sorted(row["missing_by_coverpoint"].items(),
                                    key=lambda kv: -kv[1])[:6]:
                print(f"      {cp}: {count}")
        else:
            print("  no gap vs reference")

    if args.json:
        Path(args.json).write_text(
            json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
