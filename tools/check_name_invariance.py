#!/usr/bin/env python3
"""End-to-end name-invariance check.

Holds the circuit, the coverage model and the harness fixed, and varies only
the specification text.  Any coverage difference is therefore attributable to
parsing rather than to the design under test - which is exactly the property
that matters for an unseen DUT: the same hardware described in a different
house style must still be covered.

Run:  python3 tools/check_name_invariance.py [--steps 5000]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from inference import InferenceInterface                        # noqa: E402

BASE = ROOT / "validation_duts/tlb_mmu_validation/tlb_mmu_validation"
VARIANTS = ROOT / "synthesis/spec_variants"


def load_harness():
    spec = importlib.util.spec_from_file_location("tlb_harness",
                                                  BASE / "harness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TlbMmuHarness


def episode(harness_cls, spec_path, steps, seed=260923):
    os.environ["EDA_STIMULUS_SEED"] = str(seed)
    harness = harness_cls()
    state = harness.reset()
    agent = InferenceInterface(str(spec_path),
                               str(BASE / "dut" / "covergroup.svh"))
    for step in range(steps):
        state = harness.step(agent.predict(state, step, steps))[0]
    return int(np.sum(state)), harness.total_bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    harness_cls = load_harness()
    rows = [("baseline (original spec)", BASE / "dut" / "dut_spec.md")]
    rows += [(path.stem, path) for path in sorted(VARIANTS.glob("*.md"))]

    print("%-28s %12s %9s" % ("spec variant", "covered", "verdict"))
    baseline = None
    worst = 0
    for name, path in rows:
        hit, total = episode(harness_cls, path, args.steps)
        if baseline is None:
            baseline, verdict = hit, "-"
        else:
            worst = max(worst, baseline - hit)
            verdict = "SAME" if hit >= baseline else "-%d" % (baseline - hit)
        print("%-28s %6d/%-4d %9s" % (name, hit, total, verdict))
    print()
    print("same circuit / coverage model / harness; only the spec text differs")
    print("worst shortfall vs baseline: %d bins" % worst)
    raise SystemExit(1 if worst > 0 else 0)


if __name__ == "__main__":
    main()
