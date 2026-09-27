#!/usr/bin/env python3
"""Try the LLM semantic enricher on a real DUT.

Two modes:

  --inspect   only build the request, call the model, print the audit
              (what was accepted, what was rejected and why, latency, tokens)
  --run       additionally run the coverage episode with the hints applied and
              compare against the same episode with the enricher disabled

Nothing here is part of the submission path; the enrich switch stays off in the
built artifact unless EDA_LLM_ENRICH=1 is set at run time.

  DEEPSEEK_API_KEY=... python3 tools/run_llm_enrichment.py --dut spi_master_public
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from inference.coverage_targets import load_coverage_targets   # noqa: E402
from inference.joint_candidates import compile_joint_candidates  # noqa: E402
from inference.llm_enrichment import LLMEnricher                # noqa: E402
from inference.semantic_ir import build_semantic_ir             # noqa: E402

DUTS = {
    "dma_xfer_public": "public_duts/dma_xfer_public/dma_xfer_public",
    "spi_master_public": "public_duts/spi_master_public/spi_master_public",
    "spi_xfer_public": "public_duts/spi_xfer_public/spi_xfer_public",
    "tlb_mmu_validation":
        "validation_duts/tlb_mmu_validation/tlb_mmu_validation",
    "branch_predictor_validation":
        "validation_duts/branch_predictor_validation/"
        "branch_predictor_validation",
    "cache_ctrl_validation":
        "validation_duts/cache_ctrl_validation/cache_ctrl_validation",
    "watchdog_safety_validation":
        "validation_duts/watchdog_safety_validation/"
        "watchdog_safety_validation",
}


def inspect(dut, base):
    spec = (ROOT / base / "dut" / "dut_spec.md").read_text(encoding="utf-8")
    targets = load_coverage_targets(str(ROOT / base / "dut" / "covergroup.svh"))
    ir = build_semantic_ir(spec)
    compiled = compile_joint_candidates(ir, targets)
    cross_gap = sum(1 for item in targets
                    if item.kind == "cross"
                    and item.index not in {c.target_index for c in compiled})
    sequential = sum(1 for item in targets if item.kind == "sequential")
    enricher = LLMEnricher()
    print("dut                    : %s" % dut)
    print("switch EDA_LLM_ENRICH  : %s" % enricher.enabled)
    print("model                  : %s" % enricher.planner.model)
    print("base url               : %s" % enricher.planner.base_url)
    print("api key present        : %s" % bool(enricher.planner.api_key))
    print("fields / dims          : %d / %d" % (len(ir.fields), ir.action_dim))
    print("registers              : %d" % len(ir.registers))
    print("cross bins unresolved  : %d (the enricher's main target)" % cross_gap)
    print("sequential bins        : %d" % sequential)
    if not enricher.enabled:
        print()
        print("Enricher is off (or no key).  Set DEEPSEEK_API_KEY and")
        print("EDA_LLM_ENRICH=1 to make a real request.")
        return None
    prompt = enricher._prompt(spec, ir, list(targets), set())
    print("prompt chars           : %d" % len(prompt))
    hints = enricher.enrich(spec, ir, targets)
    print()
    print(json.dumps(hints.as_record(), ensure_ascii=False, indent=2))
    for item in hints.field_roles.items():
        print("  role  %s -> %s" % item)
    for item in hints.joint_writes[:6]:
        print("  joint %s writes=%s hold=%d" % (
            item["target_name"], item["writes"], item["hold_cycles"]))
    for item in hints.sequences[:4]:
        print("  seq   %s goal=%s steps=%d cycles=%d" % (
            item["name"], item["goal"], len(item["steps"]), item["cycles"]))
    return hints


def run_episode(dut, base, steps, enrich):
    base = ROOT / base
    spec = importlib.util.spec_from_file_location("h", base / "harness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    harness_cls = next(getattr(module, name) for name in dir(module)
                       if name.endswith("Harness"))
    from inference import InferenceInterface
    os.environ["EDA_LLM_ENRICH"] = "1" if enrich else "0"
    harness = harness_cls()
    state = harness.reset()
    agent = InferenceInterface(str(base / "dut" / "dut_spec.md"),
                               str(base / "dut" / "covergroup.svh"))
    for step in range(steps):
        state = harness.step(agent.predict(state, step, steps))[0]
    return int(np.sum(state)), harness.total_bins, agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dut", default="spi_master_public", choices=sorted(DUTS))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--run", action="store_true",
                        help="also run the episode, local backend, in-process")
    args = parser.parse_args()
    base = DUTS[args.dut]
    if not base:
        raise SystemExit("no path registered for %s" % args.dut)
    hints = inspect(args.dut, base)
    if not args.run:
        return
    if hints is None:
        raise SystemExit("refusing to compare: the enricher made no request")
    off, total, _ = run_episode(args.dut, base, args.steps, enrich=False)
    on, _, agent = run_episode(args.dut, base, args.steps, enrich=True)
    print()
    print("local %d steps, %s" % (args.steps, args.dut))
    print("  enricher off : %d/%d" % (off, total))
    print("  enricher on  : %d/%d  (%+d)" % (on, total, on - off))
    print("  audit        : %s" % json.dumps(
        agent.llm_enrichment_status, ensure_ascii=False))
    print("  llm joint candidates    : %d" % agent._policy.llm_joint_candidate_count)
    print("  llm sequence candidates : %d" % agent._policy.llm_sequence_candidate_count)


if __name__ == "__main__":
    main()
