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


def list_models():
    """Ask the endpoint which model ids it serves, so the v4.1 id can be read
    off instead of guessed."""
    import urllib.request
    from inference.deepseek_planner import DeepSeekPlanner
    planner = DeepSeekPlanner()
    if not planner.api_key:
        return {"status": "no_api_key"}
    request = urllib.request.Request(
        planner.base_url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + planner.api_key})
    try:
        with urllib.request.urlopen(request, timeout=planner.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "error_" + type(exc).__name__, "detail": str(exc)[:200]}
    ids = [item.get("id") for item in payload.get("data", [])
           if isinstance(item, dict)]
    return {"status": "ok", "base_url": planner.base_url,
            "models": sorted(item for item in ids if item)}


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


def apply_api_key_file(path):
    """Read the key from a file so it never appears in a command line or log.

    The file form is preferred over DEEPSEEK_API_KEY=... on the command line:
    argv is visible to other processes and tends to end up in shell history and
    transcripts.  The value is only ever put into this process's environment and
    is never printed.
    """
    if not path:
        return False
    raw = Path(path).expanduser().read_text(encoding="utf-8").strip()
    if not raw:
        raise SystemExit("api key file is empty: %s" % path)
    # Tolerate KEY=value lines and stray quotes.
    if "=" in raw and not raw.startswith("sk-"):
        raw = raw.split("=", 1)[1]
    os.environ["DEEPSEEK_API_KEY"] = raw.strip().strip("'\"")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dut", default="spi_master_public", choices=sorted(DUTS))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--run", action="store_true",
                        help="also run the episode, local backend, in-process")
    parser.add_argument("--api-key-file", default=os.environ.get(
        "DEEPSEEK_API_KEY_FILE"),
        help="file holding the API key; preferred over an inline env var")
    parser.add_argument("--list-models", action="store_true",
                        help="GET /models and print the available model ids")
    args = parser.parse_args()
    apply_api_key_file(args.api_key_file)
    if args.run or args.list_models:
        os.environ["EDA_LLM_ENRICH"] = "1"
    if args.list_models:
        print(json.dumps(list_models(), ensure_ascii=False, indent=2))
        return
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
