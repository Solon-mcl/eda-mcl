#!/usr/bin/env python3
"""One-call live smoke test; never prints credentials or raw model output."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from inference import InferenceInterface  # noqa: E402
from inference.deepseek_planner import DeepSeekPlanner  # noqa: E402

DUTS = {
    "dma": ROOT / "public_duts/dma_xfer_public/dma_xfer_public/dut",
    "spi_master": ROOT / "public_duts/spi_master_public/spi_master_public/dut",
    "spi_xfer": ROOT / "public_duts/spi_xfer_public/spi_xfer_public/dut",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dut", choices=DUTS, default="spi_xfer")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.compact:
        spec = """Toy ALU. action = [reset_n, opcode, operand_a, operand_b].
reset_n is 0 for reset and 1 to run. opcode 0=add, 1=sub, 2=and.
Operands are unsigned bytes. Return four float32-compatible action values."""
        cover = """covergroup alu; opcode: coverpoint opcode { bins ops[] = {0,1,2}; }
zero: coverpoint result { bins zero = {0}; } endgroup"""
        plan, status = DeepSeekPlanner().plan(spec, cover)
        plan = plan or {}
        print(json.dumps({
            "compact": True, "llm_status": status,
            "family_hint": plan.get("family_hint"),
            "action_dim": plan.get("action_dim"),
            "validated_program_actions": len(plan.get("program", [])),
        }, ensure_ascii=False, indent=2))
        raise SystemExit(0 if status.get("status") == "ok" else 2)
    base = DUTS[args.dut]
    agent = InferenceInterface(str(base / "dut_spec.md"),
                               str(base / "covergroup.svh"))
    plan = agent.llm_plan or {}
    action = agent.predict(np.zeros(
        len(agent.coverage_targets), dtype=np.float32), 0, 50000)
    report = {
        "dut": args.dut,
        "environment": {
            "api_key_present": bool(os.environ.get("DEEPSEEK_API_KEY")),
            "base_url_present": bool(os.environ.get("DEEPSEEK_BASE_URL")),
            "model_present": bool(os.environ.get("DEEPSEEK_MODEL")),
            "explicitly_disabled": os.environ.get(
                "DEEPSEEK_ENABLED", "1").lower() in ("0", "false", "no"),
        },
        "llm_status": agent.llm_status,
        "family_hint": plan.get("family_hint"),
        "action_dim": plan.get("action_dim"),
        "validated_program_actions": len(plan.get("program", [])),
        "first_action_dim": int(action.size),
        "selected_policy": type(agent._policy).__name__,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if agent.llm_status.get("status") != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
