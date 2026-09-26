#!/usr/bin/env python3
"""Audit local DUT packages against the committee inference contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from inference.coverage_targets import load_coverage_targets  # noqa: E402
from inference.semantic_ir import build_semantic_ir  # noqa: E402


def discover_duts():
    for group in ("public_duts", "validation_duts"):
        root = ROOT / group
        if not root.exists():
            continue
        for spec_path in sorted(root.glob("*/*/dut/dut_spec.md")):
            dut_dir = spec_path.parent
            yield group, spec_path.parents[2].name, dut_dir


def audit_one(group: str, name: str, dut_dir: Path):
    spec_path = dut_dir / "dut_spec.md"
    covergroup_path = dut_dir / "covergroup.svh"
    metadata_path = dut_dir / "coverage_meta.json"
    spec = spec_path.read_text(encoding="utf-8")
    ir = build_semantic_ir(spec)
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    flattened = sum(len(item.get("bins", []))
                    for item in metadata.get("coverpoints", []))
    declared_bins = metadata.get("total_bins")
    targets = load_coverage_targets(str(covergroup_path))
    risky_values = []
    for target in targets:
        for value in target.values:
            if isinstance(value, int) and abs(value) > 2**24:
                risky_values.append({
                    "index": target.index,
                    "bin": f"{target.coverpoint}.{target.bin_name}",
                    "value": value,
                })
    issues = []
    if not covergroup_path.exists():
        issues.append("missing covergroup.svh")
    if not metadata_path.exists():
        issues.append("missing coverage_meta.json; runtime must degrade safely")
    if ir.declared_dim is not None and ir.declared_dim != ir.action_dim:
        issues.append("declared action dimension disagrees with action field list")
    if declared_bins is not None and int(declared_bins) != flattened:
        issues.append("metadata total_bins disagrees with flattened bins")
    if metadata_path.exists() and flattened and not targets:
        issues.append("coverage metadata rejected by parser validation")
    if risky_values:
        issues.append("coverage values exceed exact float32 integer range")
    return {
        "group": group,
        "dut": name,
        "spec_path": str(spec_path.relative_to(ROOT)),
        "action_dim": ir.action_dim,
        "declared_action_dim": ir.declared_dim,
        "action_fields": len(ir.fields),
        "field_roles": {item.name: item.role for item in ir.fields},
        "register_map": [{"address": item.address, "name": item.name}
                         for item in ir.registers],
        "declared_bins": declared_bins,
        "flattened_bins": flattened,
        "parsed_targets": len(targets),
        "float32_integer_risks": risky_values,
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/interface_contract_audit.json")
    args = parser.parse_args()
    records = [audit_one(*item) for item in discover_duts()]
    report = {
        "committee_constructor_inputs": ["dut_spec_path", "covergroup_path"],
        "runtime_metadata_policy": (
            "probe sibling coverage_meta.json; validate flattened bin count; "
            "fall back when absent or inconsistent"),
        "local_duts": records,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
