#!/usr/bin/env python3
"""Measure parsing generalization on the synthetic spec corpus.

Reports two numbers the bundled DUTs cannot report:

* field-role recall -- did the parser recover the intended role from the name?
* candidate yield  -- did the policy produce at least one stimulus candidate?

Run:  python3 tools/check_parsing_generalization.py [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

from inference.semantic_ir import build_semantic_ir            # noqa: E402
from inference import _GenericPolicy                            # noqa: E402
from synthesis.spec_corpus import CORPUS                        # noqa: E402


def evaluate():
    role_total = role_ok = 0
    cand_ok = 0
    rows = []
    candidates = []
    for entry in CORPUS:
        spec = entry["spec"]
        ir = build_semantic_ir(spec)
        got = {field.name: field.role for field in ir.fields}
        misses = []
        for name, expected in entry["expect_roles"].items():
            choices = expected if isinstance(expected, tuple) else (expected,)
            role_total += 1
            if got.get(name) in choices:
                role_ok += 1
            else:
                misses.append("%s: want %s, got %s" % (name, "/".join(choices),
                                                       got.get(name)))
        fields = [field.name for field in ir.fields]
        policy = _GenericPolicy(max(1, ir.action_dim), fields=fields, spec=spec,
                                semantic_ir=ir, coverage_targets=[])
        count = len(policy._generic_sequence_search.candidates)
        ok = count >= entry["min_candidates"]
        cand_ok += int(ok)
        candidates.append(count)
        rows.append((entry["id"], entry["style"], len(ir.fields), count, ok, misses))
    return role_ok, role_total, cand_ok, candidates, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    role_ok, role_total, cand_ok, counts, rows = evaluate()
    total = len(rows)

    print("%-24s %-11s %6s %7s %6s" % ("corpus entry", "style", "fields",
                                       "cands", "verdict"))
    failed = 0
    for entry_id, style, n_fields, count, ok, misses in rows:
        if not ok:
            failed += 1
        print("%-24s %-11s %6d %7d %6s" % (entry_id, style, n_fields, count,
                                           "ok" if ok else "FAIL"))
        if args.verbose and misses:
            for miss in misses:
                print("      role miss: " + miss)

    recall = (100.0 * role_ok / role_total) if role_total else 100.0
    zero = sum(1 for value in counts if value == 0)
    print()
    print("field-role recall      : %d/%d  (%.1f%%)" % (role_ok, role_total, recall))
    print("candidate yield >= min : %d/%d  (%.1f%%)" % (cand_ok, total,
                                                       100.0 * cand_ok / total))
    print("entries with zero cands: %d/%d" % (zero, total))
    print("mean candidates        : %.1f" % (sum(counts) / float(total)))
    print("verdict                : %s" % ("FAIL" if failed else "PASS"))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
