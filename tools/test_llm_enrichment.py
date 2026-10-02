#!/usr/bin/env python3
"""Offline contract tests for the LLM semantic enricher.

No network: a fake opener stands in for the API.  The point is to pin down the
three properties the design depends on.

1. Validation actually rejects.  A hallucinated field, an out-of-vocabulary
   role, a `padding` role, an unknown target, an illegal register address and an
   out-of-range value must all be dropped with a reason.
2. Accepted hints take effect.  Role overrides reach the IR, and joint /
   sequence hypotheses reach the candidate pool.
3. Disabled is inert and silent.  With the switch off nothing is requested and
   the built policy is identical to one built with no hints at all.

Run:  python3 tools/test_llm_enrichment.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from inference import InferenceInterface, UniversalPolicy   # noqa: E402
from inference.coverage_targets import load_coverage_targets  # noqa: E402
from inference.deepseek_planner import DeepSeekPlanner       # noqa: E402
from inference.llm_enrichment import LLMEnricher, SEMANTIC_ROLES  # noqa: E402
from inference.joint_candidates import compile_joint_candidates    # noqa: E402
from inference.semantic_ir import build_semantic_ir          # noqa: E402

BASE = ROOT / "public_duts/spi_master_public/spi_master_public"


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def build_payload(spec, ir, targets):
    """A payload that is mostly valid, with specific invalid items mixed in."""
    compiled = {item.target_index
                for item in compile_joint_candidates(ir, targets)}
    # Any target the local compiler could not turn into a program will do: the
    # enrichment exists precisely to cover those.  This used to be restricted
    # to a cross, because crosses were the only kind compiled at all; now that
    # single-condition targets are compiled too, the unresolved ones live in
    # other kinds, and requiring a cross would test nothing.
    gap = next((item for item in targets if item.index not in compiled), None)
    assert gap is not None, "the bundled SPI-master spec has no unresolved target"
    address = ir.registers[0].address if ir.registers else 0
    scalar = next((item.name for item in ir.fields if item.role == "scalar"),
                  None)
    assert scalar is not None, "expected at least one unclassified field"
    used = {scalar}
    others = []
    for item in ir.fields:
        # padding-excluded and distinct, so each invalid case exercises exactly
        # one rule instead of colliding with another (dict keys are unique).
        if item.role != "padding" and item.name not in used:
            used.add(item.name)
            others.append(item)
    assert len(others) >= 3, "need three distinct classified fields"
    bad_role, pad_target, no_change = others[:3]
    return {
        "field_roles": {
            scalar: "enable",                 # accepted
            "no_such_field": "enable",        # unknown field
            bad_role.name: "definitely_not_a_role",   # outside vocabulary
            pad_target.name: "padding",       # padding is never assignable
            no_change.name: no_change.role,   # no change
        },
        "joint_writes": [
            {"target": "%s.%s" % (gap.coverpoint, gap.bin_name),
             "writes": [[address, 0x1], [0x7FFFFFFF, 0x2]],  # 2nd addr illegal
             "hold_cycles": 96, "reason": "test"},
            {"target": "no.such_target", "writes": [[address, 1]]},
            {"target": "%s.%s" % (gap.coverpoint, gap.bin_name),
             "writes": [[0x100, 1]]},          # address out of range
            {"target": "%s.%s" % (gap.coverpoint, gap.bin_name),
             "writes": [[address, 1 << 33]]},  # value out of range
        ],
        "sequences": [
            {"name": "enter wait state",
             "goal": "%s.%s" % (gap.coverpoint, gap.bin_name),
             "steps": [{"set": {bad_role.name: 1}, "cycles": 4},
                       {"set": {pad_target.name: 0}, "cycles": 8}]},
            {"name": "unknown field only",
             "steps": [{"set": {"no_such_field": 1}, "cycles": 4}]},
        ],
    }


def main():
    os.environ["DEEPSEEK_API_KEY"] = "test-only-not-a-secret"
    os.environ["EDA_LLM_ENRICH"] = "1"
    # The validator is exercised with sequences enabled; their default-off
    # behaviour is asserted separately below.
    os.environ["EDA_LLM_SEQUENCES"] = "1"
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        return FakeResponse({
            "choices": [{"message": {"content": json.dumps(payload)}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 300,
                      "total_tokens": 1200},
        })

    spec = (BASE / "dut" / "dut_spec.md").read_text(encoding="utf-8")
    covergroup = BASE / "dut" / "covergroup.svh"
    targets = load_coverage_targets(str(covergroup))
    ir = build_semantic_ir(spec)
    payload = build_payload(spec, ir, targets)

    enricher = LLMEnricher(planner=DeepSeekPlanner(opener=opener))
    hints = enricher.enrich(spec, ir, targets)

    assert hints.status["status"] == "ok", hints.status
    assert hints.status["usage"]["total_tokens"] == 1200, hints.status
    assert len(calls) == 1, calls

    # ---- 1. validation rejects -------------------------------------------
    accepted_roles = dict(hints.field_roles)
    assert len(accepted_roles) == 1, accepted_roles
    assert len(hints.joint_writes) == 1, hints.joint_writes
    assert hints.joint_writes[0]["writes"] == ((ir.registers[0].address, 1),), \
        hints.joint_writes[0]
    assert len(hints.sequences) == 1, hints.sequences
    reasons = {item["reason"] for item in hints.rejected}
    for expected in ("not a declared field", "role outside vocabulary",
                     "padding must stay zeroed", "unknown target",
                     "no legal register write", "no legal step"):
        assert expected in reasons, (expected, reasons)

    # ---- 2. accepted hints take effect -----------------------------------
    merged = build_semantic_ir(spec, role_overrides=hints.field_roles)
    before = {item.name: item.role for item in ir.fields}
    after = {item.name: item.role for item in merged.fields}
    assert after != before and all(
        after[name] == before[name] for name in before
        if name not in hints.field_roles), "overrides must touch only the hint"

    plain = UniversalPolicy(ir.action_dim, fields=[f.name for f in ir.fields],
                            spec=spec, semantic_ir=ir,
                            coverage_targets=targets)
    # Compare against the same (role-merged) IR without hints: the override
    # itself legitimately changes the role-driven families, so the honest
    # control is "merged IR, no hints".
    merged_plain = UniversalPolicy(merged.action_dim,
                                   fields=[f.name for f in merged.fields],
                                   spec=spec, semantic_ir=merged,
                                   coverage_targets=targets)
    enriched = UniversalPolicy(merged.action_dim,
                               fields=[f.name for f in merged.fields],
                               spec=spec, semantic_ir=merged,
                               coverage_targets=targets, llm_hints=hints)
    assert enriched.llm_joint_candidate_count == 1
    assert enriched.llm_sequence_candidate_count == 1
    plain_count = len(merged_plain._generic_sequence_search.candidates)
    enriched_count = len(enriched._generic_sequence_search.candidates)
    assert enriched_count == plain_count + 1, (plain_count, enriched_count)
    # The role override moved exactly one field out of `scalar`, so the
    # un-hinted policy is still a valid baseline to compare against.
    assert len(plain._generic_sequence_search.candidates) > 0
    ids = {item.candidate_id for item in enriched._generic_sequence_search.candidates}
    assert "llm.enter_wait_state" in ids, sorted(ids)

    # ---- 3. disabled is inert and silent --------------------------------
    os.environ["EDA_LLM_ENRICH"] = "0"
    silent_calls = []

    def silent_opener(request, timeout):  # pragma: no cover - must not run
        silent_calls.append(request.full_url)
        return FakeResponse({"choices": [{"message": {"content": "{}"}}]})

    off = LLMEnricher(planner=DeepSeekPlanner(opener=silent_opener))
    off_hints = off.enrich(spec, ir, targets)
    assert off_hints.status["status"] == "disabled"
    assert not off_hints.field_roles and not off_hints.joint_writes
    assert not off_hints.sequences and not silent_calls

    agent = InferenceInterface(str(BASE / "dut" / "dut_spec.md"),
                               str(covergroup))
    assert agent.llm_enrichment_status["status"] == "disabled"
    assert agent._policy.llm_joint_candidate_count == 0
    assert agent._policy.llm_sequence_candidate_count == 0
    # predict() must never reach the network
    state = np.zeros(len(agent.coverage_targets), dtype=np.float32)
    for step in range(3):
        agent.predict(state, step, 100)
    assert not silent_calls, "predict() touched the network"

    # ---- 3b. the sequence switch is honoured ----------------------------
    os.environ["EDA_LLM_ENRICH"] = "1"
    os.environ["EDA_LLM_SEQUENCES"] = "0"
    seq_off = LLMEnricher(planner=DeepSeekPlanner(opener=opener)).enrich(
        spec, ir, targets)
    assert not seq_off.sequences, seq_off.sequences
    assert seq_off.joint_writes, "joint hints must stay on when sequences are off"
    assert not seq_off.status["sequences_enabled"]
    os.environ["EDA_LLM_SEQUENCES"] = "1"

    # ---- 4. enabled but unreachable must equal disabled -------------------    # This is the property that decides whether the feature can be left on in a
    # scoring environment with no network: it has to degrade to exactly the
    # local policy, not to a broken one.
    os.environ["EDA_LLM_ENRICH"] = "1"
    os.environ["DEEPSEEK_API_KEY"] = "test-only-not-a-secret"

    def broken_opener(request, timeout):
        raise OSError("simulated network failure")

    broken = LLMEnricher(planner=DeepSeekPlanner(opener=broken_opener))
    broken_hints = broken.enrich(spec, ir, targets)
    assert broken_hints.status["status"].startswith("error_"), \
        broken_hints.status
    assert not broken_hints.field_roles and not broken_hints.joint_writes
    assert not broken_hints.sequences
    broken_policy = UniversalPolicy(ir.action_dim,
                                    fields=[f.name for f in ir.fields],
                                    spec=spec, semantic_ir=ir,
                                    coverage_targets=targets,
                                    llm_hints=broken_hints)
    assert broken_policy.llm_joint_candidate_count == 0
    assert broken_policy.llm_sequence_candidate_count == 0
    assert len(broken_policy._generic_sequence_search.candidates) == \
        len(plain._generic_sequence_search.candidates), \
        "an unreachable model must not change the candidate pool"
    assert broken_policy.joint_candidates == plain.joint_candidates
    fail_closed_ok = True

    os.environ["EDA_LLM_ENRICH"] = "1"
    os.environ.pop("DEEPSEEK_API_KEY", None)
    print(json.dumps({
        "accepted": hints.as_record()["accepted"],
        "rejected": len(hints.rejected),
        "rejection_reasons": sorted(reasons),
        "candidates_plain": plain_count,
        "candidates_enriched": enriched_count,
        "roles_available": len(SEMANTIC_ROLES),
        "disabled_is_silent": not silent_calls,
        "fail_closed": fail_closed_ok,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
