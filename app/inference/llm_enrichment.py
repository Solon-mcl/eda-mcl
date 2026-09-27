"""One-shot LLM semantic enrichment: hypotheses in, validated hints out.

Why an LLM here at all
----------------------
The local parser is deliberately keyword-driven, so it recovers a field's role
from names and prose but cannot be expected to *reason* about an unseen
interface.  Measured gaps that are exactly of that kind:

* `cross` targets: only 7 of SPI-master's 16 cross bins compile into a joint
  register write, because the bit-field reverse lookup cannot resolve the rest.
* `sequential` targets: 4 of 4 FSM-state bins are unreachable, because entering
  an intermediate protocol state needs a multi-cycle plan, not a single write.
* unusual naming: 79% role recall before the vocabulary work, and the residual
  gap is prose that no keyword list anticipated.

This module asks a model for those three things, but never lets it decide:

1. one request, during construction - `predict()` never touches the network;
2. every item is range-checked against the parsed IR and dropped with a reason
   if it does not fit, so a hallucinated register or field cannot reach the DUT;
3. sequence hypotheses are additionally handed to the existing online search,
   which retires a program after two attempts if it produced no coverage.

So the model's family knowledge is only ever a *hypothesis*; nothing branches on
DUT identity, and nothing it says is trusted without arithmetic evidence.
"""

from __future__ import annotations

import json
import os
import time

from .deepseek_planner import DeepSeekPlanner, _first_env, _json_object

# Roles that a hint may name.  Kept as an import-time copy of the parser's own
# vocabulary so the two cannot drift; validated again on use.
try:
    from .semantic_ir import SEMANTIC_ROLES
except ImportError:  # pragma: no cover - the parser always ships with us
    SEMANTIC_ROLES = frozenset()

MAX_SEQUENCE_STEPS = 24
MAX_SEQUENCE_CYCLES = 8192
MAX_WRITE_VALUE = (1 << 32) - 1


def _int_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number


class EnrichmentHints:
    """Validated enrichment, plus an audit of everything that was rejected."""

    def __init__(self):
        self.field_roles = {}
        self.joint_writes = []
        self.sequences = []
        self.rejected = []
        self.status = {"status": "disabled", "model": None, "latency_s": 0.0,
                       "usage": {}}

    def reject(self, kind, item, reason):
        self.rejected.append({"kind": kind, "item": item, "reason": reason})

    def as_record(self):
        return {
            "status": self.status.get("status"),
            "model": self.status.get("model"),
            "latency_s": round(float(self.status.get("latency_s", 0.0)), 3),
            "usage": self.status.get("usage", {}),
            "finish_reason": self.status.get("finish_reason"),
            "max_tokens": self.status.get("max_tokens"),
            "accepted": {
                "field_roles": len(self.field_roles),
                "joint_writes": len(self.joint_writes),
                "sequences": len(self.sequences),
            },
            "sequences_switch": self.status.get("sequences_enabled"),
            "rejected": len(self.rejected),
            "rejected_sample": self.rejected[:8],
        }


class LLMEnricher:
    """OpenAI-compatible client that only ever returns validated hints."""

    def __init__(self, planner=None, opener=None):
        self.planner = planner or DeepSeekPlanner(opener=opener)
        self.enabled = bool(self.planner.api_key) and os.environ.get(
            "EDA_LLM_ENRICH", "0").lower() in ("1", "true", "yes")
        self.override_all_roles = os.environ.get(
            "EDA_LLM_ROLE_OVERRIDE_ALL", "0").lower() in ("1", "true", "yes")
        self.max_targets = max(1, int(os.environ.get(
            "EDA_LLM_ENRICH_MAX_TARGETS", "48")))
        self.max_fields = max(1, int(os.environ.get(
            "EDA_LLM_ENRICH_MAX_FIELDS", "64")))
        # A truncated answer is worse than no answer: it looks like a response
        # but parses into a partial object.  Measured with deepseek-flash, a
        # 4096-token cap produced completion_tokens == 4096 and a body that
        # recovered as "not a JSON object", silently yielding zero hypotheses.
        self.max_tokens = min(16384, max(1024, int(os.environ.get(
            "EDA_LLM_ENRICH_MAX_TOKENS", "8192"))))
        # Sequence hypotheses put programs into the flat candidate pool, which
        # is swept rather than budgeted.  Measured on SPI-xfer (which receives
        # no joint hints, only sequences): 56 -> 38/46/47 bins over three seeds,
        # i.e. -12.3 on average.  They are therefore off by default and kept
        # switchable, while joint hints - which feed the already budget-aware
        # joint-candidate path - stay on.
        self.sequences_enabled = os.environ.get(
            "EDA_LLM_SEQUENCES", "0").lower() in ("1", "true", "yes")

    # ------------------------------------------------------------------ prompt

    def _prompt(self, spec, ir, targets, compiled_targets):
        fields = ", ".join("%s:%s" % (item.name, item.role)
                           for item in ir.fields[:self.max_fields])
        registers = "\n".join(
            "  0x%02X %s %s" % (item.address, item.name,
                                ", ".join("%s[%d:%d]" % (f.name, f.msb, f.lsb)
                                          for f in item.fields))
            for item in ir.registers[:40])
        unresolved = []
        for target in targets:
            if target.kind != "cross" or target.index in compiled_targets:
                continue
            conditions = ", ".join(
                "%s=%s" % (signal, value)
                for signal, value in (target.target_conditions or ()))
            unresolved.append("  %s.%s  <- %s" % (target.coverpoint,
                                                 target.bin_name,
                                                 conditions or "(unnamed)"))
            if len(unresolved) >= self.max_targets:
                break
        unresolved_block = "\n".join(unresolved) or "  (none)"
        return f"""You are given a hardware stimulus specification and the local
parser's current reading of it.  Produce extra semantic hints as JSON.

Already-parsed reading of the stimulus vector (name:role):
{fields}

Parsed register map (address name bit-fields):
{registers or "  (the specification declares no register map)"}

`cross` coverage targets that the local compiler could NOT turn into a joint
register write (coverpoint.bin -> required stimulus conditions):
{unresolved_block}

Return exactly one JSON object:

{{
  "field_roles": {{"<declared field name>": "<one role from the list>"}},
  "joint_writes": [
    {{"target": "<coverpoint>.<bin>",
      "writes": [[<register address>, <32-bit value>]],
      "hold_cycles": 64,
      "reason": "why these writes satisfy the conditions"}}
  ],
  "sequences": [
    {{"name": "<short label>",
      "goal": "<coverpoint.bin this is trying to reach>",
      "steps": [{{"set": {{"<declared field>": <integer>}}, "cycles": 4}}]}}
  ]
}}

Allowed roles: {", ".join(sorted(SEMANTIC_ROLES))}

Rules:
- Use only field names and target names that appear above. Never invent one.
- `writes` values must be the complete 32-bit register word to write, as an
  integer in [0, 4294967295]; addresses must be legal register addresses.
- A `sequences` entry is a multi-cycle plan to reach a state or timing bin that
  a single write cannot reach (for example an FSM intermediate state, a wait or
  a completion window). Keep each sequence under {MAX_SEQUENCE_STEPS} steps and
  give every step a positive `cycles` value.
- Keep `reason` to at most 10 words. Answer with at most 12 joint_writes and at
  most 6 sequences: a short, correct answer is far more useful than a long one,
  because the answer is truncated (and then discarded) if it does not fit.
- Empty arrays are a perfectly good answer. Do not pad the answer to look
  complete; a wrong hypothesis costs evaluation budget.
- No Markdown, no extra keys.

=== SPECIFICATION ===
{spec[:60000]}
"""

    # -------------------------------------------------------------- validation

    def _validate_roles(self, raw, hints, field_roles):
        for name, role in (raw.get("field_roles") or {}).items():
            key = str(name).strip().lower()
            value = str(role).strip().lower()
            if key not in field_roles:
                hints.reject("field_role", key, "not a declared field")
                continue
            if value not in SEMANTIC_ROLES:
                hints.reject("field_role", key, "role outside vocabulary")
                continue
            if value == "padding":
                hints.reject("field_role", key, "padding must stay zeroed")
                continue
            current = field_roles[key]
            if current != "scalar" and not self.override_all_roles:
                hints.reject("field_role", key,
                             "already resolved as %s" % current)
                continue
            if current == value:
                hints.reject("field_role", key, "no change")
                continue
            hints.field_roles[key] = value

    def _validate_joint(self, raw, hints, target_by_name, compiled_targets,
                        legal_addresses):
        for item in (raw.get("joint_writes") or [])[:self.max_targets]:
            if not isinstance(item, dict):
                hints.reject("joint_write", item, "not an object")
                continue
            target_name = str(item.get("target", "")).strip()
            target = target_by_name.get(target_name)
            if target is None:
                hints.reject("joint_write", target_name, "unknown target")
                continue
            if target.index in compiled_targets:
                hints.reject("joint_write", target_name,
                             "already compiled by the local rule")
                continue
            writes = []
            for pair in (item.get("writes") or [])[:8]:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                address = _int_or_none(pair[0])
                value = _int_or_none(pair[1])
                if address is None or value is None:
                    continue
                if not 0 <= address <= 0xFF:
                    continue
                if legal_addresses and address not in legal_addresses:
                    continue
                if not 0 <= value <= MAX_WRITE_VALUE:
                    continue
                writes.append((address, value))
            if not writes:
                hints.reject("joint_write", target_name,
                             "no legal register write")
                continue
            hold = _int_or_none(item.get("hold_cycles"))
            hold = 64 if hold is None else min(4096, max(1, hold))
            hints.joint_writes.append({
                "target_index": target.index,
                "target_name": target_name,
                "writes": tuple(sorted(writes)),
                "hold_cycles": hold,
                "reason": str(item.get("reason", ""))[:200],
            })

    def _validate_sequences(self, raw, hints, field_index):
        for item in (raw.get("sequences") or [])[:16]:
            if not isinstance(item, dict):
                hints.reject("sequence", item, "not an object")
                continue
            name = str(item.get("name", "")).strip()[:40] or "llm_sequence"
            steps = []
            total_cycles = 0
            for step in (item.get("steps") or [])[:MAX_SEQUENCE_STEPS]:
                if not isinstance(step, dict):
                    continue
                assignments = {}
                for field, value in (step.get("set") or {}).items():
                    index = field_index.get(str(field).strip().lower())
                    number = _int_or_none(value)
                    if index is None or number is None:
                        continue
                    if not 0 <= number <= MAX_WRITE_VALUE:
                        continue
                    assignments[index] = number
                if not assignments:
                    continue
                cycles = _int_or_none(step.get("cycles"))
                cycles = 1 if cycles is None else min(2048, max(1, cycles))
                total_cycles += cycles
                steps.append((assignments, cycles))
            if not steps:
                hints.reject("sequence", name, "no legal step")
                continue
            if total_cycles > MAX_SEQUENCE_CYCLES:
                hints.reject("sequence", name, "program longer than %d cycles"
                             % MAX_SEQUENCE_CYCLES)
                continue
            hints.sequences.append({
                "name": name,
                "goal": str(item.get("goal", ""))[:80],
                "steps": tuple(steps),
                "cycles": total_cycles,
            })

    def _validate(self, raw, ir, targets):
        hints = EnrichmentHints()
        if not isinstance(raw, dict):
            hints.reject("response", None, "not a JSON object")
            return hints
        field_roles = {item.name.lower(): item.role for item in ir.fields}
        field_index = {item.name.lower(): item.index for item in ir.fields}
        target_by_name = {"%s.%s" % (item.coverpoint, item.bin_name): item
                          for item in targets}
        compiled = set()
        try:
            from .joint_candidates import compile_joint_candidates
            compiled = {item.target_index
                        for item in compile_joint_candidates(ir, targets)}
        except Exception:  # pragma: no cover - degrade to "nothing compiled"
            compiled = set()
        legal_addresses = {item.address for item in ir.registers}
        self._validate_roles(raw, hints, field_roles)
        self._validate_joint(raw, hints, target_by_name, compiled,
                             legal_addresses)
        if self.sequences_enabled:
            self._validate_sequences(raw, hints, field_index)
        return hints

    # ------------------------------------------------------------------- entry

    def enrich(self, spec, ir, targets):
        if not self.enabled:
            return EnrichmentHints()
        started = time.perf_counter()
        content, status = self.planner.request_json(
            system=("You extract verified hardware-interface facts from a "
                    "specification. Obey the requested JSON schema exactly and "
                    "never invent field or target names."),
            prompt=self._prompt(spec, ir, list(targets or ()), set()),
            max_tokens=self.max_tokens,
            # This feature has its own switch; the planner's DEEPSEEK_ENABLED
            # stays independent so either can be used without the other.
            force=True)
        # request_json hands back raw text; recover the object before validating
        # it.  Feeding the string straight in silently accepted nothing.
        hints = (self._validate(_json_object(content), ir, targets)
                 if content else EnrichmentHints())
        hints.status = dict(status)
        hints.status["latency_s"] = time.perf_counter() - started
        hints.status["model"] = self.planner.model
        hints.status["sequences_enabled"] = self.sequences_enabled
        usage = hints.status.get("usage") or {}
        requested = int(hints.status.get("max_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        if (content is not None and requested and completion >= requested):
            # The endpoint does not always report finish_reason, so the token
            # count is the reliable signal.  Say so loudly instead of reporting
            # a successful run that quietly produced nothing.
            hints.status["status"] = "truncated"
        return hints


def enrichment_record(hints):
    return hints.as_record() if hints is not None else None


def emit_json(hints):
    """Small helper kept for the offline test and ad-hoc inspection."""
    return json.dumps(hints.as_record(), ensure_ascii=False, indent=2)


__all__ = ["LLMEnricher", "EnrichmentHints", "SEMANTIC_ROLES",
           "enrichment_record", "emit_json"]
