"""Structured, protocol-agnostic targets extracted from coverage metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os


@dataclass(frozen=True)
class CoverageTargetIR:
    index: int
    coverpoint: str
    bin_name: str
    kind: str
    signals: tuple[str, ...]
    values: tuple[int | float | str, ...]
    sequence: str | None
    stage: str
    difficulty: str
    source: str
    macro_hints: tuple[str, ...]
    metadata_valid: bool = True

    @property
    def target_conditions(self) -> tuple[tuple[str, int | float | str], ...]:
        return tuple(zip(self.signals, self.values))


def _macro_hints(kind: str, stage: str, name: str):
    text = f"{kind} {stage} {name}".lower()
    hints = []
    if kind in ("basic", "boundary") or "cfg" in text:
        hints.append("configure")
    if kind in ("condition", "cross"):
        hints.append("control")
    if kind in ("temporal", "sequential", "seq_hit") or any(
            token in text for token in ("fsm", "counter", "hold", "done", "window")):
        hints.append("temporal")
    if any(token in text for token in (
            "fault", "error", "overflow", "underflow", "collision", "recover")):
        hints.append("recovery")
    return tuple(dict.fromkeys(hints or ("control",)))


def load_coverage_targets(covergroup_path: str | None,
                          expected_bins: int | None = None) -> list[CoverageTargetIR]:
    if not covergroup_path:
        return []
    path = os.path.join(os.path.dirname(os.path.abspath(covergroup_path)),
                        "coverage_meta.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError, AttributeError):
        return []

    targets = []
    for coverpoint in meta.get("coverpoints", []):
        name = str(coverpoint.get("name", ""))
        kind = str(coverpoint.get("type", "basic")).lower()
        signals = tuple(item.strip().lower() for item in
                        str(coverpoint.get("signal", "")).split(",") if item.strip())
        sequence = coverpoint.get("seq")
        for item in coverpoint.get("bins", []):
            if "cross" in item:
                raw_values = item.get("cross", [])
            elif "range" in item:
                raw_values = item.get("range", [])
            elif "value" in item:
                raw_values = [item.get("value")]
            else:
                raw_values = []
            values = tuple(raw_values)
            bin_name = str(item.get("name", f"bin_{len(targets)}"))
            targets.append(CoverageTargetIR(
                index=len(targets), coverpoint=name, bin_name=bin_name,
                kind=kind, signals=signals, values=values,
                sequence=str(sequence) if sequence is not None else None,
                stage=str(coverpoint.get("stage", "")),
                difficulty=str(coverpoint.get("difficulty", "")),
                source=str(coverpoint.get("source", "")),
                macro_hints=_macro_hints(kind,
                                         str(coverpoint.get("stage", "")),
                                         f"{name} {bin_name}")))
    declared = meta.get("total_bins")
    try:
        declared = int(declared) if declared is not None else None
    except (TypeError, ValueError):
        declared = None
    valid = ((declared is None or declared == len(targets)) and
             (expected_bins is None or int(expected_bins) == len(targets)))
    if not valid:
        return []
    return targets


def missing_target_weights(state, targets):
    weights = {name: 0.0 for name in
               ("configure", "control", "temporal", "recovery")}
    missing = []
    if targets and len(targets) != len(state):
        return missing, weights
    for target in targets:
        if target.index >= len(state) or float(state[target.index]) < 0.5:
            missing.append(target)
            difficulty = {"easy": 1.0, "medium": 1.2, "hard": 1.5,
                          "hardest": 2.0}.get(target.difficulty.lower(), 1.0)
            share = difficulty / max(1, len(target.macro_hints))
            for macro in target.macro_hints:
                weights[macro] += share
    total = sum(weights.values())
    if total:
        weights = {name: value / total for name, value in weights.items()}
    return missing, weights
