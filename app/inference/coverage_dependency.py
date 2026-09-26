"""Evidence-carrying coverage-to-stimulus dependency graph.

The graph is conservative: metadata signals are observations, while action
fields are controllable inputs. Direct edges are emitted only for exact names;
lexical/role matches remain low-confidence hypotheses for online exploration.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .coverage_targets import CoverageTargetIR
from .semantic_ir import DutSemanticIR, FieldIR


@dataclass(frozen=True)
class CoverageDependencyEdge:
    target_index: int
    field_index: int
    field_name: str
    evidence: str
    confidence: float
    reason: str
    values: tuple[int | float | str, ...] = ()


@dataclass
class CoverageDependencyGraph:
    targets: list[CoverageTargetIR]
    edges: list[CoverageDependencyEdge]

    def for_target(self, target_index: int) -> list[CoverageDependencyEdge]:
        return [edge for edge in self.edges
                if edge.target_index == int(target_index)]

    def target_confidence(self, target_index: int) -> float:
        related = self.for_target(target_index)
        return max((edge.confidence for edge in related), default=0.0)


def _tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.lower())
            if len(token) > 1 and token not in {"cov", "cfg", "cp", "bin"}}


def _role_matches(field: FieldIR, target: CoverageTargetIR) -> bool:
    text = " ".join((target.coverpoint, target.bin_name,
                     target.stage, *target.signals)).lower()
    role_tokens = {
        "reset": ("reset", "rst", "idle"),
        "register_address": ("reg", "register", "cfg", "csr"),
        "register_data": ("data", "value", "cfg", "config", "token"),
        "write_enable": ("write", "cfg", "config"),
        "read_enable": ("read", "rx"),
        "data_lane": ("data", "token", "key", "rx", "tx"),
        "address_lane": ("addr", "address", "target", "source"),
        "operation": ("op", "opcode", "mode", "kind", "type", "protocol"),
        "request": ("start", "request", "valid", "done", "xfer", "transfer"),
        "advance": ("advance", "tick", "count", "window", "fsm", "state"),
        "event": ("event", "trigger", "service", "kick", "token"),
        "fault": ("fault", "error", "overflow", "underflow", "collision"),
        "recovery": ("recover", "flush", "clear"),
        "stall": ("stall", "hold", "backpressure"),
        "enable": ("enable", "active", "busy"),
    }
    return any(token in text for token in role_tokens.get(field.role, ()))


def build_coverage_dependency_graph(
        semantic_ir: DutSemanticIR,
        targets: list[CoverageTargetIR]) -> CoverageDependencyGraph:
    edges: list[CoverageDependencyEdge] = []
    seen: set[tuple[int, int]] = set()
    for target in targets:
        target_text = " ".join((target.coverpoint, target.bin_name,
                                target.stage, *target.signals))
        target_tokens = _tokens(target_text)
        for field in semantic_ir.fields:
            key = (target.index, field.index)
            evidence = ""
            confidence = 0.0
            reason = ""
            if any(signal.lower() == field.name.lower()
                   for signal in target.signals):
                evidence, confidence = "direct", 1.0
                reason = "coverage signal exactly matches action field"
            else:
                shared = target_tokens & _tokens(field.name)
                if shared:
                    evidence, confidence = "inferred", 0.55
                    reason = "shared semantic token: " + ",".join(sorted(shared))
                elif _role_matches(field, target):
                    evidence, confidence = "inferred", 0.30
                    reason = f"field role {field.role} matches target semantics"
            if evidence and key not in seen:
                edges.append(CoverageDependencyEdge(
                    target.index, field.index, field.name, evidence,
                    confidence, reason, target.values))
                seen.add(key)
    return CoverageDependencyGraph(list(targets), edges)


def missing_field_bias(state, graph: CoverageDependencyGraph) -> dict[int, float]:
    """Return normalized action-field priors from currently missing bins."""
    weights: dict[int, float] = {}
    for target in graph.targets:
        if target.index < len(state) and float(state[target.index]) >= 0.5:
            continue
        difficulty = {"easy": 1.0, "medium": 1.2, "hard": 1.5,
                      "hardest": 2.0}.get(target.difficulty.lower(), 1.0)
        for edge in graph.for_target(target.index):
            weights[edge.field_index] = (weights.get(edge.field_index, 0.0) +
                                         difficulty * edge.confidence)
    total = sum(weights.values())
    return ({index: value / total for index, value in weights.items()}
            if total else {})
