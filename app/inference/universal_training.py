"""Versioned training records for the single universal runtime policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os

import numpy as np

from .generic_planner import MACROS


ROLE_ORDER = (
    "write_enable", "read_enable", "register_address", "request",
    "operation", "advance", "event", "fault", "enable", "reset",
    "recovery", "padding", "scalar",
)
KIND_ORDER = (
    "basic", "boundary", "condition", "cross", "sequential",
    "temporal", "seq_hit",
)
SCHEMA_VERSION = 2


@dataclass
class UniversalTrainingSample:
    schema_version: int
    source_id: str
    family_split: str
    features: list[float]
    option: str
    action_sequence: list[dict]
    semantic_program: list[dict]
    cycles: int
    new_bin_indices: list[int]
    direct_reward: float
    reward_per_cycle: float

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))


def family_split_id(spec: str, targets) -> str:
    """Hash structural semantics, not DUT name, for family-level holdout."""
    roles = sorted(item.role for item in spec.fields)
    kinds = sorted(target.kind for target in targets)
    shape = json.dumps({
        "roles": roles,
        "kinds": kinds,
        "registers": len(spec.registers),
        "dependencies": len(spec.dependencies),
        "timings": len(spec.timing_constraints),
    }, sort_keys=True)
    return hashlib.sha256(shape.encode("utf-8")).hexdigest()[:16]


def encode_features(semantic_ir, targets, record, total_bins: int):
    role_counts = [len(semantic_ir.indices(role)) for role in ROLE_ORDER]
    kind_counts = [sum(target.kind == kind for target in targets)
                   for kind in KIND_ORDER]
    context = record.get("context", {})
    target_weights = context.get("target_weights", {})
    step = int(context.get("step", 0))
    max_steps = max(1, int(context.get("max_steps", 1)))
    covered = int(context.get("covered_bins", 0))
    scale = max(1, semantic_ir.action_dim)
    features = [count / scale for count in role_counts]
    features += [count / max(1, total_bins) for count in kind_counts]
    features += [float(target_weights.get(name, 0.0)) for name in MACROS]
    features += [
        semantic_ir.action_dim / 64.0,
        len(semantic_ir.registers) / 32.0,
        len(semantic_ir.constraints) / 32.0,
        len(semantic_ir.dependencies) / 32.0,
        len(semantic_ir.timing_constraints) / 16.0,
        covered / max(1, total_bins),
        step / max_steps,
    ]
    return np.asarray(features, dtype=np.float32)


def encode_runtime_features(semantic_ir, targets, covered: int,
                            total_bins: int, step: int, max_steps: int,
                            target_weights):
    return encode_features(semantic_ir, targets, {"context": {
        "step": int(step), "max_steps": int(max_steps),
        "covered_bins": int(covered),
        "target_weights": dict(target_weights),
    }}, total_bins)


def default_universal_model_path():
    return os.path.join(os.path.dirname(__file__), "model",
                        "universal_option_model.npz")


class UniversalOptionModel:
    def __init__(self, path=None):
        self.available = False
        try:
            data = np.load(path or default_universal_model_path(),
                           allow_pickle=False)
            self.weights = data["weights"].astype(np.float32)
            self.bias = data["bias"].astype(np.float32)
            approved = bool(data["quality_approved"].reshape(-1)[0])
            self.available = (self.weights.shape[0] == len(MACROS) and
                              self.bias.shape == (len(MACROS),) and approved)
        except (OSError, KeyError, ValueError):
            pass

    def predict(self, features):
        values = np.asarray(features, dtype=np.float32).reshape(-1)
        if not self.available or self.weights.shape[1] != values.size:
            return np.zeros(len(MACROS), dtype=np.float32)
        return (self.weights @ values + self.bias).astype(np.float32)


def samples_from_policy(policy, source_id: str, total_bins: int):
    split = family_split_id(policy.semantic_ir, policy.coverage_targets)
    output = []
    for record in policy._macro_scheduler.history:
        option = str(record.get("macro", ""))
        if option not in MACROS:
            continue
        cycles = max(1, int(record.get("cycles", 1)))
        reward = float(record.get("new_bins", 0))
        output.append(UniversalTrainingSample(
            schema_version=SCHEMA_VERSION,
            source_id=str(source_id), family_split=split,
            features=encode_features(policy.semantic_ir,
                                     policy.coverage_targets,
                                     record, total_bins).tolist(),
            option=option,
            action_sequence=list(record.get("action_sequence", [])),
            semantic_program=semanticize_action_sequence(
                policy.semantic_ir, record.get("action_sequence", [])),
            cycles=cycles,
            new_bin_indices=[int(value) for value in
                             record.get("new_bin_indices", [])],
            direct_reward=reward,
            reward_per_cycle=reward / cycles,
        ))
    return output


def semanticize_action_sequence(semantic_ir, action_sequence):
    """Translate raw vectors into role/ordinal assignments for distillation."""
    ordinals = {}
    field_ordinals = {}
    for field in semantic_ir.fields:
        field_ordinals[field.index] = ordinals.get(field.role, 0)
        ordinals[field.role] = field_ordinals[field.index] + 1
    program = []
    for step in action_sequence:
        vector = list(step.get("action", []))
        assignments = []
        for field in semantic_ir.fields:
            if field.index >= len(vector):
                continue
            value = float(vector[field.index])
            if value == 0 and field.role not in ("reset", "ready"):
                continue
            span = max(1, field.maximum - field.minimum)
            assignments.append({
                "role": field.role,
                "ordinal": field_ordinals[field.index],
                "normalized_value": (value - field.minimum) / span,
                "boundary": ("min" if value == field.minimum else
                             "max" if value == field.maximum else "interior"),
            })
        program.append({"assignments": assignments,
                        "cycles": int(step.get("cycles", 1))})
    return program
