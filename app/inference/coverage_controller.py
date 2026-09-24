"""Permutation-invariant neural controller over variable-length coverage bins."""

from __future__ import annotations

import json
import os

import numpy as np


MACROS = ("basic", "boundary", "cross", "temporal")
TYPES = ("basic", "boundary", "condition", "cross", "sequential", "temporal", "seq_hit")
TYPE_TO_INDEX = {name: index for index, name in enumerate(TYPES)}
ELEMENT_DIM = 13


def load_bin_descriptors(covergroup_path: str | None):
    """Return one descriptor per flattened coverage bin."""
    if not covergroup_path:
        return []
    meta_path = os.path.join(os.path.dirname(os.path.abspath(covergroup_path)),
                             "coverage_meta.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        return []
    descriptors = []
    coverpoints = meta.get("coverpoints", [])
    cp_denominator = max(1, len(coverpoints) - 1)
    for cp_index, cp in enumerate(coverpoints):
        bins = cp.get("bins", [])
        bin_denominator = max(1, len(bins) - 1)
        cp_type = cp.get("type", "basic")
        type_index = TYPE_TO_INDEX.get(cp_type, 0)
        for bin_index, _ in enumerate(bins):
            descriptors.append((type_index, bin_index / bin_denominator,
                                cp_index / cp_denominator))
    return descriptors


def element_features(state, descriptors, progress: float):
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    count = len(state)
    output = np.zeros((count, ELEMENT_DIM), dtype=np.float32)
    for index in range(count):
        hit = float(state[index] > 0.5)
        output[index, 0] = hit
        output[index, 1] = 1.0 - hit
        if index < len(descriptors):
            type_index, local_position, cp_position = descriptors[index]
        else:
            type_index, local_position = 0, index / max(1, count - 1)
            cp_position = local_position
        output[index, 2 + type_index] = 1.0
        output[index, 9] = local_position
        output[index, 10] = cp_position
        output[index, 11] = float(progress)
        output[index, 12] = float(np.mean(state)) if count else 0.0
    return output


class CoverageSetController:
    """Deep-Sets encoder plus MLP macro-action value head."""

    def __init__(self, model_path: str, covergroup_path: str | None):
        self.descriptors = load_bin_descriptors(covergroup_path)
        self._static = self._build_static(self.descriptors)
        self.available = False
        self.previous_covered = 0
        self.stagnation = 0
        try:
            model = np.load(model_path, allow_pickle=False)
            for name in ("phi_w", "phi_b", "rho_w1", "rho_b1", "rho_w2", "rho_b2"):
                setattr(self, name, model[name].astype(np.float32))
            self.available = True
        except (OSError, KeyError, ValueError):
            pass

    @staticmethod
    def _build_static(descriptors):
        output = np.zeros((len(descriptors), ELEMENT_DIM), dtype=np.float32)
        if not descriptors:
            return output
        type_indices = np.asarray([item[0] for item in descriptors], dtype=np.int64)
        rows = np.arange(len(descriptors))
        output[rows, 2 + type_indices] = 1.0
        output[:, 9] = np.asarray([item[1] for item in descriptors], dtype=np.float32)
        output[:, 10] = np.asarray([item[2] for item in descriptors], dtype=np.float32)
        return output

    def _features(self, state, progress):
        if len(self._static) == len(state):
            output = self._static.copy()
            hit = (state > 0.5).astype(np.float32)
            output[:, 0] = hit
            output[:, 1] = 1.0 - hit
            output[:, 11] = progress
            output[:, 12] = float(np.mean(state)) if len(state) else 0.0
            return output
        return element_features(state, self.descriptors, progress)

    def predict(self, state, step: int, max_steps: int) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        covered = int(np.sum(state))
        self.stagnation = self.stagnation + 1 if covered == self.previous_covered else 0
        self.previous_covered = covered
        progress = float(step) / max(1, int(max_steps) - 1)
        if not self.available or state.size == 0:
            # Deterministic fallback with an early-basic and late-hard prior.
            coverage = float(np.mean(state)) if state.size else 0.0
            return np.asarray([1.2 - progress, 0.8, coverage + 0.2,
                               progress + min(1.0, self.stagnation / 512)], dtype=np.float32)
        elements = self._features(state, progress)
        embeddings = np.maximum(elements @ self.phi_w + self.phi_b, 0.0)
        all_pool = embeddings.mean(axis=0)
        uncovered = state < 0.5
        missing_pool = (embeddings[uncovered].mean(axis=0) if np.any(uncovered)
                        else np.zeros(embeddings.shape[1], dtype=np.float32))
        global_features = np.asarray([
            float(np.mean(state)), progress,
            min(1.0, self.stagnation / 512.0),
        ], dtype=np.float32)
        pooled = np.concatenate([all_pool, missing_pool, global_features])
        hidden = np.maximum(pooled @ self.rho_w1 + self.rho_b1, 0.0)
        return (hidden @ self.rho_w2 + self.rho_b2).astype(np.float32)


def default_controller_path() -> str:
    return os.path.join(os.path.dirname(__file__), "model", "coverage_controller.npz")


def default_q_controller_path() -> str:
    """Controller head refit on measured RTL macro rewards."""
    return os.path.join(os.path.dirname(__file__), "model", "coverage_q_controller.npz")
