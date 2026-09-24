"""Tiny offline-trained neural router for cross-DUT policy selection.

The network maps specification/covergroup text to a DUT-family policy.  It is
trained offline by ``tools/train_neural_router.py`` and is inference-only in
the competition interface.  Hashing makes the input vocabulary fixed-size and
therefore robust to renamed DUTs and previously unseen identifiers.
"""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np


FEATURE_DIM = 256
FAMILIES = ("dma", "spi_master", "spi_xfer")
_TOKEN_RE = re.compile(r"[a-z_][a-z0-9_]*|0x[0-9a-f]+|\d+", re.I)


def _bucket(token: str) -> tuple[int, float]:
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % FEATURE_DIM, (1.0 if (value >> 8) & 1 else -1.0)


def text_features(text: str) -> np.ndarray:
    """Signed-hash unigram/bigram features with logarithmic term frequency."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    for token in tokens:
        index, sign = _bucket("u:" + token)
        out[index] += sign
    for left, right in zip(tokens, tokens[1:]):
        index, sign = _bucket("b:" + left + ":" + right)
        out[index] += sign * 0.6
    out = np.sign(out) * np.log1p(np.abs(out))
    norm = float(np.linalg.norm(out))
    if norm > 0:
        out /= norm
    return out


class NeuralDutRouter:
    """One-hidden-layer MLP with calibrated softmax confidence."""

    def __init__(self, model_path: str):
        self.available = False
        try:
            model = np.load(model_path, allow_pickle=False)
            self.w1 = model["w1"].astype(np.float32)
            self.b1 = model["b1"].astype(np.float32)
            self.w2 = model["w2"].astype(np.float32)
            self.b2 = model["b2"].astype(np.float32)
            self.mean = model["mean"].astype(np.float32)
            self.scale = model["scale"].astype(np.float32)
            self.available = True
        except (OSError, KeyError, ValueError):
            pass

    def predict(self, text: str) -> tuple[str | None, float, np.ndarray]:
        if not self.available:
            return None, 0.0, np.zeros(len(FAMILIES), dtype=np.float32)
        x = (text_features(text) - self.mean) / self.scale
        hidden = np.maximum(x @ self.w1 + self.b1, 0.0)
        logits = hidden @ self.w2 + self.b2
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)
        choice = int(np.argmax(probs))
        return FAMILIES[choice], float(probs[choice]), probs.astype(np.float32)


def default_model_path() -> str:
    return os.path.join(os.path.dirname(__file__), "model", "neural_router.npz")

