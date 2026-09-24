#!/usr/bin/env python3
"""Offline training for the neural DUT-family router (NumPy only)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from inference.neural_router import FEATURE_DIM, FAMILIES, text_features  # noqa: E402

SOURCES = {
    "dma": ROOT / "public_duts/dma_xfer_public/dma_xfer_public",
    "spi_master": ROOT / "public_duts/spi_master_public/spi_master_public",
    "spi_xfer": ROOT / "public_duts/spi_xfer_public/spi_xfer_public",
}


def source_text(base: Path) -> str:
    parts = [
        (base / "dut/dut_spec.md").read_text(encoding="utf-8"),
        (base / "dut/covergroup.svh").read_text(encoding="utf-8"),
    ]
    text = "\n".join(parts).lower()
    # Avoid solving the task from the literal public DUT name.
    for name in ("dma_xfer_public", "spi_master_public", "spi_xfer_public"):
        text = text.replace(name, "dut_under_test")
    return text


def augment(text: str, rng: np.random.RandomState) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    keep = rng.uniform(size=len(lines)) > rng.uniform(0.10, 0.42)
    selected = [line for line, use in zip(lines, keep) if use]
    if not selected:
        selected = lines[:1]
    # Identifier masking forces the model to use interface/coverage semantics.
    joined = "\n".join(selected)
    identifiers = sorted(set(re.findall(r"\b[a-z_][a-z0-9_]{4,}\b", joined)))
    if identifiers:
        mask_count = int(len(identifiers) * rng.uniform(0.02, 0.12))
        for token in rng.choice(identifiers, size=mask_count, replace=False):
            joined = re.sub(rf"\b{re.escape(str(token))}\b", "masked_id", joined)
    return joined


def dataset(seed: int, samples_per_family: int):
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    texts = {family: source_text(base) for family, base in SOURCES.items()}
    for label, family in enumerate(FAMILIES):
        for _ in range(samples_per_family):
            xs.append(text_features(augment(texts[family], rng)))
            ys.append(label)
    order = rng.permutation(len(xs))
    return np.stack(xs)[order], np.asarray(ys, dtype=np.int64)[order], texts


def train(seed=260923, samples=500, epochs=500, hidden=48):
    rng = np.random.RandomState(seed)
    x, y, texts = dataset(seed, samples)
    split = int(len(x) * 0.85)
    x_train, y_train = x[:split], y[:split]
    x_valid, y_valid = x[split:], y[split:]
    mean = x_train.mean(0).astype(np.float32)
    scale = np.maximum(x_train.std(0), 0.05).astype(np.float32)
    x_train = (x_train - mean) / scale
    x_valid = (x_valid - mean) / scale

    w1 = (rng.randn(FEATURE_DIM, hidden) * np.sqrt(2 / FEATURE_DIM)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = (rng.randn(hidden, len(FAMILIES)) * np.sqrt(2 / hidden)).astype(np.float32)
    b2 = np.zeros(len(FAMILIES), dtype=np.float32)
    velocity = [np.zeros_like(v) for v in (w1, b1, w2, b2)]
    lr, momentum, batch = 0.025, 0.9, 96

    for epoch in range(epochs):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            xb, yb = x_train[idx], y_train[idx]
            h_pre = xb @ w1 + b1
            h = np.maximum(h_pre, 0)
            logits = h @ w2 + b2
            logits -= logits.max(1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(1, keepdims=True)
            probs[np.arange(len(yb)), yb] -= 1
            probs /= len(yb)
            grads = [
                xb.T @ ((probs @ w2.T) * (h_pre > 0)) + 1e-4 * w1,
                ((probs @ w2.T) * (h_pre > 0)).sum(0),
                h.T @ probs + 1e-4 * w2,
                probs.sum(0),
            ]
            for param, vel, grad in zip((w1, b1, w2, b2), velocity, grads):
                vel *= momentum
                vel += grad
                param -= lr * vel
        if epoch in (250, 400):
            lr *= 0.35

    def accuracy(features, labels):
        hidden_values = np.maximum(features @ w1 + b1, 0)
        return float(np.mean(np.argmax(hidden_values @ w2 + b2, axis=1) == labels))

    full_predictions = {}
    for family, text in texts.items():
        feature = (text_features(text) - mean) / scale
        logits = np.maximum(feature @ w1 + b1, 0) @ w2 + b2
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        full_predictions[family] = {FAMILIES[i]: float(v) for i, v in enumerate(probs)}
    metrics = {
        "train_accuracy": accuracy(x_train, y_train),
        "validation_accuracy": accuracy(x_valid, y_valid),
        "samples": len(x),
        "seed": seed,
        "full_document_predictions": full_predictions,
    }
    return (w1, b1, w2, b2, mean, scale), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="app/inference/model/neural_router.npz")
    parser.add_argument("--metrics", default="results/neural_router_training.json")
    args = parser.parse_args()
    weights, metrics = train()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, w1=weights[0], b1=weights[1], w2=weights[2],
                        b2=weights[3], mean=weights[4], scale=weights[5])
    metrics_path = ROOT / args.metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
