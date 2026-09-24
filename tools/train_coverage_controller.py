#!/usr/bin/env python3
"""Train a Deep-Sets macro controller on synthetic public-coverage states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from inference.coverage_controller import (  # noqa: E402
    ELEMENT_DIM, MACROS, TYPES, element_features, load_bin_descriptors,
)

METAS = [
    ROOT / "public_duts/dma_xfer_public/dma_xfer_public/dut/covergroup.svh",
    ROOT / "public_duts/spi_master_public/spi_master_public/dut/covergroup.svh",
    ROOT / "public_duts/spi_xfer_public/spi_xfer_public/dut/covergroup.svh",
]


def macro_for_type(type_index):
    name = TYPES[type_index]
    if name == "boundary":
        return 1
    if name == "cross":
        return 2
    if name in ("sequential", "temporal", "seq_hit"):
        return 3
    return 0


def make_dataset(seed=260923, samples_per_dut=700):
    rng = np.random.RandomState(seed)
    examples = []
    for meta in METAS:
        descriptors = load_bin_descriptors(str(meta))
        size = len(descriptors)
        macro_ids = np.asarray([macro_for_type(item[0]) for item in descriptors])
        for _ in range(samples_per_dut):
            progress = float(rng.rand())
            base_probability = 0.08 + 0.88 * progress
            # Easy/basic bins tend to close first; temporal bins tend to lag.
            probabilities = np.empty(size, dtype=np.float32)
            for i, macro in enumerate(macro_ids):
                offset = (0.20 if macro == 0 else 0.05 if macro == 1 else
                          -0.06 if macro == 2 else -0.14)
                probabilities[i] = np.clip(base_probability + offset +
                                           rng.normal(0, 0.08), 0.01, 0.99)
            state = (rng.rand(size) < probabilities).astype(np.float32)
            features = element_features(state, descriptors, progress)
            # Oracle utility: uncovered fraction, with early basic and late
            # temporal priors. This is learned, not evaluated at inference.
            missing = np.asarray([
                np.mean(state[macro_ids == macro] < 0.5) if np.any(macro_ids == macro) else 0
                for macro in range(4)
            ], dtype=np.float32)
            utility = missing + np.asarray([
                0.42 * (1 - progress), 0.10, 0.16 * progress, 0.38 * progress
            ], dtype=np.float32)
            label = int(np.argmax(utility))
            examples.append((features, label))
    rng.shuffle(examples)
    return examples


def train(seed=260923, epochs=70, embedding=24, hidden=32):
    rng = np.random.RandomState(seed)
    data = make_dataset(seed)
    split = int(len(data) * 0.85)
    train_data, valid_data = data[:split], data[split:]
    phi_w = (rng.randn(ELEMENT_DIM, embedding) * np.sqrt(2 / ELEMENT_DIM)).astype(np.float32)
    phi_b = np.zeros(embedding, dtype=np.float32)
    rho_w1 = (rng.randn(embedding * 2 + 3, hidden) * np.sqrt(2 / (embedding * 2 + 3))).astype(np.float32)
    rho_b1 = np.zeros(hidden, dtype=np.float32)
    rho_w2 = (rng.randn(hidden, len(MACROS)) * np.sqrt(2 / hidden)).astype(np.float32)
    rho_b2 = np.zeros(len(MACROS), dtype=np.float32)
    params = (phi_w, phi_b, rho_w1, rho_b1, rho_w2, rho_b2)
    velocity = [np.zeros_like(p) for p in params]

    def forward(elements):
        pre = elements @ phi_w + phi_b
        emb = np.maximum(pre, 0)
        hit = elements[:, 0] > 0.5
        all_pool = emb.mean(0)
        miss_pool = emb[~hit].mean(0) if np.any(~hit) else np.zeros(embedding, dtype=np.float32)
        glob = np.asarray([elements[0, 12], elements[0, 11], 0.0], dtype=np.float32)
        pooled = np.concatenate([all_pool, miss_pool, glob])
        h_pre = pooled @ rho_w1 + rho_b1
        h = np.maximum(h_pre, 0)
        return pre, emb, pooled, h_pre, h, h @ rho_w2 + rho_b2

    lr = 0.012
    for epoch in range(epochs):
        rng.shuffle(train_data)
        for elements, label in train_data:
            pre, emb, pooled, h_pre, h, logits = forward(elements)
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            probs[label] -= 1
            g_rho_w2 = np.outer(h, probs) + 1e-5 * rho_w2
            g_rho_b2 = probs
            g_h_pre = (rho_w2 @ probs) * (h_pre > 0)
            g_rho_w1 = np.outer(pooled, g_h_pre) + 1e-5 * rho_w1
            g_rho_b1 = g_h_pre
            g_pooled = rho_w1 @ g_h_pre
            g_emb = np.zeros_like(emb)
            g_emb += g_pooled[:embedding] / len(emb)
            missing = elements[:, 0] < 0.5
            if np.any(missing):
                g_emb[missing] += g_pooled[embedding:embedding * 2] / np.sum(missing)
            g_pre = g_emb * (pre > 0)
            g_phi_w = elements.T @ g_pre + 1e-5 * phi_w
            g_phi_b = g_pre.sum(0)
            grads = (g_phi_w, g_phi_b, g_rho_w1, g_rho_b1, g_rho_w2, g_rho_b2)
            for param, vel, grad in zip(params, velocity, grads):
                np.clip(grad, -2.0, 2.0, out=grad)
                vel *= 0.90
                vel += grad
                param -= lr * vel
        if epoch in (30, 52):
            lr *= 0.3

    def accuracy(dataset):
        return float(np.mean([np.argmax(forward(x)[-1]) == y for x, y in dataset]))
    metrics = {
        "architecture": f"DeepSets phi({ELEMENT_DIM},24) + rho(51,32,4)",
        "train_accuracy": accuracy(train_data),
        "validation_accuracy": accuracy(valid_data),
        "samples": len(data),
        "seed": seed,
    }
    return params, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="app/inference/model/coverage_controller.npz")
    parser.add_argument("--metrics", default="results/coverage_controller_training.json")
    args = parser.parse_args()
    params, metrics = train()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, phi_w=params[0], phi_b=params[1],
                        rho_w1=params[2], rho_b1=params[3],
                        rho_w2=params[4], rho_b2=params[5])
    metrics_path = ROOT / args.metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
