#!/usr/bin/env python3
"""Robustness checks for renamed and partially observed DUT specifications."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from inference.neural_router import NeuralDutRouter, default_model_path  # noqa: E402

CASES = {
    "dma": ROOT / "public_duts/dma_xfer_public/dma_xfer_public",
    "spi_master": ROOT / "public_duts/spi_master_public/spi_master_public",
    "spi_xfer": ROOT / "public_duts/spi_xfer_public/spi_xfer_public",
}

rng = np.random.RandomState(240926)
router = NeuralDutRouter(default_model_path())
assert router.available
correct = total = 0
min_confidence = 1.0
for expected, base in CASES.items():
    text = ((base / "dut/dut_spec.md").read_text(encoding="utf-8") + "\n" +
            (base / "dut/covergroup.svh").read_text(encoding="utf-8"))
    for public_name in ("dma_xfer_public", "spi_master_public", "spi_xfer_public"):
        text = text.replace(public_name, "renamed_hidden_core")
    lines = [line for line in text.splitlines() if line.strip()]
    for _ in range(100):
        # Keep a random 55%-90% of the document and permute local blocks. This
        # approximates renamed DUTs and differently formatted specifications.
        kept = [line for line in lines if rng.rand() < rng.uniform(0.55, 0.90)]
        if rng.rand() < 0.5:
            rng.shuffle(kept)
        predicted, confidence, _ = router.predict("\n".join(kept))
        correct += int(predicted == expected)
        total += 1
        min_confidence = min(min_confidence, confidence)

accuracy = correct / total
print(f"mutation_accuracy={accuracy:.4f} min_confidence={min_confidence:.4f} cases={total}")
assert accuracy >= 0.98

