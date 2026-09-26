#!/usr/bin/env python3
"""Fit a small option-value model from universal JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MACROS = ("configure", "control", "temporal", "recovery")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="training/universal_public_v1.jsonl")
    parser.add_argument("--output",
                        default="app/inference/model/universal_option_model.npz")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("no training samples")
    x = np.asarray([row["features"] for row in rows], dtype=np.float32)
    option = np.asarray([MACROS.index(row["option"]) for row in rows])
    reward = np.asarray([row["reward_per_cycle"] for row in rows],
                        dtype=np.float32)
    # Counterfactual options are unknown, so fit observed option values with a
    # ridge-regularized linear head. Online UCB remains responsible for safe
    # exploration of options absent from a state neighborhood.
    design = np.zeros((len(rows), x.shape[1] * len(MACROS) + len(MACROS)),
                      dtype=np.float32)
    for index, macro in enumerate(option):
        start = macro * x.shape[1]
        design[index, start:start + x.shape[1]] = x[index]
        design[index, x.shape[1] * len(MACROS) + macro] = 1.0
    regularization = 1e-3 * np.eye(design.shape[1], dtype=np.float32)
    weights = np.linalg.solve(design.T @ design + regularization,
                              design.T @ reward)
    feature_weights = weights[:x.shape[1] * len(MACROS)].reshape(
        len(MACROS), x.shape[1])
    bias = weights[x.shape[1] * len(MACROS):]
    prediction = np.sum(feature_weights[option] * x, axis=1) + bias[option]
    positive_by_option = [int(np.sum((option == index) & (reward > 0)))
                          for index in range(len(MACROS))]
    family_count = len({row["family_split"] for row in rows})
    quality_approved = family_count >= 3 and min(positive_by_option) >= 20
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, weights=feature_weights.astype(np.float32),
             bias=bias.astype(np.float32), macros=np.asarray(MACROS),
             feature_dim=np.asarray([x.shape[1]], dtype=np.int32),
             quality_approved=np.asarray([quality_approved], dtype=np.bool_))
    print(f"samples={len(rows)} feature_dim={x.shape[1]} "
          f"mse={float(np.mean((prediction - reward) ** 2)):.8f} "
          f"positive_by_option={positive_by_option} families={family_count} "
          f"approved={quality_approved} output={output}")


if __name__ == "__main__":
    main()
