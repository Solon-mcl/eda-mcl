#!/usr/bin/env python3
"""Targeted tests for variable-length Deep-Sets macro selection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from inference.coverage_controller import (  # noqa: E402
    CoverageSetController, TYPES, default_controller_path,
)


def macro_for_type(type_index):
    name = TYPES[type_index]
    if name == "boundary":
        return 1
    if name == "cross":
        return 2
    if name in ("sequential", "temporal", "seq_hit"):
        return 3
    return 0


covergroups = [
    ROOT / "public_duts/dma_xfer_public/dma_xfer_public/dut/covergroup.svh",
    ROOT / "public_duts/spi_master_public/spi_master_public/dut/covergroup.svh",
    ROOT / "public_duts/spi_xfer_public/spi_xfer_public/dut/covergroup.svh",
]
correct = total = 0
for covergroup in covergroups:
    controller = CoverageSetController(default_controller_path(), str(covergroup))
    macro_ids = np.asarray([macro_for_type(item[0]) for item in controller.descriptors])
    for target in range(4):
        if not np.any(macro_ids == target):
            continue
        state = np.ones(len(macro_ids), dtype=np.float32)
        state[macro_ids == target] = 0.0
        scores = controller.predict(state, 15000, 30000)
        selected = int(np.argmax(scores))
        print(covergroup.parent.parent.name, "missing", target,
              "selected", selected, "scores", np.round(scores, 3).tolist())
        correct += int(selected == target)
        total += 1
accuracy = correct / total
print(f"targeted_macro_accuracy={accuracy:.4f} cases={total}")
assert accuracy >= 0.80

