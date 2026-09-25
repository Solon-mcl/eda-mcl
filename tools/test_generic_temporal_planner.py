#!/usr/bin/env python3
"""Synthetic tests for the protocol-agnostic temporal macro planner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from inference import _GenericPolicy  # noqa: E402
from inference.generic_planner import CoverageMacroScheduler  # noqa: E402
from inference.semantic_ir import build_semantic_ir  # noqa: E402


SPEC = """
## Action space (12 dims)
action = [bus_we, bus_re, reg_addr, d0..d3, advance, trigger, error_in,
          reset, pad0]

- `bus_we`: write enable.
- `bus_re`: read enable.
- `reg_addr`: register address (0~3).
- `d0..d3`: four little-endian 8-bit data lanes.
- `advance`: advances the internal state by one cycle.
- `trigger`: presents d0 as an event token.
- `error_in`: injects an error condition.
- `reset`: active-high reset.
- `pad0`: reserved.

Register 0 (CONTROL): mode and enable.
Register 2 (LIMIT): temporal threshold.
"""


def test_scheduler_feedback():
    scheduler = CoverageMacroScheduler()
    assert scheduler.select(0, 0, 1000) == "configure"
    assert scheduler.select(0, 10, 1000) == "control"
    assert scheduler.select(0, 20, 1000) == "temporal"
    assert scheduler.select(5, 30, 1000) == "recovery"
    assert scheduler.select(5, 40, 1000) == "temporal"
    assert scheduler.rewards["temporal"] == 5


def test_synthetic_protocol():
    ir = build_semantic_ir(SPEC)
    policy = _GenericPolicy(ir.action_dim, [item.name for item in ir.fields],
                            SPEC, semantic_ir=ir)
    actions = []
    coverage = np.zeros(16, dtype=np.float32)
    for step in range(320):
        actions.append(policy.predict(coverage, step, 2000))
    matrix = np.asarray(actions)

    assert matrix.shape == (320, 12)
    assert np.all(matrix[:, 11] == 0), "reserved lane must stay zero"
    assert set(np.unique(matrix[:, 2])).issubset({0, 2})
    written = matrix[matrix[:, 0] == 1, 2]
    assert {0, 2}.issubset(set(written.astype(int)))
    assert np.any(matrix[:, 1] == 1), "read macro never ran"
    assert np.any(matrix[:, 7] == 1), "advance macro never ran"
    assert np.any(matrix[:, 8] == 1), "event macro never ran"
    assert np.any(matrix[:, 9] == 1), "fault macro never ran"
    assert np.any(matrix[4:, 10] == 1), "recovery reset never ran"
    assert set(policy.macro_trace) == {
        "configure", "control", "temporal", "recovery"
    }


def main():
    test_scheduler_feedback()
    test_synthetic_protocol()
    print("generic_temporal_planner=ok")


if __name__ == "__main__":
    main()
