#!/usr/bin/env python3
"""Offline tests for schema parsing and safe generic stimulus behavior."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from inference import (  # noqa: E402
    _GenericPolicy,
    _coverage_bin_indices,
    _infer_action_dims,
    _parse_action_fields,
)
from inference.semantic_ir import build_semantic_ir  # noqa: E402


def main():
    compact = "action = [ch, d0..d3, pad0..pad6]"
    fields = _parse_action_fields(compact)
    assert fields == ["ch", "d0", "d1", "d2", "d3",
                      "pad0", "pad1", "pad2", "pad3", "pad4", "pad5", "pad6"]
    assert _infer_action_dims(compact, fields) == 12

    synthetic = """
## Action space (10 dims)
action = [bus_we, reg_addr, d0..d3, advance, trigger, reset, pad0]

- `bus_we`: write enable.
- `reg_addr`: register address (0~7).
- `d0..d3`: four little-endian 8-bit data lanes.
- `advance`: advances the internal state by one cycle.
- `trigger`: presents d0 as an event token.
- `reset`: active-high reset.
- `pad0`: reserved.

Register 0 (`CONTROL`): enable and mode.
Register 2 (`LIMIT`): lower 16 bits select the limit.
The effective constraint is 2 <= CONTROL < LIMIT.
"""
    ir = build_semantic_ir(synthetic)
    assert ir.action_dim == 10 and ir.declared_dim == 10
    assert ir.register_addresses == [0, 2]
    assert ir.indices("write_enable") == [0]
    assert ir.indices("register_address") == [1]
    assert ir.indices("advance") == [6] and ir.indices("event") == [7]
    assert ir.indices("reset") == [8] and not ir.field(8).active_low
    assert ir.field(1).maximum == 7 and ir.field(2).maximum == 255
    assert ir.indices("padding") == [9] and ir.constraints

    with tempfile.TemporaryDirectory() as directory:
        meta = Path(directory) / "coverage_meta.json"
        meta.write_text(
            '{"coverpoints":[{"name":"first","bins":[{},{}]},'
            '{"name":"xfer_done_ok","bins":[{},{},{},{}]}]}',
            encoding="utf-8")
        covergroup = Path(directory) / "covergroup.svh"
        covergroup.write_text("", encoding="utf-8")
        assert _coverage_bin_indices(str(covergroup), "xfer_done_ok") == [2, 3, 4, 5]

    active_high = _GenericPolicy(3, fields=["reset", "valid", "mystery"])
    asserted = active_high.predict(np.zeros(1, dtype=np.float32), 0, 100)
    deasserted = active_high.predict(np.zeros(1, dtype=np.float32), 1, 100)
    assert asserted[0] == 1 and deasserted[0] == 1
    deasserted = active_high.predict(np.zeros(1, dtype=np.float32), 2, 100)
    assert deasserted[0] == 0

    previous = os.environ.get("EDA_STIMULUS_SEED")
    os.environ["EDA_STIMULUS_SEED"] = "17"
    left = _GenericPolicy(1, fields=["mystery"])
    os.environ["EDA_STIMULUS_SEED"] = "18"
    right = _GenericPolicy(1, fields=["mystery"])
    left_values = [int(left.predict(np.zeros(1), step, 100)[0]) for step in range(30)]
    right_values = [int(right.predict(np.zeros(1), step, 100)[0]) for step in range(30)]
    assert set(left_values + right_values) <= {0, 1}
    assert left_values != right_values
    if previous is None:
        os.environ.pop("EDA_STIMULUS_SEED", None)
    else:
        os.environ["EDA_STIMULUS_SEED"] = previous

    print("inference_contract=ok")


if __name__ == "__main__":
    main()
