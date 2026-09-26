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
from inference.coverage_targets import (  # noqa: E402
    CoverageTargetIR,
    load_coverage_targets,
    missing_target_weights,
)
from inference.coverage_dependency import (  # noqa: E402
    build_coverage_dependency_graph,
    missing_field_bias,
)
from inference.joint_candidates import compile_joint_candidates  # noqa: E402


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
`bus_we` is accepted only when `enable` = 1.
`trigger` requires `enable`.
`trigger` occurs after `advance` by 2..8 cycles.
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
    assert any(item.left == "control" and item.operator == ">=" and
               item.right == 2 for item in ir.constraints)
    assert any(item.left == "control" and item.operator == "<" and
               item.right == "limit" for item in ir.constraints)
    assert any(item.operation == "bus_we" and item.prerequisite == "enable" and
               item.relation == "write_condition" for item in ir.dependencies)
    assert any(item.operation == "trigger" and item.prerequisite == "enable"
               for item in ir.dependencies)
    assert ir.timing_constraints[0].minimum_cycles == 2
    assert ir.timing_constraints[0].maximum_cycles == 8

    scalar_data = build_semantic_ir("""action = [reg_we, reg_addr, reg_wdata]
    `reg_we`: write enable.
    `reg_addr`: register address 0..9.
    `reg_wdata`: 32-bit write data.""")
    assert scalar_data.indices("register_data") == [2]
    assert scalar_data.field(2).maximum == 0xFFFFFFFF
    chinese_width = build_semantic_ir("""action = [reg_wdata]
    | `reg_wdata` | 32 位写数据 |""")
    assert chinese_width.field(0).maximum == 0xFFFFFFFF
    active_low_input = build_semantic_ir("""action = [ss_in_n]
    `ss_in_n`: active-low external select input.""")
    assert active_low_input.field(0).active_low

    register_range = build_semantic_ir("""action = [reg_we, reg_addr, reg_wdata]
    | Address | Name | Mode | Description |
    | 0x02 | ENABLE | RW | global enable |
    | 0x18~0x1A | DATA | RW | FIFO data ports |""")
    assert register_range.register_addresses == [2, 24, 25, 26]

    register_policy = _GenericPolicy(
        4, fields=["reg_we", "reg_addr", "reg_wdata", "rst_n"],
        spec="""action = [reg_we, reg_addr, reg_wdata, rst_n]
        `reg_wdata`: 32-bit write data.
        `rst_n`: active-low reset.
        | Address | Name | Mode | Description |
        | 0x00 | CONTROL | RW | mode control |
        | 0x02 | ENABLE | RW | global enable |
        | 0x18~0x19 | DATA | RW | FIFO data ports |""")
    assert register_policy._register_transaction_program(0)
    queued = [action.tolist() for action, _ in register_policy.queue]
    assert any(action[0] == 1 and action[1] == 2 and action[2] == 0
               for action in queued)
    assert any(action[0] == 1 and action[1] == 2 and action[2] == 1
               for action in queued)
    assert any(action[0] == 1 and action[1] == 24 for action in queued)

    joint_ir = build_semantic_ir("""action = [reg_we, reg_addr, reg_wdata]
    `reg_wdata`: 32-bit write data.
    | Address | Name | Mode | Fields |
    | 0x00 | CONTROL | RW | [4:0]=DFS, [7:6]=FRF, [8]=SCPH, [11:10]=TMOD |
    | 0x02 | ENABLE | RW | [0]=EN |
    | 0x09 | DR | RW | [31:0]=DATA |""")
    joint_targets = [CoverageTargetIR(
        0, "protocol_x_tmod", "spi1_tmod2", "cross",
        ("protocol", "cov_tmod"), (1, 2), None, "CFG", "hardest",
        "derived", ("control",))]
    joint = compile_joint_candidates(joint_ir, joint_targets)
    assert len(joint) == 1 and joint[0].complete
    # DFS default=7, SPI1 means FRF=0/SCPH=1, and TMOD=2.
    assert joint[0].register_writes == ((0, 7 | (1 << 8) | (2 << 10)),)

    field_selected_spec = """action = [ch_sel, conf_wr, conf_field,
    d0..d3, start]
    `ch_sel`: channel select (0~3).
    `conf_wr`: configuration write enable.
    `conf_field`: 2-bit field select, 0=saddr / 1=daddr / 2=packed.
    `d0..d3`: four little-endian 8-bit data lanes.
    `start`: starts the selected channel.
    conf_data[15:0] = len
    conf_data[20:16] = dir_mode
    conf_data[23:21] = burst
    """
    field_ir = build_semantic_ir(field_selected_spec)
    assert field_ir.indices("instance_select") == [0], [
        (item.name, item.role, item.description) for item in field_ir.fields]
    assert [(item.name, item.lsb, item.msb) for item in
            field_ir.packed_fields] == [
                ("len", 0, 15), ("dir_mode", 16, 20),
                ("burst", 21, 23)]
    field_policy = _GenericPolicy(
        field_ir.action_dim, [item.name for item in field_ir.fields],
        field_selected_spec, semantic_ir=field_ir)
    assert field_policy._field_selected_transaction_program(8)
    write_cycles = [cycles for action, cycles in field_policy.queue
                    if action[1] == 1]
    assert write_cycles and set(write_cycles) == {
        field_policy.field_write_repeats}
    field_actions = [action for action, _ in field_policy.queue]
    writes = [item for item in field_actions if item[1] == 1]
    assert writes and all(item[0] == 0 for item in writes)
    assert {int(item[2]) for item in writes} >= {0, 1, 2}
    # 0xFFFFFFFE and 0xFFFFFFFF remain exact through byte lanes.
    words = [sum(int(item[3 + lane]) << (8 * lane) for lane in range(4))
             for item in writes]
    assert 0xFFFFFFFE in words and 0xFFFFFFFF in words
    assert any(item[7] == 1 for item in field_actions)

    dependency_policy = _GenericPolicy(
        3, fields=["bus_we", "enable", "pad0"],
        spec="""action = [bus_we, enable, pad0]
        `bus_we` is accepted only when `enable` = 1.
        `pad0` is reserved.""")
    constrained = dependency_policy._sanitize([1, 0, 1])
    assert constrained.tolist() == [1, 1, 0]

    with tempfile.TemporaryDirectory() as directory:
        meta = Path(directory) / "coverage_meta.json"
        meta.write_text(
            '{"coverpoints":[{"name":"first","bins":[{},{}]},'
            '{"name":"xfer_done_ok","bins":[{},{},{},{}]}]}',
            encoding="utf-8")
        covergroup = Path(directory) / "covergroup.svh"
        covergroup.write_text("", encoding="utf-8")
        assert _coverage_bin_indices(str(covergroup), "xfer_done_ok") == [2, 3, 4, 5]

        meta.write_text(
            '{"coverpoints":['
            '{"name":"level","type":"boundary","signal":"trigger",'
            '"difficulty":"medium","bins":[{"name":"zero","value":0}]},'
            '{"name":"mode_x_done","type":"cross",'
            '"signal":"mode,done","difficulty":"hardest",'
            '"bins":[{"name":"m1_done","cross":[1,1]}]},'
            '{"name":"fault_seq","type":"sequential","signal":"error",'
            '"stage":"FSM","bins":[{"name":"hit","value":1}]}]}',
            encoding="utf-8")
        targets = load_coverage_targets(str(covergroup))
        assert targets[0].target_conditions == (("trigger", 0),)
        assert targets[1].target_conditions == (("mode", 1), ("done", 1))
        assert "temporal" in targets[2].macro_hints
        missing, weights = missing_target_weights(np.asarray([1, 0, 0]), targets)
        assert [item.index for item in missing] == [1, 2]
        assert weights["temporal"] > 0 and weights["control"] > 0
        graph = build_coverage_dependency_graph(ir, targets)
        direct = graph.for_target(0)
        assert direct and direct[0].evidence == "direct"
        assert all(edge.evidence in ("direct", "inferred")
                   for edge in graph.edges)
        bias = missing_field_bias(np.asarray([0, 1, 1]), graph)
        assert abs(sum(bias.values()) - 1.0) < 1e-6

        # Metadata whose declared total disagrees with the flattened bin list
        # is rejected instead of silently misaligning coverage_state indices.
        meta.write_text(
            '{"total_bins":9,"coverpoints":[{"name":"x","bins":[{}]}]}',
            encoding="utf-8")
        assert load_coverage_targets(str(covergroup)) == []

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
