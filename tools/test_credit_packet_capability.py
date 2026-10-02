#!/usr/bin/env python3
"""Check that mesh packet programs require explicit, usable spec evidence."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from inference import _GenericPolicy  # noqa: E402
from inference.semantic_ir import build_semantic_ir  # noqa: E402


SPEC = """Five-port mesh router at coordinate (1,1).
Ports are LOCAL=0, NORTH=1, EAST=2, SOUTH=3, and WEST=4.
action = [valid, in_port, dest_x, dest_y, vc, flit_type,
          credit_mask, congestion_mask, router_stall, reset_n]
- `flit_type`: 0=head, 1=body, 2=tail, 3=single-flit packet.
- `credit_mask`: bit per output; one means the downstream accepts a flit.
- `congestion_mask`: bit per output indicating congestion.
- `buffer_depth` (2..5): per-input FIFO depth.
"""


def main():
    ir = build_semantic_ir(SPEC)
    fields = {item.name: item for item in ir.fields}
    assert fields["in_port"].maximum == 4
    assert fields["flit_type"].maximum == 3
    assert fields["dest_x"].maximum >= 2
    assert fields["dest_y"].maximum >= 2
    policy = _GenericPolicy(ir.action_dim, fields=list(fields), spec=SPEC,
                            semantic_ir=ir, coverage_targets=[])
    candidates = policy._build_credit_packet_candidates()
    assert {item.candidate_id.rsplit(".", 1)[-1] for item in candidates} == {
        "sweep", "packet", "blocked", "congestion", "conflict",
        "tail_recovery", "stall"}
    ordinary = SPEC.replace("credit_mask", "capacity_mask")
    other_ir = build_semantic_ir(ordinary)
    other = _GenericPolicy(other_ir.action_dim,
                           fields=[item.name for item in other_ir.fields],
                           spec=ordinary, semantic_ir=other_ir,
                           coverage_targets=[])
    assert not other._build_credit_packet_candidates()
    print("credit_packet_capability=ok")


if __name__ == "__main__":
    main()
