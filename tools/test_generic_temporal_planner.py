#!/usr/bin/env python3
"""Synthetic tests for the protocol-agnostic temporal macro planner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from inference import _GenericPolicy  # noqa: E402
from inference.generic_planner import (  # noqa: E402
    CoverageMacroScheduler,
    EpisodeManager,
    JointCandidateRanker,
)
from inference.generic_sequence_search import (  # noqa: E402
    GenericSequenceSearch,
    SequenceCandidate,
)
from inference.joint_candidates import JointCoverageCandidate  # noqa: E402
from inference.semantic_ir import build_semantic_ir  # noqa: E402
from inference.coverage_targets import CoverageTargetIR  # noqa: E402
from inference.universal_training import semanticize_action_sequence  # noqa: E402


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

CONTEXT_SPEC = """
## Action space (16 dims)
action = [valid, vpn0, vpn1, vpn2, vpn3, op, priv, asid, global,
          fence, flush, satp, stall, reset_n, pad0, pad1]
- `valid`: request.
- `vpn0..vpn3`: little-endian virtual address lanes.
- `op`: 0=load, 1=store, 2=fetch, 3=atomic.
- `priv`: 0=user, 1=supervisor.
- `asid`: context identifier (0~3).
- `global`: global scope.
- `fence`: context-scoped invalidation.
- `flush`: full invalidation.
- `satp`: root change.
- `stall`: backpressure.
- `reset_n`: active-low reset.
"""

CAPABILITY_SPEC = """
## Action space (20 dims)
action = [req_valid, ready, addr0, data0, txn_id, length, byte_en, qos,
          push, pop, irq_in, irq_ack, acquire, release, credit_in, sleep,
          fault_inject, abort, reset_n, pad0]
- `req_valid`: request valid.
- `ready`: receiver ready.
- `addr0`: address byte.
- `data0`: payload byte.
- `txn_id`: transaction identifier (0~7).
- `length`: transfer length (0~15).
- `byte_en`: byte mask (0~15).
- `qos`: priority (0~3).
- `push`: enqueue.
- `pop`: dequeue.
- `irq_in`: interrupt input.
- `irq_ack`: interrupt acknowledge.
- `acquire`: acquire lock.
- `release`: release lock.
- `credit_in`: replenish credit.
- `sleep`: enter low power state.
- `fault_inject`: inject fault.
- `abort`: abort and recover.
- `reset_n`: active-low reset.
- `pad0`: reserved.
"""


def test_scheduler_feedback():
    scheduler = CoverageMacroScheduler()
    assert scheduler.select(0, 0, 1000) == "configure"
    assert scheduler.select(0, 10, 1000) == "control"
    assert scheduler.select(0, 20, 1000) == "temporal"
    assert scheduler.select(5, 30, 1000) == "recovery"
    assert scheduler.select(5, 40, 1000) == "temporal"
    assert scheduler.direct_rewards["temporal"] == 5
    assert scheduler.rewards["temporal"] == 5
    assert scheduler.rewards["control"] == 2.5
    assert scheduler.rewards["configure"] == 1.25


def test_stagnation_campaign():
    scheduler = CoverageMacroScheduler()
    for step in (0, 10, 20, 30):
        scheduler.select(0, step, 1000)
    assert scheduler.select(0, 200, 1000) == "configure"
    assert scheduler.select(0, 210, 1000) == "temporal"
    assert scheduler.select(0, 220, 1000) == "control"
    assert scheduler.campaign_history[-1]["macros"] == [
        "configure", "temporal", "control"
    ]


def test_exact_causal_trace():
    scheduler = CoverageMacroScheduler()
    scheduler.select(0, 0, 1000, covered_bins=set())
    scheduler.attach_plan([{"action": [1, 0], "cycles": 2}])
    scheduler.select(2, 5, 1000, covered_bins={2, 7})
    record = scheduler.history[-1]
    assert record["new_bin_indices"] == [2, 7]
    assert record["new_bins"] == 2 and record["cycles"] == 5
    assert record["action_sequence"] == [{"action": [1, 0], "cycles": 2}]


def test_episode_manager():
    manager = EpisodeManager()
    assert manager.observe(set(), 0, 1000) is None
    assert manager.observe({3}, 70, 1000) == "coverage_gain"
    assert manager.observe({3}, 330, 1000) == "stagnation"
    assert [item["reason"] for item in manager.history] == [
        "coverage_gain", "stagnation"
    ]


def test_joint_candidate_ranker():
    first = JointCoverageCandidate(
        10, "cross.first", ((0, 1),), ("a", "b"), (), 64)
    second = JointCoverageCandidate(
        11, "cross.second", ((0, 2),), ("a", "b"), (), 64)
    ranker = JointCandidateRanker()
    assert ranker.select([first, second]) == first
    ranker.observe([{
        "cycles": 64, "new_bins": 0, "new_bin_indices": [],
        "context": {"joint_candidate": True, "target_name": "cross.first",
                    "target_index": 10},
    }])
    assert ranker.select([first, second]) == second
    assert ranker.snapshot()["cross.first"]["attempts"] == 1
    ranker.observe([
        {"cycles": 64, "new_bins": 0, "new_bin_indices": [],
         "context": {"joint_candidate": True,
                     "target_name": "cross.first", "target_index": 10}},
        {"cycles": 64, "new_bins": 0, "new_bin_indices": [],
         "context": {"joint_candidate": True,
                     "target_name": "cross.first", "target_index": 10}},
        {"cycles": 64, "new_bins": 0, "new_bin_indices": [],
         "context": {"joint_candidate": True,
                     "target_name": "cross.second", "target_index": 11}},
        {"cycles": 64, "new_bins": 0, "new_bin_indices": [],
         "context": {"joint_candidate": True,
                     "target_name": "cross.second", "target_index": 11}},
    ])
    assert ranker.select([first, second]) is None


def test_generic_sequence_search_lifecycle():
    first = SequenceCandidate("first", (("a", 1),))
    second = SequenceCandidate("second", (("b", 1),))
    search = GenericSequenceSearch((first, second))

    # Every new candidate receives one trial before scored retries begin.
    assert search.select() is first
    history = [{
        "cycles": 6, "new_bins": 3, "new_bin_indices": [1, 4, 7],
        "context": {"generic_sequence_candidate": "first"},
    }]
    gains = search.observe(history)
    assert gains == [(first, 3, (1, 4, 7))]
    assert (first.attempts, first.new_bins, first.cycles) == (1, 3, 6)
    assert search.select() is second

    history.append({
        "cycles": 4, "new_bins": 0, "new_bin_indices": [],
        "context": {"generic_sequence_candidate": "second"},
    })
    assert search.observe(history) == []
    assert search.select() is first

    history.append({
        "cycles": 2, "new_bins": 1, "new_bin_indices": [9],
        "context": {"generic_sequence_candidate": "first"},
    })
    search.observe(history)
    assert (first.attempts, first.new_bins, first.cycles) == (2, 4, 8)
    assert search.select() is second

    history.append({
        "cycles": 4, "new_bins": 0, "new_bin_indices": [],
        "context": {"generic_sequence_candidate": "second"},
    })
    search.observe(history)
    assert search.select() is None

    # A feedback-derived candidate can extend an exhausted search online.
    third = SequenceCandidate("third", (("c", 1),))
    assert search.add(third)
    assert search.select() is third

    prioritized = GenericSequenceSearch((
        SequenceCandidate("static", (("a", 1),), {"priority": 10}),
        SequenceCandidate("learned", (("b", 1),), {"priority": 30}),
    ))
    assert prioritized.select().candidate_id == "learned"
    budgeted = GenericSequenceSearch((
        SequenceCandidate("long", (("a", 100),),
                          {"priority": 30, "learned_trace": True}),
        SequenceCandidate("short", (("b", 5),), {"priority": 10}),
    ))
    assert budgeted.select(max_cycles=10).candidate_id == "short"
    assert budgeted.select(allow_learned=False).candidate_id == "short"


def test_capability_sequence_activation_after_stagnation():
    ir = build_semantic_ir(SPEC)
    policy = _GenericPolicy(ir.action_dim, [item.name for item in ir.fields],
                            SPEC, semantic_ir=ir)
    search = policy._generic_sequence_search
    assert policy.generic_sequence_search_applicable
    assert any(item.candidate_id.startswith("capability.register.")
               for item in search.candidates)
    assert not policy._generic_sequence_activation_ready(1023, 2000)
    assert policy._generic_sequence_activation_ready(1024, 2000)
    policy._generic_sequence_last_probe_step = 1024
    assert not policy._generic_sequence_activation_ready(1500, 2000)
    policy._generic_sequence_last_probe_step = 0
    assert not policy._generic_sequence_activation_ready(1800, 2000)


def test_successful_trace_mining():
    ir = build_semantic_ir(SPEC)
    policy = _GenericPolicy(ir.action_dim, [item.name for item in ir.fields],
                            SPEC, semantic_ir=ir)
    action = np.zeros(ir.action_dim, dtype=np.float32)
    action[0], action[2], action[3] = 1, 2, 0x55
    policy._macro_scheduler.history.append({
        "cycles": 4, "new_bins": 2, "new_bin_indices": [3, 5],
        "action_sequence": [{"action": action.tolist(), "cycles": 1}],
        "context": {},
    })
    before = len(policy._generic_sequence_search.candidates)
    policy._mine_successful_sequences()
    learned = [item for item in policy._generic_sequence_search.candidates
               if item.metadata.get("learned_trace")]
    assert len(policy._generic_sequence_search.candidates) > before
    assert policy._generic_trace_seed_count == 1
    assert learned and all(item.metadata["priority"] == 30
                           for item in learned)
    assert {item.metadata["mutation"] for item in learned}.issuperset(
        {"repeat2", "delay1", "hold2"})
    assert not policy._trace_exploration_allowed()
    policy._generic_trace_bootstrap_gains = 1
    assert policy._trace_exploration_allowed()
    count = len(policy._generic_sequence_search.candidates)
    policy._mine_successful_sequences()
    assert len(policy._generic_sequence_search.candidates) == count


def test_context_address_sequence_compilation():
    ir = build_semantic_ir(CONTEXT_SPEC)
    assert len(ir.indices("address_lane")) == 4
    assert len(ir.indices("context_id")) == 1
    assert len(ir.indices("privilege")) == 1
    assert len(ir.indices("scope")) == 1
    assert len(ir.indices("recovery")) == 3
    target = CoverageTargetIR(
        index=0, coverpoint="context_sequence", bin_name="hit",
        kind="seq_hit", signals=(), values=(), sequence="seq1",
        stage="MMU", difficulty="hard", source="top",
        macro_hints=("temporal",))
    policy = _GenericPolicy(
        ir.action_dim, [item.name for item in ir.fields], CONTEXT_SPEC,
        semantic_ir=ir, coverage_targets=[target])
    names = {item.candidate_id
             for item in policy._generic_sequence_search.candidates}
    assert "stream.context.alias" in names
    assert "stream.address.repeat.op1" in names
    assert "stream.scope.recovery.0.s0" in names
    assert "stream.scope.recovery.0.s1" in names


def test_extended_capability_sequence_compilation():
    ir = build_semantic_ir(CAPABILITY_SPEC)
    expected_roles = {
        "request", "ready", "address_lane", "data_lane",
        "transaction_id", "length", "mask", "priority", "queue_push",
        "queue_pop", "interrupt", "ack", "acquire", "release", "credit",
        "power", "fault", "recovery", "reset", "padding",
    }
    assert expected_roles <= {item.role for item in ir.fields}
    target = CoverageTargetIR(
        index=0, coverpoint="protocol_sequence", bin_name="hit",
        kind="seq_hit", signals=(), values=(), sequence="seq1",
        stage="PROTOCOL", difficulty="hard", source="top",
        macro_hints=("temporal",))
    policy = _GenericPolicy(
        ir.action_dim, [item.name for item in ir.fields], CAPABILITY_SPEC,
        semantic_ir=ir, coverage_targets=[target])
    names = {item.candidate_id
             for item in policy._generic_sequence_search.candidates}
    required = {
        "semantic.qualifier.id.4",
        "semantic.qualifier.length.5",
        "semantic.qualifier.mask.6",
        "semantic.qualifier.priority.7",
        "semantic.qualifier.pairwise",
        "semantic.handshake.hold4",
        "semantic.queue.fill_drain",
        "semantic.queue.simultaneous",
        "semantic.interrupt.ack_retrigger",
        "semantic.lock.acquire_release",
        "semantic.credit.consume_replenish",
        "semantic.power.toggle.15",
        "semantic.fault.recovery",
        "semantic.reset.mid_transaction.2",
    }
    assert required <= names


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
    event_tokens = set(matrix[matrix[:, 8] == 1, 3].astype(int))
    assert {0x55, 0xAA}.issubset(event_tokens)


def test_target_value_drives_action():
    ir = build_semantic_ir(SPEC)
    target = CoverageTargetIR(
        index=0, coverpoint="address_target", bin_name="address_two",
        kind="boundary", signals=("reg_addr",), values=(2,),
        sequence=None, stage="CFG", difficulty="medium", source="top",
        macro_hints=("configure",))
    policy = _GenericPolicy(ir.action_dim, [item.name for item in ir.fields],
                            SPEC, semantic_ir=ir, coverage_targets=[target])
    actions = [policy.predict(np.zeros(1), step, 1000) for step in range(12)]
    writes = [action for action in actions if action[0] == 1]
    assert writes and writes[0][2] == 2


def test_semantic_program_encoding():
    ir = build_semantic_ir(SPEC)
    program = semanticize_action_sequence(ir, [{
        "action": [1, 0, 2, 0x55, 0, 0, 0, 0, 0, 0, 0, 0],
        "cycles": 3,
    }])
    roles = {item["role"] for item in program[0]["assignments"]}
    assert "write_enable" in roles and "register_address" in roles
    assert "data_lane" in roles and program[0]["cycles"] == 3


def main():
    test_scheduler_feedback()
    test_stagnation_campaign()
    test_exact_causal_trace()
    test_episode_manager()
    test_joint_candidate_ranker()
    test_generic_sequence_search_lifecycle()
    test_capability_sequence_activation_after_stagnation()
    test_successful_trace_mining()
    test_context_address_sequence_compilation()
    test_extended_capability_sequence_compilation()
    test_synthetic_protocol()
    test_target_value_drives_action()
    test_semantic_program_encoding()
    print("generic_temporal_planner=ok")


if __name__ == "__main__":
    main()
