"""Coverage-directed, transaction-level stimulus generator.

The evaluator imports :class:`InferenceInterface` from this package.  The
implementation deliberately depends only on NumPy and keeps all adaptation in
inference-time state; no model parameter is trained or modified.
"""

from __future__ import annotations

import os
import re
import json
from collections import deque

import numpy as np

from .coverage_controller import CoverageSetController, default_q_controller_path
from .deepseek_planner import DeepSeekPlanner
from .coverage_targets import load_coverage_targets, missing_target_weights
from .coverage_dependency import build_coverage_dependency_graph
from .joint_candidates import compile_joint_candidates
from .generic_planner import (
    CoverageMacroScheduler,
    EpisodeManager,
    JointCandidateRanker,
)
from .generic_sequence_search import GenericSequenceSearch, SequenceCandidate
from .semantic_ir import (
    DutSemanticIR,
    build_semantic_ir,
    expand_action_field,
    infer_action_dim,
    parse_action_fields,
)
from .universal_training import UniversalOptionModel, encode_runtime_features


def _read(path: str | None) -> str:
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


def _expand_action_field(raw: str) -> list[str]:
    return expand_action_field(raw)


def _parse_action_fields(spec: str) -> list[str]:
    """Extract the ordered action-vector field names from a DUT spec."""
    return parse_action_fields(spec)


def _coverage_bin_indices(covergroup_path: str | None, coverpoint_name: str):
    """Locate a coverpoint in coverage_meta.json without flat offsets."""
    if not covergroup_path:
        return []
    path = os.path.join(os.path.dirname(os.path.abspath(covergroup_path)),
                        "coverage_meta.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            coverpoints = json.load(handle).get("coverpoints", [])
    except (OSError, ValueError, AttributeError):
        return []
    offset = 0
    wanted = coverpoint_name.lower()
    for coverpoint in coverpoints:
        bins = coverpoint.get("bins", [])
        if str(coverpoint.get("name", "")).lower() == wanted:
            return list(range(offset, offset + len(bins)))
        offset += len(bins)
    return []


def _runtime_seed(default: int = 260923) -> int:
    try:
        return int(os.environ.get("EDA_STIMULUS_SEED", default))
    except ValueError:
        return default


def _infer_action_dims(spec: str, fields=None) -> int:
    return infer_action_dim(spec, fields)


class _QueuePolicy:
    """Small run-length encoded action sequencer."""

    dims = 1

    def __init__(self, seed: int | None = None):
        self.queue: deque[tuple[np.ndarray, int]] = deque()
        self.seed = _runtime_seed() if seed is None else int(seed)
        self.rng = np.random.RandomState(self.seed)

    def put(self, action, cycles: int = 1):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != self.dims:
            raise ValueError(f"expected {self.dims} action values, got {a.size}")
        self.queue.append((a, max(1, int(cycles))))

    def take(self) -> np.ndarray:
        if not self.queue:
            return np.zeros(self.dims, dtype=np.float32)
        action, left = self.queue[0]
        if left <= 1:
            self.queue.popleft()
        else:
            self.queue[0] = (action, left - 1)
        return action.copy()

    def predict(self, coverage_state, step: int, max_steps: int,
                macro_scores=None) -> np.ndarray:
        return self.take()


# ---------------------------------------------------------------------------
# Active submission policy: schema-driven, coverage-directed transaction search.
# ---------------------------------------------------------------------------

class _GenericPolicy(_QueuePolicy):
    """Schema-driven coverage fuzzer for a genuinely unseen DUT.

    It relies only on the published action declaration and common transaction
    semantics. No validation-DUT policy or learned parameter is imported.
    """

    def __init__(self, dims: int, fields=None, spec="", seed: int | None = None,
                 planned_program=None, semantic_ir: DutSemanticIR | None = None,
                 coverage_targets=None):
        self.dims = max(1, int(dims))
        super().__init__(seed)
        self.fields = list(fields or [])[:self.dims]
        self.fields += [f"field_{i}" for i in range(len(self.fields), self.dims)]
        self.spec = str(spec).lower()
        self.semantic_ir = semantic_ir or build_semantic_ir(spec)
        if not self.semantic_ir.fields and self.fields:
            declaration = "action = [" + ", ".join(self.fields) + "]"
            self.semantic_ir = build_semantic_ir(declaration)
        self._tx_index = 0
        self._last_covered = 0
        self._last_gain_step = 0
        self._last_coverage_gain_step = 0
        self._last_request = None
        self._explore_level = 0
        self._macro_scheduler = CoverageMacroScheduler()
        self._episode_manager = EpisodeManager()
        self.episode_history = self._episode_manager.history
        self.coverage_targets = list(coverage_targets or ())
        self.coverage_dependency_graph = build_coverage_dependency_graph(
            self.semantic_ir, self.coverage_targets)
        joint_enabled = os.environ.get(
            "EDA_JOINT_CANDIDATES", "1").lower() not in ("0", "false", "no")
        self.joint_candidates = (compile_joint_candidates(
            self.semantic_ir, self.coverage_targets) if joint_enabled else [])
        self._joint_candidate_by_target = {
            item.target_index: item for item in self.joint_candidates}
        self.adaptive_joint_ranking = os.environ.get(
            "EDA_ADAPTIVE_JOINT_RANKING", "1").lower() not in (
                "0", "false", "no")
        self.adaptive_joint_max_candidates = max(1, int(os.environ.get(
            "EDA_ADAPTIVE_JOINT_MAX_CANDIDATES", "8")))
        self.adaptive_joint_ranking_applied = (
            self.adaptive_joint_ranking and 0 < len(self.joint_candidates) <=
            self.adaptive_joint_max_candidates)
        configured_retry_budget = os.environ.get(
            "EDA_JOINT_MAX_FAILED_ATTEMPTS")
        joint_retry_budget = (max(1, int(configured_retry_budget))
                              if configured_retry_budget is not None else
                              (3 if len(self.joint_candidates) <= 8 else 2))
        self.joint_retry_budget = joint_retry_budget
        self._joint_ranker = (JointCandidateRanker(
            max_failed_attempts=joint_retry_budget)
                              if self.adaptive_joint_ranking_applied else None)
        self._active_target = None
        self._field_by_name = {item.name.lower(): item.index
                               for item in self.semantic_ir.fields}
        self.macro_trace = []
        self._macro_cursor = {name: 0 for name in
                              ("configure", "control", "temporal", "recovery")}
        self.field_transaction_templates = os.environ.get(
            "EDA_FIELD_TRANSACTION_TEMPLATES", "1").lower() not in (
                "0", "false", "no")
        self.robust_field_writes = os.environ.get(
            "EDA_ROBUST_FIELD_WRITES", "1").lower() not in (
                "0", "false", "no")
        self.field_write_repeats = max(1, int(os.environ.get(
            "EDA_FIELD_WRITE_REPEATS", "16")))
        self._field_transaction_cursor = 0
        self.generic_sequence_search_enabled = os.environ.get(
            "EDA_GENERIC_SEQUENCE_SEARCH", "1").lower() not in (
                "0", "false", "no")
        self.generic_trace_learning_enabled = os.environ.get(
            "EDA_GENERIC_TRACE_LEARNING", "1").lower() not in (
                "0", "false", "no")

        role_indices = self.semantic_ir.indices
        reset_all = role_indices("reset")
        self.reset_low_indices = [i for i in reset_all
                                  if (self.semantic_ir.field(i) and
                                      self.semantic_ir.field(i).active_low)]
        self.reset_high_indices = [i for i in reset_all
                                   if i not in self.reset_low_indices]
        self.reset_indices = self.reset_low_indices + self.reset_high_indices
        self.valid_indices = role_indices("request")
        self.ready_indices = role_indices("ready")
        self.op_indices = role_indices("operation")
        self.write_enable_indices = role_indices("write_enable")
        self.read_enable_indices = role_indices("read_enable")
        self.register_address_indices = role_indices("register_address")
        self.register_data_indices = role_indices("register_data")
        self.instance_select_indices = role_indices("instance_select")
        self.advance_indices = role_indices("advance")
        self.event_indices = role_indices("event")
        self.fault_indices = role_indices("fault")
        self.enable_indices = role_indices("enable")
        self.context_indices = role_indices("context_id")
        self.privilege_indices = role_indices("privilege")
        self.scope_indices = role_indices("scope")
        self.mode_indices = role_indices("mode")
        self.transaction_id_indices = role_indices("transaction_id")
        self.length_indices = role_indices("length")
        self.mask_indices = role_indices("mask")
        self.priority_indices = role_indices("priority")
        self.selector_indices = role_indices("selector")
        self.queue_push_indices = role_indices("queue_push")
        self.queue_pop_indices = role_indices("queue_pop")
        self.ack_indices = role_indices("ack")
        self.interrupt_indices = role_indices("interrupt")
        self.acquire_indices = role_indices("acquire")
        self.release_indices = role_indices("release")
        self.credit_indices = role_indices("credit")
        self.power_indices = role_indices("power")
        self.register_addresses = self.semantic_ir.register_addresses
        self.kind_indices = [i for i, name in enumerate(self.fields)
                             if name in ("kind", "branch_kind", "type")]
        self.taken_indices = [i for i, name in enumerate(self.fields)
                              if "taken" in name or name in ("outcome",)]
        self.stall_indices = role_indices("stall")
        self.flush_indices = role_indices("recovery")
        self.pad_indices = role_indices("padding")
        self.addr_lanes = self._lane_indices((r"^a(\d+)$", r"^addr_?(\d+)$",
                                              r"^vpn_?(\d+)$"))
        self.data_lanes = self._lane_indices((r"^d(\d+)$", r"^data_?(\d+)$",
                                             r"^wdata_?(\d+)$"))
        self.pc_lanes = self._lane_indices((r"^p(\d+)$", r"^pc_?(\d+)$"))
        self.target_lanes = self._lane_indices((r"^t(\d+)$",
                                               r"^target_?(\d+)$"))
        self._target_index = {
            (item.coverpoint, item.bin_name): int(item.index)
            for item in self.coverage_targets}
        self.field_selected_interface = bool(
            self.field_transaction_templates and
            self.instance_select_indices and self.register_address_indices and
            self.write_enable_indices and self.valid_indices and
            (self.data_lanes or self.register_data_indices))
        self.generic_sequence_has_explicit_targets = any(
            item.sequence is not None or item.kind in
            ("sequence", "sequential", "temporal", "seq_hit")
            for item in self.coverage_targets)
        generic_candidates = (self._build_generic_sequence_candidates()
                              if self.generic_sequence_search_enabled else ())
        self.generic_sequence_native_candidate_count = sum(
            not str(item.metadata.get("family", "")).startswith("capability_")
            for item in generic_candidates)
        self.generic_sequence_immediate = bool(
            self.generic_sequence_has_explicit_targets and
            self.generic_sequence_native_candidate_count)
        self._generic_sequence_search = GenericSequenceSearch(
            generic_candidates)
        self.generic_sequence_search_applicable = bool(
            self.generic_sequence_search_enabled and generic_candidates)
        self._generic_sequence_started = False
        self._generic_sequence_last_probe_step = 0
        self._generic_trace_cursor = 0
        self._generic_trace_seed_limit = max(1, int(os.environ.get(
            "EDA_GENERIC_TRACE_MAX_SEEDS", "32")))
        self._generic_trace_max_program_cycles = max(16, int(os.environ.get(
            "EDA_GENERIC_TRACE_MAX_PROGRAM_CYCLES", "2048")))
        self._generic_trace_seed_count = 0
        self._generic_successful_traces = []
        self._generic_trace_failure_limit = max(1, int(os.environ.get(
            "EDA_GENERIC_TRACE_FAILURE_LIMIT", "2")))
        self._generic_trace_failures = 0
        self._generic_trace_successes = 0
        self._generic_trace_bootstrap_gains = 0
        self._generic_trace_outcome_snapshot = {}

        # Functional coverage persists across reset. Begin from a known DUT
        # state, then keep active-low resets deasserted.
        if self.reset_indices:
            reset = self._base_action()
            for index in self.reset_low_indices:
                reset[index] = 0
            for index in self.reset_high_indices:
                reset[index] = 1
            self.put(reset, 2)
            self.put(self._base_action(), 2)
        for item in planned_program or ():
            action = item.get("action", [])
            if len(action) == self.dims:
                self.put(self._sanitize(action), item.get("cycles", 1))

    def _lane_indices(self, patterns):
        lanes = []
        for index, name in enumerate(self.fields):
            for pattern in patterns:
                match = re.match(pattern, name)
                if match:
                    lanes.append((int(match.group(1)), index))
                    break
        return [index for _, index in sorted(lanes)]

    def _base_action(self):
        action = np.zeros(self.dims, dtype=np.float32)
        active_low_inputs = [item.index for item in self.semantic_ir.fields
                             if item.active_low]
        for index in set(active_low_inputs + self.ready_indices):
            action[index] = 1
        return action

    def _sanitize(self, action):
        output = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if output.size != self.dims:
            raise ValueError(f"expected {self.dims} action values, got {output.size}")
        # Enforce spec-visible operation prerequisites whenever both sides are
        # controllable action fields. Unknown/internal prerequisites remain
        # descriptive IR and are not guessed.
        for dependency in self.semantic_ir.dependencies:
            if dependency.relation not in ("requires", "write_condition"):
                continue
            operation = self._field_by_name.get(dependency.operation)
            prerequisite = self._field_by_name.get(dependency.prerequisite)
            if (operation is not None and prerequisite is not None and
                    output[operation] > 0):
                field = self.semantic_ir.field(prerequisite)
                output[prerequisite] = min(1, field.maximum if field else 1)
        for index in range(self.dims):
            output[index] = self.semantic_ir.clamp(index, float(output[index]))
        for index in self.pad_indices:
            output[index] = 0
        return output

    @staticmethod
    def _canonical_signal(name):
        value = str(name).lower()
        for prefix in ("cov_", "cfg_", "conf_"):
            if value.startswith(prefix):
                value = value[len(prefix):]
        return value

    def _apply_active_target(self, action):
        target = self._active_target
        if target is None:
            return action
        for signal, value in target.target_conditions:
            if not isinstance(value, (int, float)):
                continue
            canonical = self._canonical_signal(signal)
            for field_name, index in self._field_by_name.items():
                if self._canonical_signal(field_name) == canonical:
                    action[index] = value
                    break
        return action

    @staticmethod
    def _write_lanes(action, indices, value):
        for lane, index in enumerate(indices):
            action[index] = (int(value) >> (8 * lane)) & 0xFF

    def _write_data(self, action, value):
        if self.data_lanes:
            self._write_lanes(action, self.data_lanes, value)
            return
        for index in self.register_data_indices:
            field = self.semantic_ir.field(index)
            bounded = min(field.maximum if field else 2**24,
                          max(field.minimum if field else 0, int(value)))
            encoded = np.float32(bounded)
            # np.float32(2**32-1) rounds upward to 2**32, which an adapter
            # commonly masks back to zero. Stay inside the declared range.
            if field is not None and float(encoded) > field.maximum:
                encoded = np.nextafter(encoded, np.float32(-np.inf))
            action[index] = encoded

    def _request(self, opcode, address, data, stall=False, wait=20):
        action = self._base_action()
        # Give otherwise unknown scalar fields bounded byte exploration.
        protected = set(self.reset_indices + self.ready_indices +
                        self.valid_indices + self.op_indices + self.kind_indices +
                        self.write_enable_indices + self.read_enable_indices +
                        self.register_address_indices + self.register_data_indices +
                        self.advance_indices +
                        self.event_indices + self.fault_indices + self.enable_indices +
                        self.context_indices + self.privilege_indices +
                        self.scope_indices + self.mode_indices +
                        self.transaction_id_indices + self.length_indices +
                        self.mask_indices + self.priority_indices +
                        self.selector_indices + self.queue_push_indices +
                        self.queue_pop_indices + self.ack_indices +
                        self.interrupt_indices + self.acquire_indices +
                        self.release_indices + self.credit_indices +
                        self.power_indices +
                        self.taken_indices + self.stall_indices +
                        self.flush_indices + self.pad_indices + self.addr_lanes +
                        self.data_lanes + self.pc_lanes + self.target_lanes)
        for index in range(self.dims):
            if index not in protected:
                field = self.semantic_ir.field(index)
                high = field.maximum if field else 1
                action[index] = self.rng.randint(0, min(high, 255) + 1)
        for index in self.valid_indices:
            action[index] = 1
        for index in self.op_indices:
            action[index] = int(opcode)
        self._write_lanes(action, self.addr_lanes, address)
        self._write_data(action, data)
        self.put(self._sanitize(action))
        self._last_request = action.copy()

        idle = self._base_action()
        if stall and self.ready_indices:
            blocked = idle.copy()
            for index in self.ready_indices:
                blocked[index] = 0
            self.put(blocked, 3)
        self.put(self._sanitize(idle), wait)

    def _program_step(self, assignments=None, data=None, address=None,
                      valid=False, cycles=1):
        action = self._base_action()
        for index, value in (assignments or {}).items():
            action[int(index)] = value
        if valid:
            for index in self.valid_indices:
                action[index] = 1
        if address is not None:
            self._write_lanes(action, self.addr_lanes or self.pc_lanes, address)
        if data is not None:
            lanes = self.data_lanes or self.target_lanes
            if lanes:
                self._write_lanes(action, lanes, data)
            else:
                self._write_data(action, data)
        return (self._sanitize(action), int(cycles))

    def _field_probe_values(self, index, limit=6):
        field = self.semantic_ir.field(index)
        if field is None:
            return (0, 1)
        if field.enums:
            values = sorted(field.enums)
        else:
            low, high = int(field.minimum), int(field.maximum)
            values = [low, min(high, low + 1)]
            for value in (2, 3, 0x7F, 0x80, 0xFF, high - 1, high):
                if low <= value <= high:
                    values.append(value)
        return tuple(dict.fromkeys(values))[:max(1, int(limit))]

    def _build_extended_semantic_candidates(self):
        """Compile bounded programs for common protocol capabilities."""
        candidates = []
        controls = self.op_indices or self.mode_indices
        control_value = (self._field_probe_values(controls[0], 1)[0]
                         if controls else 0)

        # Sweep scalar transaction qualifiers independently, then pair them in
        # one orthogonal sequence. This covers sizes, masks, IDs, selectors,
        # modes and QoS without a Cartesian explosion.
        qualifier_groups = (
            ("mode", self.mode_indices),
            ("id", self.transaction_id_indices),
            ("length", self.length_indices),
            ("mask", self.mask_indices),
            ("priority", self.priority_indices),
            ("selector", self.selector_indices),
        )
        qualifier_fields = []
        if self.valid_indices or self.queue_push_indices:
            for label, indices in qualifier_groups:
                for index in indices:
                    qualifier_fields.append(index)
                    steps = []
                    for ordinal, value in enumerate(
                            self._field_probe_values(index)):
                        assignments = {index: value}
                        assignments.update({item: control_value
                                            for item in controls})
                        steps.append(self._program_step(
                            assignments, data=(0, 1, 0x55, 0xAA, 0xFF)[
                                ordinal % 5],
                            address=(0, 1, 4, 16, 0xFF)[ordinal % 5],
                            valid=bool(self.valid_indices)))
                        steps.append(self._program_step())
                    candidates.append(SequenceCandidate(
                        f"semantic.qualifier.{label}.{index}", tuple(steps),
                        {"family": "semantic_qualifier", "role": label}))
            if len(qualifier_fields) >= 2:
                steps = []
                for phase in range(8):
                    assignments = {item: control_value for item in controls}
                    for offset, index in enumerate(qualifier_fields[:6]):
                        values = self._field_probe_values(index)
                        assignments[index] = values[(phase + offset) % len(values)]
                    steps.append(self._program_step(
                        assignments, data=(phase * 0x55) & 0xFF,
                        address=phase * 4, valid=bool(self.valid_indices)))
                    steps.append(self._program_step())
                candidates.append(SequenceCandidate(
                    "semantic.qualifier.pairwise", tuple(steps),
                    {"family": "semantic_pairwise"}))

        # Four-phase handshake: request, blocked hold, acceptance and explicit
        # acknowledgement. Each controllable backpressure/ack field is tested
        # separately so unrelated controls are never asserted together.
        if self.valid_indices and (self.ready_indices or self.stall_indices or
                                   self.ack_indices):
            for duration in (1, 2, 4, 8):
                steps = []
                request = {item: control_value for item in controls}
                steps.append(self._program_step(
                    request, data=0x55, address=duration, valid=True))
                for index in self.ready_indices:
                    blocked = dict(request)
                    blocked[index] = 0
                    steps.append(self._program_step(
                        blocked, data=0xAA, address=duration,
                        valid=True, cycles=duration))
                for index in self.stall_indices:
                    blocked = dict(request)
                    blocked[index] = 1
                    steps.append(self._program_step(
                        blocked, data=0xAA, address=duration,
                        valid=True, cycles=duration))
                steps.append(self._program_step(
                    request, data=0x55, address=duration, valid=True))
                for index in self.ack_indices:
                    steps.append(self._program_step({index: 1}))
                steps.append(self._program_step(cycles=duration))
                candidates.append(SequenceCandidate(
                    f"semantic.handshake.hold{duration}", tuple(steps),
                    {"family": "semantic_handshake"}))

        # FIFO/queue campaigns cover empty pop, fill, simultaneous transfer,
        # drain and wraparound using only push/pop roles.
        if self.queue_push_indices or self.queue_pop_indices:
            push = {index: 1 for index in self.queue_push_indices}
            pop = {index: 1 for index in self.queue_pop_indices}
            fill_drain = []
            for value in range(16):
                fill_drain.append(self._program_step(push, data=value))
            for _ in range(18):
                fill_drain.append(self._program_step(pop))
            candidates.append(SequenceCandidate(
                "semantic.queue.fill_drain", tuple(fill_drain),
                {"family": "semantic_queue"}))
            if self.queue_push_indices and self.queue_pop_indices:
                simultaneous = {**push, **pop}
                steps = []
                for value in (0, 1, 0x55, 0xAA, 0xFF):
                    steps.extend((self._program_step(simultaneous, data=value),
                                  self._program_step(pop),
                                  self._program_step(push, data=value)))
                candidates.append(SequenceCandidate(
                    "semantic.queue.simultaneous", tuple(steps),
                    {"family": "semantic_queue"}))

        if self.interrupt_indices and self.ack_indices:
            steps = []
            for duration in (1, 2, 4, 8):
                irq = {index: 1 for index in self.interrupt_indices}
                ack = {index: 1 for index in self.ack_indices}
                steps.extend((self._program_step(irq, cycles=duration),
                              self._program_step(),
                              self._program_step(ack),
                              self._program_step(irq),
                              self._program_step(ack)))
            candidates.append(SequenceCandidate(
                "semantic.interrupt.ack_retrigger", tuple(steps),
                {"family": "semantic_interrupt"}))

        if self.acquire_indices or self.release_indices:
            acquire = {index: 1 for index in self.acquire_indices}
            release = {index: 1 for index in self.release_indices}
            steps = [self._program_step(release)]
            for _ in range(2):
                steps.append(self._program_step(acquire))
            if self.valid_indices:
                steps.append(self._program_step(
                    {item: control_value for item in controls},
                    address=4, valid=True))
            steps.extend((self._program_step(release),
                          self._program_step(release)))
            candidates.append(SequenceCandidate(
                "semantic.lock.acquire_release", tuple(steps),
                {"family": "semantic_lock"}))

        if self.credit_indices and self.valid_indices:
            steps = []
            for _ in range(8):
                steps.append(self._program_step(valid=True))
            for index in self.credit_indices:
                for duration in (1, 2, 4):
                    steps.append(self._program_step(
                        {index: 1}, cycles=duration))
                    steps.append(self._program_step(valid=True))
            candidates.append(SequenceCandidate(
                "semantic.credit.consume_replenish", tuple(steps),
                {"family": "semantic_credit"}))

        if self.power_indices:
            for index in self.power_indices:
                steps = []
                for duration in (1, 2, 8, 16):
                    steps.extend((self._program_step({index: 1},
                                                     cycles=duration),
                                  self._program_step(cycles=2)))
                    if self.valid_indices:
                        steps.append(self._program_step(valid=True))
                candidates.append(SequenceCandidate(
                    f"semantic.power.toggle.{index}", tuple(steps),
                    {"family": "semantic_power"}))

        if self.fault_indices:
            steps = []
            if self.valid_indices:
                steps.append(self._program_step(valid=True))
            for fault_index in self.fault_indices:
                for duration in (1, 2, 4):
                    steps.append(self._program_step(
                        {fault_index: 1}, cycles=duration))
                    for recovery_index in self.flush_indices:
                        steps.append(self._program_step({recovery_index: 1}))
                    if self.valid_indices:
                        steps.append(self._program_step(valid=True))
            candidates.append(SequenceCandidate(
                "semantic.fault.recovery", tuple(steps),
                {"family": "semantic_fault_recovery"}))

        # Reset during an active transaction probes abort, state cleanup and
        # post-reset recovery. Polarity comes from the IR.
        if self.reset_indices and (self.valid_indices or
                                   self.queue_push_indices):
            for duration in (1, 2):
                asserted = self._base_action()
                for index in self.reset_low_indices:
                    asserted[index] = 0
                for index in self.reset_high_indices:
                    asserted[index] = 1
                active = {index: 1 for index in
                          (self.valid_indices + self.queue_push_indices)}
                steps = (self._program_step(active, data=0x55, address=4),
                         (self._sanitize(asserted), duration),
                         self._program_step(cycles=2),
                         self._program_step(active, data=0xAA, address=8))
                candidates.append(SequenceCandidate(
                    f"semantic.reset.mid_transaction.{duration}", steps,
                    {"family": "semantic_reset_recovery"}))

        return candidates

    def _generic_config_prefix(self):
        steps = []
        if self.reset_indices:
            reset = self._base_action()
            for index in self.reset_low_indices:
                reset[index] = 0
            for index in self.reset_high_indices:
                reset[index] = 1
            steps.extend(((self._sanitize(reset), 1),
                          (self._sanitize(self._base_action()), 1)))
        addresses = self.register_addresses or list(range(min(
            3, int(self.semantic_ir.field(
                self.register_address_indices[0]).maximum)) + 1))
        nonzero = [value for value in addresses if value != 0]
        ordered = nonzero + ([0] if 0 in addresses else [])
        for position, address in enumerate(ordered):
            value = 1 if address == 0 else 2 + 4 * position
            assignments = {index: 1 for index in self.write_enable_indices}
            assignments.update({index: address
                                for index in self.register_address_indices})
            steps.append(self._program_step(assignments, data=value))
            steps.append(self._program_step())
        return tuple(steps)

    def _build_generic_sequence_candidates(self):
        """Create role-derived programs without classifying the DUT family."""
        candidates = []
        control = self.op_indices or self.kind_indices
        address_lanes = self.addr_lanes or self.pc_lanes
        payload_lanes = self.data_lanes or self.target_lanes
        if self.valid_indices and control and address_lanes:
            field = self.semantic_ir.field(control[0])
            values = range(max(0, int(field.minimum)),
                           min(3, int(field.maximum)) + 1)
            wait = 32 if self.ready_indices else 1
            for value in values:
                for outcome in (0, 1):
                    assignments = {index: value for index in control}
                    assignments.update({index: outcome
                                        for index in self.taken_indices})
                    steps = []
                    for repeat in range(16):
                        address = 0 if repeat < 8 else (repeat % 4) * 64
                        target = address + 64
                        steps.append(self._program_step(
                            assignments, data=target if payload_lanes else None,
                            address=address, valid=True))
                        if wait > 1:
                            steps.append(self._program_step(cycles=wait))
                    candidates.append(SequenceCandidate(
                        f"stream.v{value}.o{outcome}", tuple(steps),
                        {"family": "role_stream", "control": value}))
            alternating = []
            for repeat in range(32):
                assignments = {index: list(values)[repeat % len(list(values))]
                               for index in control}
                assignments.update({index: repeat & 1
                                    for index in self.taken_indices})
                address = (repeat % 8) * 4
                alternating.append(self._program_step(
                    assignments, data=address + 4, address=address,
                    valid=True))
                if wait > 1:
                    alternating.append(self._program_step(cycles=wait))
            candidates.append(SequenceCandidate(
                "stream.alternating", tuple(alternating),
                {"family": "role_stream"}))

            # Every enum-to-enum transition is explored with a data relation
            # derived only from the address lanes. This covers stateful opcode
            # protocols and paired control-flow operations without naming them.
            for first in values:
                for second in values:
                    for stride in (4, 64):
                        steps = []
                        for repeat in range(2):
                            address = repeat * stride
                            for value, target in ((first, address + stride),
                                                  (second, address + 4)):
                                assignments = {index: value for index in control}
                                assignments.update({index: 1
                                                    for index in self.taken_indices})
                                steps.append(self._program_step(
                                    assignments, data=target, address=address,
                                    valid=True))
                                if wait > 1:
                                    steps.append(self._program_step(cycles=wait))
                        candidates.append(SequenceCandidate(
                            f"stream.transition.{first}.{second}.s{stride}",
                            tuple(steps), {"family": "enum_transition"}))

            # Exercise generic backpressure/recovery controls around a valid
            # transaction. Active-high stall/flush and active-low ready are
            # represented by their semantic roles, independent of names.
            for indices, label, asserted in (
                    (self.stall_indices, "stall", 1),
                    (self.flush_indices, "recovery", 1)):
                if not indices:
                    continue
                for ordinal, selected in enumerate(indices):
                    steps = []
                    for duration in (1, 2, 4, 8):
                        assignments = {selected: asserted}
                        assignments.update({index: 1
                                            for index in self.valid_indices})
                        steps.append(self._program_step(
                            assignments, data=64, address=0, cycles=duration))
                        steps.append(self._program_step(
                            {index: 1 for index in control}, data=64,
                            address=0, valid=True))
                        if wait > 1:
                            steps.append(self._program_step(cycles=wait))
                    candidates.append(SequenceCandidate(
                        f"stream.control.{label}.{ordinal}", tuple(steps),
                        {"family": "control_recovery"}))
            if self.ready_indices:
                steps = []
                control_values = list(values)
                dirty_value = control_values[min(1, len(control_values) - 1)]
                for ordinal, duration in enumerate((1, 2, 4, 8)):
                    base = 0x10000 + ordinal * 0x1000
                    assignments = {index: control_values[0] for index in control}
                    steps.append(self._program_step(
                        assignments, data=0xAAAAAAAA, address=base,
                        valid=True))
                    blocked = self._base_action()
                    for index in self.ready_indices:
                        blocked[index] = 0
                    steps.append((blocked, duration))
                    steps.append(self._program_step(cycles=wait))
                    # Build two persistent entries at a power-of-two stride,
                    # then access a third while backpressured. This explores
                    # both ordinary wait and replacement wait states.
                    for address in (base, base + 64):
                        assignments = {index: dirty_value for index in control}
                        steps.append(self._program_step(
                            assignments, data=0xAAAAAAAA, address=address,
                            valid=True))
                        steps.append(self._program_step(cycles=wait))
                    assignments = {index: control_values[0] for index in control}
                    steps.append(self._program_step(
                        assignments, data=0xAAAAAAAA, address=base + 128,
                        valid=True))
                    steps.append((blocked.copy(), max(4, duration)))
                    steps.append(self._program_step(cycles=wait))
                for ordinal, value in enumerate(control_values):
                    assignments = {index: value for index in control}
                    steps.append(self._program_step(
                        assignments, data=0xAAAAAAAA,
                        address=0x80000 + ordinal * 64, valid=True))
                    steps.append((blocked.copy(), 8))
                    steps.append(self._program_step(cycles=wait))
                candidates.append(SequenceCandidate(
                    "stream.control.backpressure", tuple(steps),
                    {"family": "control_recovery"}))
            # Cross-operation persistent-state program with same and aliased
            # addresses. It is legal for any enumerated operation field.
            mixed = []
            control_values = list(values)
            for round_index in range(3):
                for value in control_values:
                    for address in (0, 64, 128, 192, 0):
                        assignments = {index: value for index in control}
                        assignments.update({index: round_index & 1
                                            for index in self.taken_indices})
                        mixed.append(self._program_step(
                            assignments, data=(0xAAAAAAAA if round_index & 1
                                               else 0xFFFFFFFF),
                            address=address, valid=True))
                        if wait > 1:
                            mixed.append(self._program_step(cycles=wait))
            candidates.append(SequenceCandidate(
                "stream.mixed.persistence", tuple(mixed),
                {"family": "role_stream"}))
            occupancy = []
            for value in values:
                for address in (0, 16, 32, 48, 64, 80, 96, 112,
                                4, 8, 15):
                    assignments = {index: value for index in control}
                    occupancy.append(self._program_step(
                        assignments, data=0xAAAAAAAA, address=address,
                        valid=True))
                    if wait > 1:
                        occupancy.append(self._program_step(cycles=wait))
            candidates.append(SequenceCandidate(
                "stream.occupancy.boundaries", tuple(occupancy),
                {"family": "persistent_boundaries"}))
            for value in values:
                repeated = []
                for address in range(32):
                    assignments = {index: value for index in control}
                    assignments.update({index: 1
                                        for index in self.privilege_indices})
                    for _repeat in range(2):
                        repeated.append(self._program_step(
                            assignments, address=address, valid=True))
                        if wait > 1:
                            repeated.append(self._program_step(cycles=wait))
                candidates.append(SequenceCandidate(
                    f"stream.address.repeat.op{value}", tuple(repeated),
                    {"family": "address_repeat"}))
            if self.ready_indices:
                probes = []
                base_control = list(values)[0]
                for number in range(64):
                    for shift in (0, 2, 4, 6, 8):
                        assignments = {index: base_control for index in control}
                        probes.append(self._program_step(
                            assignments, data=number, address=number << shift,
                            valid=True))
                        probes.append(self._program_step(cycles=wait))
                candidates.append(SequenceCandidate(
                    "stream.address.multiscale_probe", tuple(probes),
                    {"family": "address_scale_search"}))

            # Context-tagged interfaces need the same address exercised under
            # multiple protection domains and scopes. Roles are inferred from
            # the schema, so this also applies to IOMMU, tagged-cache and
            # virtual-memory style interfaces without classifying the DUT.
            qualifiers = (self.context_indices + self.privilege_indices +
                          self.scope_indices)
            if qualifiers:
                contexts = [0, 1]
                if self.context_indices:
                    field = self.semantic_ir.field(self.context_indices[0])
                    contexts = list(range(min(3, int(field.maximum)) + 1))
                for operation in values:
                    for privilege in (0, 1):
                        steps = []
                        for address in (0, 1, 2, 3, 4, 8, 16, 31):
                            for context in contexts:
                                assignments = {index: operation
                                               for index in control}
                                assignments.update({index: context
                                                    for index in self.context_indices})
                                assignments.update({index: privilege
                                                    for index in self.privilege_indices})
                                assignments.update({
                                    index: (address + context) & 1
                                    for index in self.scope_indices})
                                steps.append(self._program_step(
                                    assignments, address=address, valid=True))
                                if wait > 1:
                                    steps.append(self._program_step(cycles=wait))
                        candidates.append(SequenceCandidate(
                            f"stream.context.op{operation}.p{privilege}",
                            tuple(steps), {"family": "context_transition"}))

                # Fill one address in two contexts, then revisit the first;
                # this exposes aliasing and context-sensitive replacement.
                if self.context_indices and len(contexts) > 1:
                    steps = []
                    for address in (1, 17, 33, 65):
                        for context in (contexts[0], contexts[-1], contexts[0]):
                            assignments = {index: list(values)[0]
                                           for index in control}
                            assignments.update({index: context
                                                for index in self.context_indices})
                            steps.append(self._program_step(
                                assignments, address=address, valid=True))
                            if wait > 1:
                                steps.append(self._program_step(cycles=wait))
                    candidates.append(SequenceCandidate(
                        "stream.context.alias", tuple(steps),
                        {"family": "context_alias"}))

            if self.scope_indices and self.flush_indices:
                for ordinal, recovery_index in enumerate(self.flush_indices):
                    for scope_value in (0, 1):
                        steps = []
                        for address in range(16):
                            common = {index: list(values)[0]
                                      for index in control}
                            common.update({index: 1
                                           for index in self.privilege_indices})
                            common.update({index: 0
                                           for index in self.context_indices})
                            common.update({index: scope_value
                                           for index in self.scope_indices})
                            steps.append(self._program_step(
                                common, address=address, valid=True))
                            scoped = dict(common)
                            scoped[recovery_index] = 1
                            steps.append(self._program_step(
                                scoped, address=address, valid=True))
                            steps.append(self._program_step(
                                common, address=address, valid=True))
                        candidates.append(SequenceCandidate(
                            f"stream.scope.recovery.{ordinal}.s{scope_value}",
                            tuple(steps), {"family": "scope_recovery"}))

        if (self.write_enable_indices and self.register_address_indices and
                self.event_indices and (self.data_lanes or
                                        self.register_data_indices)):
            prefix = self._generic_config_prefix()
            addresses = self.register_addresses or list(range(4))
            for address in addresses:
                for value in (0, 1, 2, 3, 0xFF):
                    steps = []
                    if self.reset_indices:
                        reset = self._base_action()
                        for index in self.reset_low_indices:
                            reset[index] = 0
                        for index in self.reset_high_indices:
                            reset[index] = 1
                        steps.extend(((self._sanitize(reset), 1),
                                      (self._sanitize(self._base_action()), 1)))
                    assignments = {index: 1 for index in self.write_enable_indices}
                    assignments.update({index: address
                                        for index in self.register_address_indices})
                    steps.append(self._program_step(assignments, data=value))
                    steps.append(self._program_step())
                    follow = addresses[(addresses.index(address) + 1) % len(addresses)]
                    assignments.update({index: follow
                                        for index in self.register_address_indices})
                    steps.append(self._program_step(assignments, data=value + 1))
                    candidates.append(SequenceCandidate(
                        f"register.boundary.{address}.{value}", tuple(steps),
                        {"family": "register_boundary"}))
            for token in range(256):
                assignments = {index: 1 for index in self.event_indices}
                steps = list(prefix)
                if self.advance_indices:
                    steps.append(self._program_step(
                        {index: 1 for index in self.advance_indices}, cycles=2))
                steps.append(self._program_step(assignments, data=token))
                candidates.append(SequenceCandidate(
                    f"token.first.{token}", tuple(steps),
                    {"family": "ordered_token", "probe_token": token}))
            for duration in (1, 2, 4, 8, 16):
                for indices, label in ((self.advance_indices, "advance"),
                                       (self.fault_indices, "fault")):
                    if not indices:
                        continue
                    steps = prefix + (self._program_step(
                        {index: 1 for index in indices}, cycles=duration),)
                    candidates.append(SequenceCandidate(
                        f"pulse.{label}.{duration}", steps,
                        {"family": "timed_pulse"}))
        candidates.extend(self._build_extended_semantic_candidates())

        # Coverage metadata is optional in the evaluation contract. Build a
        # bounded set of legal register programs from interface capabilities
        # when no explicit temporal targets are available. These candidates
        # activate only after coverage stagnates, so the established M3-M6
        # transaction builders retain the early-cycle budget.
        if ((not self.generic_sequence_has_explicit_targets or not candidates) and
                self.write_enable_indices and self.register_address_indices and
                (self.data_lanes or self.register_data_indices)):
            addresses = self._generic_addresses()
            if len(addresses) > 8:
                positions = sorted(set(
                    round(index * (len(addresses) - 1) / 7)
                    for index in range(8)))
                addresses = [addresses[index] for index in positions]
            data_index = ((self.data_lanes or
                           self.register_data_indices)[0])
            data_field = self.semantic_ir.field(data_index)
            maximum = int(data_field.maximum) if data_field else 0xFF
            values = tuple(dict.fromkeys(
                min(maximum, value) for value in (0, 1, 0x55, 0xAA, 0xFF)))

            if self.field_selected_interface:
                instance_field = self.semantic_ir.field(
                    self.instance_select_indices[0])
                instance_count = min(4, int(instance_field.maximum) + 1)
                selectors = addresses[:4]
                for instance in range(instance_count):
                    for duration in (1, 2, 4, 8):
                        steps = []
                        for ordinal, selector in enumerate(selectors):
                            assignments = {
                                index: instance
                                for index in self.instance_select_indices}
                            assignments.update({
                                index: 1 for index in self.write_enable_indices})
                            assignments.update({
                                index: selector
                                for index in self.register_address_indices})
                            steps.append(self._program_step(
                                assignments,
                                data=values[ordinal % len(values)], cycles=2))
                            steps.append(self._program_step())
                        request = {
                            index: instance
                            for index in self.instance_select_indices}
                        request.update({index: 1 for index in self.valid_indices})
                        steps.append(self._program_step(
                            request, cycles=duration))
                        steps.append(self._program_step(cycles=8 * duration))
                        candidates.append(SequenceCandidate(
                            f"capability.field.i{instance}.p{duration}",
                            tuple(steps),
                            {"family": "capability_field_sequence"}))
            else:
                for address in addresses:
                    for value in values:
                        assignments = {
                            index: 1 for index in self.write_enable_indices}
                        assignments.update({
                            index: address
                            for index in self.register_address_indices})
                        steps = [self._program_step(assignments, data=value),
                                 self._program_step()]
                        if self.read_enable_indices:
                            read = {
                                index: 1 for index in self.read_enable_indices}
                            read.update({
                                index: address
                                for index in self.register_address_indices})
                            steps.extend((self._program_step(read),
                                          self._program_step()))
                        # Repeating a legal write probes edge-sensitive and
                        # persistent register behavior without a DUT-family
                        # assumption.
                        steps.extend((self._program_step(
                            assignments, data=value, cycles=2),
                                      self._program_step(cycles=4)))
                        candidates.append(SequenceCandidate(
                            f"capability.register.a{address}.v{value}",
                            tuple(steps),
                            {"family": "capability_register_sequence"}))
        return candidates

    def _expand_generic_sequence_gains(self, gains):
        for candidate, _delta, _indices in gains:
            repeat_id = candidate.candidate_id + ".repeat2"
            self._generic_sequence_search.add(SequenceCandidate(
                repeat_id, candidate.steps + candidate.steps,
                {**candidate.metadata, "mutation": "repeat"}))
            token = candidate.metadata.get("probe_token")
            # Any token-specific gain may represent an intermediate protocol
            # state. False positives are bounded by the per-candidate retry
            # cap; filtering by token value would make low-valued secrets
            # undiscoverable.
            if token is not None:
                for second in range(256):
                    assignments = {index: 1 for index in self.event_indices}
                    second_step = self._program_step(assignments, data=second)
                    self._generic_sequence_search.add(SequenceCandidate(
                        f"token.pair.{token}.{second}",
                        candidate.steps + (second_step,),
                        {"family": "ordered_token", "first_token": int(token),
                         "second_token": second}))
            first = candidate.metadata.get("first_token")
            second = candidate.metadata.get("second_token")
            if first is None or second is None:
                continue
            prefix = self._generic_config_prefix()
            for delay in (0, 1, 2, 4, 8, 16):
                steps = list(prefix)
                if delay and self.advance_indices:
                    steps.append(self._program_step(
                        {index: 1 for index in self.advance_indices},
                        cycles=delay))
                pulse = {index: 1 for index in self.event_indices}
                steps.append(self._program_step(pulse, data=int(first)))
                steps.append(self._program_step(pulse, data=int(second)))
                self._generic_sequence_search.add(SequenceCandidate(
                    f"token.timing.{first}.{second}.{delay}", tuple(steps),
                    {"family": "ordered_token_timing", "delay": delay}))

    def _history_steps(self, record):
        steps = []
        for item in record.get("action_sequence", ()):
            action = np.asarray(item.get("action", ()), dtype=np.float32)
            if action.size != self.dims or not np.all(np.isfinite(action)):
                return ()
            steps.append((self._sanitize(action),
                          max(1, int(item.get("cycles", 1)))))
        return tuple(steps)

    def _add_trace_candidate(self, candidate_id, steps, family, source,
                             mutation):
        if (not steps or sum(int(cycles) for _action, cycles in steps) >
                self._generic_trace_max_program_cycles):
            return False
        added = self._generic_sequence_search.add(SequenceCandidate(
            candidate_id, tuple(steps), {
                "family": family,
                "mutation": mutation,
                "learned_trace": True,
                "priority": 30,
                "source_new_bins": tuple(source.get(
                    "new_bin_indices", ())),
            }))
        if added:
            self.generic_sequence_search_applicable = True
        return added

    def _mutate_successful_trace(self, source_id, steps, record):
        """Derive bounded, schema-legal programs from a rewarded transaction."""
        prefix = f"learned.{source_id}"
        idle = (self._sanitize(self._base_action()), 1)
        self._add_trace_candidate(
            prefix + ".repeat2", steps + steps,
            "learned_trace_repeat", record, "repeat2")
        for delay in (1, 4, 16):
            delayed = steps + ((idle[0].copy(), delay),) + steps
            self._add_trace_candidate(
                f"{prefix}.delay{delay}", delayed,
                "learned_trace_timing", record, f"delay{delay}")

        trigger_indices = (self.valid_indices + self.event_indices +
                           self.fault_indices + self.write_enable_indices +
                           self.read_enable_indices)
        trigger_position = next((position for position, (action, _cycles)
                                 in enumerate(steps)
                                 if any(action[index] > 0
                                        for index in trigger_indices)), None)
        if trigger_position is not None:
            for duration in (2, 4):
                held = [(action.copy(), cycles) for action, cycles in steps]
                action, _cycles = held[trigger_position]
                held[trigger_position] = (action, duration)
                self._add_trace_candidate(
                    f"{prefix}.hold{duration}", tuple(held),
                    "learned_trace_hold", record, f"hold{duration}")

        mutable_indices = list(dict.fromkeys(
            self.op_indices + self.kind_indices + self.instance_select_indices +
            self.register_address_indices + self.addr_lanes + self.pc_lanes +
            self.data_lanes + self.register_data_indices))
        mutations = 0
        for index in mutable_indices:
            position = next((position for position, (action, _cycles)
                             in enumerate(steps)
                             if action[index] != self._base_action()[index]),
                            None)
            if position is None:
                continue
            varied = [(action.copy(), cycles) for action, cycles in steps]
            action, cycles = varied[position]
            field = self.semantic_ir.field(index)
            original = float(action[index])
            candidate_value = original + 1
            if field is not None and candidate_value > field.maximum:
                candidate_value = original - 1
            action[index] = self.semantic_ir.clamp(index, candidate_value)
            if action[index] == original:
                continue
            varied[position] = (self._sanitize(action), cycles)
            self._add_trace_candidate(
                f"{prefix}.neighbor{index}", tuple(varied),
                "learned_trace_neighbor", record, f"neighbor_field_{index}")
            mutations += 1
            if mutations >= 2:
                break

        if self._generic_successful_traces:
            previous_id, previous = self._generic_successful_traces[-1]
            self._add_trace_candidate(
                f"{prefix}.after{previous_id}", previous + steps,
                "learned_trace_composition", record, "previous_then_current")
            self._add_trace_candidate(
                f"{prefix}.before{previous_id}", steps + previous,
                "learned_trace_composition", record, "current_then_previous")

    def _mine_successful_sequences(self):
        if not self.generic_trace_learning_enabled:
            return
        history = self._macro_scheduler.history
        for position, record in enumerate(
                history[self._generic_trace_cursor:],
                start=self._generic_trace_cursor):
            if self._generic_trace_seed_count >= self._generic_trace_seed_limit:
                break
            context = record.get("context", {})
            if (context.get("generic_sequence_search") or
                    int(record.get("new_bins", 0)) <= 0):
                continue
            steps = self._history_steps(record)
            if not steps:
                continue
            source_id = f"s{self._generic_trace_seed_count}.h{position}"
            self._mutate_successful_trace(source_id, steps, record)
            self._generic_successful_traces.append((source_id, steps))
            self._generic_successful_traces = self._generic_successful_traces[-8:]
            self._generic_trace_seed_count += 1
        self._generic_trace_cursor = len(history)

    def _observe_trace_candidate_outcomes(self):
        for candidate in self._generic_sequence_search.candidates:
            if not candidate.metadata.get("learned_trace"):
                continue
            previous_attempts, previous_bins = (
                self._generic_trace_outcome_snapshot.get(
                    candidate.candidate_id, (0, 0)))
            if candidate.attempts <= previous_attempts:
                continue
            attempt_delta = candidate.attempts - previous_attempts
            reward_delta = candidate.new_bins - previous_bins
            if reward_delta > 0:
                self._generic_trace_successes += reward_delta
                self._generic_trace_failures += max(0, attempt_delta - 1)
            else:
                self._generic_trace_failures += attempt_delta
            self._generic_trace_outcome_snapshot[candidate.candidate_id] = (
                candidate.attempts, candidate.new_bins)

    def _trace_exploration_allowed(self):
        allowance = (self._generic_trace_failure_limit +
                     2 * self._generic_trace_successes)
        return (self.generic_trace_learning_enabled and
                self._generic_trace_bootstrap_gains > 0 and
                self._generic_trace_failures < allowance)

    def _generic_sequence_activation_ready(self, step, max_steps):
        if not self.generic_sequence_search_applicable:
            return False
        if self.generic_sequence_immediate:
            return True
        # Preserve the final sixth of the budget for the established policy
        # after exploratory trace mutations have had their opportunity.
        if int(step) >= int(max_steps) * 5 // 6:
            return False
        patience = max(1024, min(4096, int(max_steps) // 10))
        last_activity = max(self._last_coverage_gain_step,
                            self._generic_sequence_last_probe_step)
        return int(step) - last_activity >= patience

    def _next_generic_sequence(self, step, max_steps):
        search = self._generic_sequence_search
        gains = search.observe(self._macro_scheduler.history)
        self._generic_trace_bootstrap_gains += sum(
            int(delta) for candidate, delta, _indices in gains
            if not candidate.metadata.get("learned_trace"))
        self._observe_trace_candidate_outcomes()
        self._expand_generic_sequence_gains(gains)
        self._mine_successful_sequences()
        if not self._generic_sequence_activation_ready(step, max_steps):
            return False
        max_candidate_cycles = None
        if not self.generic_sequence_immediate:
            remaining = max(0, int(max_steps) - int(step))
            max_candidate_cycles = max(16, remaining // 10)
        candidate = search.select(
            max_cycles=max_candidate_cycles,
            allow_learned=self._trace_exploration_allowed())
        if candidate is None:
            return False
        self._generic_sequence_started = True
        if not self.generic_sequence_immediate:
            self._generic_sequence_last_probe_step = int(step)
        for action, cycles in candidate.steps:
            self.put(action, cycles)
        self._macro_scheduler.attach_context(
            generic_sequence_search=True,
            generic_sequence_candidate=candidate.candidate_id,
            sequence_family=candidate.metadata.get("family", "generic"))
        return True

    def _generic_transaction(self):
        if (getattr(self, "field_selected_interface", False) and
                self._next_field_selected_transaction()):
            self._tx_index += 1
            return
        index = self._tx_index
        boundary = (0, 1, 4, 8, 15, 0x7F, 0x80, 0xFF,
                    0xFFFF, 0xFFFFFFFF)[index % 10]
        boundary ^= (self._explore_level * 0x101) & 0xFFFFFFFF
        if self.register_address_indices and self.write_enable_indices:
            # Generic register-bus discovery. Addresses come from the parsed
            # register map when available; otherwise use legal field bounds.
            addr_field = self.semantic_ir.field(self.register_address_indices[0])
            fallback_max = min(15, addr_field.maximum if addr_field else 3)
            addresses = self.register_addresses or list(range(fallback_max + 1))
            address = addresses[(index // 4) % len(addresses)]
            phase = index % 4
            action = self._base_action()
            if phase < 2:
                for item in self.write_enable_indices:
                    action[item] = 1
                for item in self.register_address_indices:
                    action[item] = address
                self._write_data(action, boundary)
                self.put(self._sanitize(action))
                self.put(self._sanitize(self._base_action()), 2)
            else:
                controls = (self.advance_indices + self.event_indices +
                            self.fault_indices + self.enable_indices +
                            self.read_enable_indices)
                if controls:
                    item = controls[((index // 2) + self._explore_level) % len(controls)]
                    action[item] = 1
                    # Event-style inputs commonly consume a data/token value.
                    self._write_data(action, boundary)
                    hold = (1, 2, 4, 8)[self._explore_level % 4]
                    self.put(self._sanitize(action), hold)
                    self.put(self._sanitize(self._base_action()), 2)
                else:
                    self._request(index & 3, boundary, boundary,
                                  stall=(index % 7 == 0), wait=8)
            self._tx_index += 1
            return
        opcode = (index + self._explore_level) & 3
        data = ((0, 1, 0xAAAAAAAA, 0xFFFFFFFF,
                 index * 0x9E3779B1)[index % 5] ^
                (self._explore_level * 0x01010101))
        wait = (8, 20, 40, 80)[self._explore_level % 4]
        self._request(opcode, boundary, data, stall=(index % 7 == 0), wait=wait)
        self._tx_index += 1

    def _generic_addresses(self):
        if not self.register_address_indices:
            return [0]
        field = self.semantic_ir.field(self.register_address_indices[0])
        fallback_max = min(15, field.maximum if field else 3)
        return self.register_addresses or list(range(fallback_max + 1))

    @staticmethod
    def _boundary_value(index):
        values = (0, 1, 2, 3, 0x7F, 0x80, 0xFF, 0xAAAA,
                  0x5555, 0xFFFF, 0xFFFFFFFF)
        return values[int(index) % len(values)]

    def _put_register_write(self, address, value):
        action = self._base_action()
        for item in self.write_enable_indices:
            action[item] = 1
        for item in self.register_address_indices:
            action[item] = address
        self._write_data(action, value)
        self._apply_active_target(action)
        self.put(self._sanitize(action))
        self.put(self._sanitize(self._base_action()), 2)

    @staticmethod
    def _set_packed_value(word, field, value):
        width_mask = (1 << (field.msb - field.lsb + 1)) - 1
        return ((int(word) & ~field.mask) |
                ((int(value) & width_mask) << field.lsb))

    def _put_field_write(self, instance, selector, value):
        action = self._base_action()
        for item in self.instance_select_indices:
            action[item] = instance
        for item in self.write_enable_indices:
            action[item] = 1
        for item in self.register_address_indices:
            action[item] = selector
        self._write_data(action, value)
        # A black-box target may reject writes while an internal completion or
        # hold window is active.  Bounded level-sensitive retries guarantee an
        # acceptance opportunity without reading private DUT signals.  They
        # also make the observation cookie settle on RTL with nonblocking
        # register updates before the next field is programmed.
        repeats = (self.field_write_repeats
                   if self.robust_field_writes else 1)
        self.put(self._sanitize(action), repeats)
        self.put(self._sanitize(self._base_action()))

    def _field_selected_transaction_program(self, cursor):
        """Compile a complete transaction for selector-addressed resources.

        This covers interfaces where one action field chooses an instance
        (channel/port), another chooses a configuration field, and byte lanes
        carry the value. All structure is taken from the semantic IR.
        """
        if not (self.instance_select_indices and self.register_address_indices and
                self.write_enable_indices and self.valid_indices and
                (self.data_lanes or self.register_data_indices)):
            return False
        instance_field = self.semantic_ir.field(self.instance_select_indices[0])
        selector_field = self.semantic_ir.field(
            self.register_address_indices[0])
        instance_count = min(16, int(instance_field.maximum) + 1)
        selector_max = min(15, int(selector_field.maximum))
        enum_values = sorted(selector_field.enums) if selector_field else []
        selector_values = sorted(set(enum_values + list(range(selector_max + 1))))
        # Selector fields are often wider than the documented legal values.
        # When at least two named fields exist, include one remaining value as
        # the packed/control field and avoid probing unrelated encodings.
        if len(enum_values) >= 2:
            selector_values = list(range(min(selector_max, max(enum_values) + 1) + 1))
        if len(selector_values) < 3:
            return False

        labels = {value: str(selector_field.enums.get(value, "")).lower()
                  for value in selector_values}
        source_selector = next((value for value, label in labels.items()
                                if "saddr" in label or "source" in label),
                               selector_values[0])
        dest_selector = next((value for value, label in labels.items()
                              if "daddr" in label or "dest" in label),
                             selector_values[1])
        packed_selector = next((value for value in selector_values
                                if value not in (source_selector, dest_selector)),
                               selector_values[2])

        instance = int(cursor) % max(1, instance_count)
        batch = int(cursor) // max(1, instance_count)
        address_values = (0, 1, 0xFFFFFFFE, 0xFFFFFFFF)
        source = address_values[batch % len(address_values)]
        dest = address_values[(batch + 1) % len(address_values)]
        length_values = (0, 1, 0xFFFE, 0xFFFF)
        boundary_length = length_values[batch % len(length_values)]
        mode = batch % 32
        # First sweep every mode across low burst values on distinct resources;
        # later passes cover the high half. Conflict campaigns cover high burst
        # values early, so both axes progress without a full Cartesian replay.
        burst = ((batch // 32) * 4 + instance) % 8
        packed = 0
        for field in self.semantic_ir.packed_fields:
            name = field.name.lower()
            if "len" in name or "length" in name:
                value = boundary_length
            elif "burst" in name:
                value = burst
            elif "dir" in name or "mode" in name:
                value = mode
            else:
                value = batch
            packed = self._set_packed_value(packed, field, value)

        self._put_field_write(instance, source_selector, source)
        self._put_field_write(instance, dest_selector, dest)
        self._put_field_write(instance, packed_selector, packed)

        # Large length boundaries are sampled by the write above, then replaced
        # by a short executable transfer so a channel cannot consume the rest
        # of the evaluation budget.
        if boundary_length == 0 or boundary_length > 16:
            safe_packed = packed
            for field in self.semantic_ir.packed_fields:
                if "len" in field.name.lower() or "length" in field.name.lower():
                    safe_packed = self._set_packed_value(
                        safe_packed, field, (1, 2, 4, 8)[batch % 4])
            self._put_field_write(instance, packed_selector, safe_packed)

        start = self._base_action()
        for item in self.instance_select_indices:
            start[item] = instance
        for item in self.valid_indices:
            start[item] = 1
        self.put(self._sanitize(start))
        self.put(self._sanitize(self._base_action()))
        # Escalating observation windows cover unknown completion/hold latency
        # while preserving early AUC for short-latency implementations.
        wait = (32, 64, 128, 256, 520)[min(4, batch // 4)]
        self.put(self._sanitize(self._base_action()), wait)
        self._macro_scheduler.attach_context(
            field_selected_transaction=True,
            instance=instance,
            selectors=[source_selector, dest_selector, packed_selector],
            source=source, destination=dest, packed=packed,
            mode=mode, burst=burst, boundary_length=boundary_length,
            wait_cycles=wait)
        return True

    def _field_selected_conflict_program(self, cursor):
        """Configure several selected resources, then request them together."""
        if not self.field_selected_interface:
            return False
        instance_field = self.semantic_ir.field(self.instance_select_indices[0])
        selector_field = self.semantic_ir.field(
            self.register_address_indices[0])
        instance_count = min(16, int(instance_field.maximum) + 1)
        if instance_count < 2:
            return False
        selector_max = min(15, int(selector_field.maximum))
        enum_values = sorted(selector_field.enums)
        selector_values = sorted(set(enum_values + list(range(selector_max + 1))))
        if len(enum_values) >= 2:
            selector_values = list(range(min(
                selector_max, max(enum_values) + 1) + 1))
        if len(selector_values) < 3:
            return False
        labels = {value: str(selector_field.enums.get(value, "")).lower()
                  for value in selector_values}
        source_selector = next((value for value, label in labels.items()
                                if "saddr" in label or "source" in label),
                               selector_values[0])
        dest_selector = next((value for value, label in labels.items()
                              if "daddr" in label or "dest" in label),
                             selector_values[1])
        packed_selector = next((value for value in selector_values
                                if value not in (source_selector, dest_selector)),
                               selector_values[2])

        campaign = max(0, int(cursor) // 8)
        length = 2 if campaign % 2 else 5
        mode = campaign % 32
        burst = 4 + (campaign % 4)
        packed = 0
        for field in self.semantic_ir.packed_fields:
            name = field.name.lower()
            if "len" in name or "length" in name:
                value = length
            elif "burst" in name:
                value = burst
            elif "dir" in name or "mode" in name:
                value = mode
            else:
                value = campaign
            packed = self._set_packed_value(packed, field, value)

        for instance in range(instance_count):
            self._put_field_write(instance, source_selector, instance)
            self._put_field_write(instance, dest_selector,
                                  instance_count - 1 - instance)
            self._put_field_write(instance, packed_selector, packed)
        # Changing the selected instance while request remains asserted creates
        # a new per-instance rising edge on selector-addressed interfaces.
        for instance in range(instance_count):
            start = self._base_action()
            for item in self.instance_select_indices:
                start[item] = instance
            for item in self.valid_indices:
                start[item] = 1
            self.put(self._sanitize(start))
        self.put(self._sanitize(self._base_action()))
        wait = (64, 128, 256, 520)[min(3, campaign // 2)]
        self.put(self._sanitize(self._base_action()), wait)
        self._macro_scheduler.attach_context(
            field_selected_transaction=True,
            field_selected_conflict=True,
            instances=instance_count, selectors=[source_selector, dest_selector,
                                                 packed_selector],
            packed=packed, mode=mode, burst=burst, length=length,
            wait_cycles=wait)
        return True

    def _next_field_selected_transaction(self):
        cursor = self._field_transaction_cursor
        conflict_campaign = cursor > 0 and cursor % 8 == 7
        built = (self._field_selected_conflict_program(cursor)
                 if conflict_campaign else
                 self._field_selected_transaction_program(cursor))
        if not built:
            return False
        self._field_transaction_cursor = cursor + 1
        return True

    def _registers_matching(self, *tokens, writable=True):
        matches = []
        for register in self.semantic_ir.registers:
            text = f"{register.name} {register.description}".lower()
            if writable and re.search(r"(?:^|\|)\s*ro\s*(?:\||$)", text):
                continue
            if any(token in text for token in tokens):
                matches.append(register)
        return matches

    def _register_transaction_program(self, cursor):
        """Build one complete configuration/start/data/wait transaction.

        Register names and access descriptions come from the current spec. The
        program is shared across DUTs and never switches on a DUT name.
        """
        if not (self.register_address_indices and self.write_enable_indices and
                self.register_data_indices and self.semantic_ir.registers):
            return False
        enables = self._registers_matching("enable", "enr", "start")
        # Prefer explicit data-port register names. FIFO status/threshold
        # registers are not payload ports even though their descriptions
        # contain the word FIFO.
        data_ports = [item for item in self.semantic_ir.registers
                      if item.name.lower() in
                      ("dr", "data", "txdata", "tx_data", "fifo_data")]
        if not data_ports:
            data_ports = self._registers_matching(
                "data port", "data fifo", "fifo push", "tx data")
        controls = self._registers_matching(
            "ctrl", "control", "mode", "select", "slave", "ser",
            "baud", "divider", "threshold", "ftlr", "frame")
        if not (enables and data_ports):
            return False

        # Many register protocols require configuration while disabled.
        for register in enables[:1]:
            self._put_register_write(register.address, 0)

        for offset, register in enumerate(controls[:6]):
            text = f"{register.name} {register.description}".lower()
            phase = cursor + offset
            if any(token in text for token in
                   ("frame", "count", "ndf", "帧数")):
                value = (0, 1, 3, 15)[phase % 4]
            elif any(token in text for token in ("baud", "divider")):
                value = (2, 8, 16)[phase % 3]
            elif any(token in text for token in ("select", "slave", " ser")):
                value = (1, 2, 4, 8)[phase % 4]
            elif any(token in text for token in ("threshold", "ftlr")):
                value = (0, 1, 7)[phase % 3]
            elif any(token in text for token in ("ctrl", "control", "mode")):
                # Legal-looking frame/control encodings spanning low fields,
                # mode fields and loopback without requiring DUT-specific code.
                value = (7, 0x807, 0x2807, 0x0C07, 0x1C07)[phase % 5]
            else:
                value = self._boundary_value(phase)
            self._put_register_write(register.address, value)

        for register in enables[:1]:
            self._put_register_write(register.address, 1)

        port = data_ports[cursor % len(data_ports)]
        tokens = (0, 0x55, 0xAA, 0xFF)
        for value in tokens[:1 + (cursor % len(tokens))]:
            self._put_register_write(port.address, value)
        # Let the configured transaction reach terminal states before another
        # macro can reset or overwrite its prerequisites.
        self.put(self._sanitize(self._base_action()),
                 (16, 32, 64, 128)[cursor % 4])
        return True

    def _joint_register_program(self, candidate, cursor):
        if not (self.register_data_indices and self.write_enable_indices and
                self.register_address_indices):
            return False
        enables = self._registers_matching("enable", "enr", "start")
        data_ports = [item for item in self.semantic_ir.registers
                      if item.name.lower() in
                      ("dr", "data", "txdata", "tx_data", "fifo_data")]
        if not (enables and data_ports):
            return False
        for register in enables[:1]:
            self._put_register_write(register.address, 0)

        candidate_addresses = {address for address, _ in
                               candidate.register_writes}
        # Add legal support configuration without overwriting target fields.
        support = self._registers_matching(
            "select", "slave", "ser", "baud", "divider",
            "threshold", "ftlr")
        for register in support:
            if register.address in candidate_addresses:
                continue
            text = f"{register.name} {register.description}".lower()
            if any(token in text for token in ("baud", "divider")):
                value = (2, 8, 16)[cursor % 3]
            elif any(token in text for token in ("threshold", "ftlr")):
                value = (0, 1, 7)[cursor % 3]
            else:
                value = (1, 2, 4, 8)[cursor % 4]
            self._put_register_write(register.address, value)
        for address, value in candidate.register_writes:
            self._put_register_write(address, value)
        for register in enables[:1]:
            self._put_register_write(register.address, 1)
        port = data_ports[cursor % len(data_ports)]
        for value in (0x55, 0xAA, 0xFF)[:1 + cursor % 3]:
            self._put_register_write(port.address, value)
        self.put(self._sanitize(self._base_action()), candidate.hold_cycles)
        self._macro_scheduler.attach_context(
            target_index=int(candidate.target_index),
            target_name=candidate.target_name,
            joint_candidate=True,
            complete=bool(candidate.complete),
            mapped_conditions=list(candidate.mapped_conditions),
            unresolved_conditions=list(candidate.unresolved_conditions),
            register_writes=[list(item) for item in candidate.register_writes])
        return True

    def _configure_macro(self):
        cursor = self._macro_cursor["configure"]
        if (self.field_selected_interface and
                self._next_field_selected_transaction()):
            pass
        elif self._register_transaction_program(cursor):
            pass
        elif self.register_address_indices and self.write_enable_indices:
            addresses = self._generic_addresses()
            address = addresses[cursor % len(addresses)]
            value = self._boundary_value(cursor // len(addresses))
            self._put_register_write(address, value)
            if self.read_enable_indices:
                read = self._base_action()
                for item in self.read_enable_indices:
                    read[item] = 1
                for item in self.register_address_indices:
                    read[item] = address
                self.put(self._sanitize(read))
                self.put(self._sanitize(self._base_action()))
        else:
            value = self._boundary_value(cursor)
            self._request(cursor & 3, value, value, wait=4)
        self._macro_cursor["configure"] += 1

    def _control_macro(self):
        cursor = self._macro_cursor["control"]
        if (self.field_selected_interface and
                self._next_field_selected_transaction()):
            self._macro_cursor["control"] += 1
            return
        controls = (self.advance_indices + self.event_indices +
                    self.fault_indices + self.enable_indices +
                    self.read_enable_indices + self.flush_indices)
        if not controls:
            self._generic_transaction()
        else:
            action = self._base_action()
            selected = controls[cursor % len(controls)]
            action[selected] = 1
            if selected in self.read_enable_indices and self.register_address_indices:
                addresses = self._generic_addresses()
                for item in self.register_address_indices:
                    action[item] = addresses[(cursor // len(controls)) % len(addresses)]
            self._write_data(action, self._boundary_value(cursor))
            self._apply_active_target(action)
            self.put(self._sanitize(action), (1, 2, 4)[cursor % 3])
            self.put(self._sanitize(self._base_action()), 2)
        self._macro_cursor["control"] += 1

    def _temporal_macro(self):
        cursor = self._macro_cursor["temporal"]
        if (self.field_selected_interface and
                self._next_field_selected_transaction()):
            self._macro_cursor["temporal"] += 1
            return
        durations = (1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 64)
        duration = durations[cursor % len(durations)]
        if self.register_address_indices and self.write_enable_indices:
            addresses = self._generic_addresses()
            self._put_register_write(addresses[cursor % len(addresses)],
                                     self._boundary_value(cursor + 1))
        self.put(self._sanitize(self._base_action()), duration)
        controls = self.event_indices + self.advance_indices + self.enable_indices
        if controls:
            pulse = self._base_action()
            pulse[controls[cursor % len(controls)]] = 1
            self._write_data(pulse, self._boundary_value(cursor + duration))
            self._apply_active_target(pulse)
            self.put(self._sanitize(pulse))
        if self.event_indices:
            # Ordered token primitive: preserve order and deassert between
            # tokens so edge-sensitive consumers see distinct events.
            tokens = (0x55, 0xAA, self._boundary_value(cursor))
            for token in tokens:
                event = self._base_action()
                event[self.event_indices[cursor % len(self.event_indices)]] = 1
                self._write_data(event, token)
                self.put(self._sanitize(event))
                self.put(self._sanitize(self._base_action()))
        if self.advance_indices:
            run = self._base_action()
            for item in self.advance_indices:
                run[item] = 1
            self.put(self._sanitize(run), duration)
        self.put(self._sanitize(self._base_action()), 2)
        self._macro_cursor["temporal"] += 1

    def _recovery_macro(self):
        cursor = self._macro_cursor["recovery"]
        # First accumulate exceptional state, then apply the declared recovery
        # mechanism and finally begin a clean episode.
        exceptional = self.fault_indices + self.event_indices
        has_recovery = bool(exceptional or self.flush_indices or
                            self.reset_indices)
        if not has_recovery:
            self._generic_transaction()
            self._macro_cursor["recovery"] += 1
            return
        if self.fault_indices:
            # Escalate both repetition count and hold duration across episodes.
            for level in range(1, 2 + (cursor % 3)):
                action = self._base_action()
                action[self.fault_indices[cursor % len(self.fault_indices)]] = 1
                self.put(self._sanitize(action), level)
                self.put(self._sanitize(self._base_action()))
        elif exceptional:
            action = self._base_action()
            action[exceptional[cursor % len(exceptional)]] = 1
            self.put(self._sanitize(action), (1, 2, 4, 8)[cursor % 4])
        if self.flush_indices:
            recover = self._base_action()
            for item in self.flush_indices:
                recover[item] = 1
            self.put(self._sanitize(recover), 2)
        if self.reset_indices:
            reset = self._base_action()
            for item in self.reset_low_indices:
                reset[item] = 0
            for item in self.reset_high_indices:
                reset[item] = 1
            self.put(self._sanitize(reset), 2)
        self.put(self._sanitize(self._base_action()), 3)
        self._macro_cursor["recovery"] += 1

    def _scheduled_generic_transaction(self, covered, covered_bins, step,
                                       max_steps, missing_targets,
                                       target_weights):
        macro = self._macro_scheduler.select(
            covered, step, max_steps, covered_bins=covered_bins,
            target_weights=target_weights,
            model_scores=getattr(self, "_universal_model_scores", None))
        self.macro_trace.append(macro)
        candidates = [target for target in missing_targets
                      if macro in target.macro_hints]
        # Prefer targets with a spec/metadata-supported path to controllable
        # fields. Keep unmapped targets in the rotation so an incomplete graph
        # cannot make a reachable bin permanently invisible.
        candidates.sort(
            key=lambda target: (
                0 if target.index in self._joint_candidate_by_target else 1,
                -self.coverage_dependency_graph.target_confidence(target.index),
                target.index))
        cursor = self._macro_cursor.get(macro, 0)
        if (self.generic_sequence_search_enabled and
                self._next_generic_sequence(step, max_steps)):
            self._macro_cursor[macro] = cursor + 1
            self._macro_scheduler.attach_plan([
                {"action": action.astype(float).tolist(), "cycles": int(cycles)}
                for action, cycles in self.queue
            ])
            self._active_target = None
            return
        selected_joint = None
        if self._joint_ranker is not None:
            self._joint_ranker.observe(self._macro_scheduler.history)
            joint_pool = [self._joint_candidate_by_target[target.index]
                          for target in candidates
                          if target.index in self._joint_candidate_by_target]
            selected_joint = self._joint_ranker.select(joint_pool)
        if selected_joint is not None:
            self._active_target = next(
                target for target in candidates
                if target.index == selected_joint.target_index)
        else:
            self._active_target = (candidates[cursor % len(candidates)]
                                   if candidates else None)
        builders = {
            "configure": self._configure_macro,
            "control": self._control_macro,
            "temporal": self._temporal_macro,
            "recovery": self._recovery_macro,
        }
        if self._joint_ranker is not None:
            # None means every still-missing joint candidate exhausted its
            # bounded retry budget. Run the normal M3 builder so failed cross
            # attempts cannot monopolize the remaining cycle budget.
            joint = selected_joint
        else:
            joint = (self._joint_candidate_by_target.get(
                self._active_target.index)
                if self._active_target is not None else None)
        if joint is None or not self._joint_register_program(joint, cursor):
            builders[macro]()
        else:
            self._macro_cursor[macro] = cursor + 1
        self._macro_scheduler.attach_plan([
            {"action": action.astype(float).tolist(), "cycles": int(cycles)}
            for action, cycles in self.queue
        ])
        self._active_target = None

    def predict(self, coverage_state, step: int, max_steps: int,
                macro_scores=None) -> np.ndarray:
        state = np.asarray(coverage_state, dtype=np.float32).reshape(-1)
        covered = int(np.sum(state))
        covered_bins = set(np.flatnonzero(state > 0.5).astype(int).tolist())
        if covered > self._last_covered:
            self._last_gain_step = int(step)
            self._last_coverage_gain_step = int(step)
            self._last_covered = covered
        if not self.queue:
            missing_targets, target_weights = missing_target_weights(
                state, self.coverage_targets)
            option_model = getattr(self, "_option_model", None)
            if option_model is not None:
                features = encode_runtime_features(
                    self.semantic_ir, self.coverage_targets, covered,
                    state.size, step, max_steps, target_weights)
                self._universal_model_scores = option_model.predict(features)
            episode_reason = self._episode_manager.observe(
                covered_bins, step, max_steps)
            if episode_reason:
                self._explore_level += 1
                self._macro_scheduler.force_next("recovery")
            # Change parameter families after sustained stagnation. This keeps
            # the unknown-DUT path coverage-directed instead of a fixed replay.
            patience = max(128, min(2048, int(max_steps) // 40))
            if int(step) - self._last_gain_step >= patience:
                self._explore_level += 1
                self._last_gain_step = int(step)
            self._scheduled_generic_transaction(
                covered, covered_bins, step, max_steps, missing_targets,
                target_weights)
        action = self.take()
        return action


class UniversalPolicy(_GenericPolicy):
    """Single runtime policy for every DUT schema.

    Legacy expert implementations remain in this module only as offline
    teachers. They are never selected by the submission inference path.
    """

    def __init__(self, *args, option_model_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._option_model = UniversalOptionModel(option_model_path)
        self._universal_model_scores = np.zeros(4, dtype=np.float32)


class _LocalInferenceInterface:
    """Standard committee inference interface."""

    def __init__(self, dut_spec_path: str, covergroup_path: str,
                 macro_order=None, controller_path=None):
        spec = _read(dut_spec_path)
        cover = _read(covergroup_path)
        self._coverage_controller = CoverageSetController(
            controller_path or default_q_controller_path(), covergroup_path)
        self._macro_scores_cache = np.zeros(4, dtype=np.float32)
        self.semantic_ir = build_semantic_ir(spec)
        self.coverage_targets = load_coverage_targets(covergroup_path)
        action_fields = [item.name for item in self.semantic_ir.fields]
        action_dims = self.semantic_ir.action_dim
        self.action_dims = action_dims
        self._policy = UniversalPolicy(
            action_dims, fields=action_fields, spec=spec,
            semantic_ir=self.semantic_ir,
            coverage_targets=self.coverage_targets)

    def predict(self, coverage_state: np.ndarray, step: int,
                max_steps: int) -> np.ndarray:
        # Macro decisions occur hundreds of cycles apart. Refreshing the set
        # encoder every 32 cycles preserves responsive feedback while avoiding
        # unnecessary matrix work on every low-level bus cycle.
        if int(step) == 0 or int(step) % 32 == 0:
            self._macro_scores_cache = self._coverage_controller.predict(
                coverage_state, int(step), int(max_steps))
        macro_scores = self._macro_scores_cache
        self.neural_macro_scores = macro_scores
        action = self._policy.predict(coverage_state, int(step), int(max_steps),
                                      macro_scores=macro_scores)
        return np.asarray(action, dtype=np.float32).reshape(-1)


class InferenceInterface(_LocalInferenceInterface):
    """Committee interface with one-shot DeepSeek semantic planning."""

    def __init__(self, dut_spec_path: str, covergroup_path: str,
                 macro_order=None, controller_path=None):
        spec = _read(dut_spec_path)
        cover = _read(covergroup_path)

        # No network request is ever made by predict(). Failure here is a
        # normal state and the complete local policy remains available.
        self._llm = DeepSeekPlanner()
        self.llm_plan, self.llm_status = self._llm.plan(spec, cover)
        super().__init__(dut_spec_path, covergroup_path, macro_order,
                         controller_path)

        plan = self.llm_plan
        if not plan:
            return
        # The LLM is advisory: it cannot replace a structurally validated
        # expert or change the action-vector shape. For an unknown DUT, only a
        # schema-consistent generic program may seed local exploration.
        family = plan["family_hint"]
        plan_dims = int(plan.get("action_dim") or 0)
        if (isinstance(self._policy, UniversalPolicy) and
                family == "generic" and plan_dims == self.action_dims):
            self._policy = UniversalPolicy(
                self.action_dims, fields=_parse_action_fields(spec), spec=spec,
                planned_program=plan.get("program"),
                semantic_ir=self.semantic_ir,
                coverage_targets=self.coverage_targets)

    @staticmethod
    def _infer_dims(spec: str) -> int:
        return _infer_action_dims(spec)
