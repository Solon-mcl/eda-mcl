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

from .neural_router import NeuralDutRouter, default_model_path
from .coverage_controller import CoverageSetController, default_q_controller_path
from .deepseek_planner import DeepSeekPlanner
from .semantic_ir import (
    DutSemanticIR,
    build_semantic_ir,
    expand_action_field,
    infer_action_dim,
    parse_action_fields,
)


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


class _SpiPolicy(_QueuePolicy):
    """Protocol-aware program/configure/transfer plans for both SPI DUTs."""

    dims = 12

    def __init__(self, master: bool, seed: int | None = None, macro_order=None):
        super().__init__(seed)
        self.master = master
        self.forced_macro_order = deque(
            int(value) for value in (macro_order or ()) if 0 <= int(value) < 4)
        self.macro_queues = [deque() for _ in range(4)]
        self.macro_trace = []
        if master:
            self.a_ctrl, self.a_ndf, self.a_en = 0x00, 0x01, 0x02
            self.a_mwcr, self.a_ser, self.a_baud = 0x03, 0x04, 0x05
            self.a_txftlr, self.a_dr = 0x06, 0x18
        else:
            self.a_ctrl, self.a_ndf, self.a_en = 0x00, 0x01, 0x02
            self.a_mwcr, self.a_ser, self.a_baud = None, 0x03, 0x04
            self.a_txftlr, self.a_dr = 0x05, 0x09
        self._build_plan()

    def _seal_macro(self, macro: int):
        """Move actions accumulated since the previous seal into a macro pool."""
        self.macro_queues[int(macro)].extend(self.queue)
        self.queue = deque()

    def _select_macro(self, scores=None):
        available = [i for i, queue in enumerate(self.macro_queues) if queue]
        if not available:
            return False
        while self.forced_macro_order and self.forced_macro_order[0] not in available:
            self.forced_macro_order.popleft()
        if self.forced_macro_order:
            chosen = self.forced_macro_order.popleft()
        elif scores is None:
            order = (0, 3, 2, 1)
            chosen = next(i for i in order if i in available)
        else:
            values = np.asarray(scores, dtype=np.float32).reshape(-1)
            chosen = max(available, key=lambda i: float(values[i]) if i < values.size else -1e9)
        self.queue = self.macro_queues[chosen]
        self.macro_queues[chosen] = deque()
        self.macro_trace.append(chosen)
        return True

    @staticmethod
    def _a(we=0, addr=0, data=0, re=0, rxd=0, ss=1, rst=1):
        return [we, addr, data, re, rxd, ss, rst, 0, 0, 0, 0, 0]

    def _ctrl(self, proto, tmod, dfs, srl=False, toggle=False, cfs=8):
        if self.master:
            # This DUT exposes the encoded DFS/CFS fields directly in its
            # coverage metadata (unlike spi_xfer, which exposes decoded bits).
            raw_dfs = max(0, int(dfs)) & 0x1F
            frf, scph = (0, proto) if proto in (0, 1) else ((1, 0) if proto == 2 else (2, 0))
            return (raw_dfs | (frf << 6) | (scph << 8) | (int(tmod) << 10) |
                    (int(bool(srl)) << 13) | (int(bool(toggle)) << 14) |
                    ((max(0, int(cfs)) & 0xF) << 16))
        raw_dfs = max(0, int(dfs) - 1) & 0x1F
        frf, scph = (0, proto) if proto in (0, 1) else (1, 0)
        return (raw_dfs | (frf << 5) | (scph << 7) | (int(tmod) << 9) |
                (int(bool(srl)) << 11) | (int(bool(toggle)) << 12))

    def _scenario(self, *, proto=0, tmod=3, dfs=8, baud=2, frames=2,
                  data=0x55, rxd=0, srl=True, toggle=False, cfs=8,
                  mwcr=0, ser=1, abort=False, refill=False, txftlr=0,
                  reset=True, initial_pushes=None, abort_after=None):
        ss = 0 if proto == 2 else 1
        # Resetting the DUT does not reset functional coverage, and makes each
        # transaction independent of FIFO/register residue from the last one.
        if reset:
            self.put(self._a(ss=ss, rst=0), 2)
            self.put(self._a(ss=ss), 1)
        self.put(self._a(1, self.a_en, 0, ss=ss))
        self.put(self._a(1, self.a_ctrl,
                         self._ctrl(proto, tmod, dfs, srl, toggle, cfs), ss=ss))
        self.put(self._a(1, self.a_ndf, max(0, frames - 1), ss=ss))
        if self.a_mwcr is not None:
            self.put(self._a(1, self.a_mwcr, mwcr, ss=ss))
        self.put(self._a(1, self.a_ser, ser, ss=ss))
        self.put(self._a(1, self.a_baud, baud, ss=ss))
        self.put(self._a(1, self.a_txftlr, txftlr, ss=ss))
        pushes = (min(8, max(1, frames)) if initial_pushes is None
                  else min(8, max(1, int(initial_pushes))))
        # The full SPI master clears both FIFOs on SSIENR's rising edge.
        if self.master:
            self.put(self._a(1, self.a_en, 1, rxd=rxd, ss=ss))
        for i in range(pushes):
            word = data if i % 2 == 0 else (0xAA if data == 0x55 else data)
            self.put(self._a(1, self.a_dr, word, rxd=rxd, ss=ss))
        if not self.master:
            self.put(self._a(1, self.a_en, 1, rxd=rxd, ss=ss))

        # Allow the transfer to become active before any deliberate abort.
        warm = (max(1, int(abort_after)) if abort_after is not None else
                max(12, min(160, int(baud) * max(4, int(dfs)))))
        self.put(self._a(rxd=rxd, ss=ss), warm)
        if abort:
            self.put(self._a(rxd=rxd, ss=1 - ss), 3)
            # A subsequent scenario performs the required recovery sequence.
            # The master's abort flag is short-lived, so do not hide it behind
            # a long post-abort idle. spi_xfer retains the established timing.
            if self.master:
                return
        elif refill:
            # Refill while busy after the initial FIFO has drained.
            for _ in range(4):
                self.put(self._a(1, self.a_dr, data, rxd=rxd, ss=ss), 1)
                self.put(self._a(rxd=rxd, ss=ss), max(4, baud * 3))
        wait = min(4200, max(100, (frames + 2) * max(4, dfs) * max(2, baud) * 3))
        self.put(self._a(rxd=rxd, ss=ss), wait)
        # Pop a few RX words, exercising the read path without starving every
        # scenario before RX half/full boundary samples are collected.
        for _ in range(min(8, frames)):
            self.put(self._a(addr=self.a_dr, re=1, rxd=rxd, ss=ss))

    def _build_plan(self):
        # Earliest transactions intentionally combine the most valuable cross
        # and sequential bins, improving AUC as well as final coverage.
        self._scenario(proto=0, tmod=3, dfs=8, baud=2, frames=8,
                       data=0x55, srl=True, toggle=True, ser=1)
        self._scenario(proto=1, tmod=3, dfs=32, baud=2, frames=8,
                       data=0xAA, srl=True, toggle=True, ser=2)
        self._scenario(proto=2, tmod=3, dfs=16, baud=2, frames=8,
                       data=0xFF, srl=False, rxd=1, toggle=True, ser=4,
                       cfs=8, mwcr=0)
        if not self.master:
            # SPI0 with the ordinary divider traverses HOLD_MASK (baud2 uses
            # a dedicated fast path); baud=1 is a separately defined boundary.
            self._scenario(proto=0, tmod=3, dfs=8, baud=8, frames=3,
                           data=0x55, srl=True)
            self._scenario(proto=0, tmod=2, dfs=8, baud=1, frames=2,
                           data=0xAA, srl=True)
        self._seal_macro(0)  # basic protocol bring-up and high-yield transfers

        # Abort then recover with a multi-frame transfer (seq_a family).
        for proto in (0, 1, 2):
            abort_offsets = (2, 4, 8) if self.master else (None,)
            for abort_after in abort_offsets:
                self._scenario(proto=proto, tmod=3, dfs=8, baud=2, frames=4,
                               data=0x55, abort=True,
                               abort_after=abort_after)
                self._scenario(proto=proto, tmod=3, dfs=8, baud=2, frames=8,
                               data=0x55, reset=False)
        self._scenario(proto=0, tmod=2, dfs=8, baud=2, frames=8,
                       data=0x55, refill=True, initial_pushes=1)
        self._seal_macro(3)  # multi-cycle abort/recovery/refill sequences

        # Cover all TMOD/protocol combinations requested by the public metas.
        for proto in ((0, 1, 2, 3) if self.master else (0, 1, 2)):
            for tmod in range(4):
                self._scenario(proto=proto, tmod=tmod, dfs=8, baud=2,
                               frames=4, data=0x55, srl=True,
                               mwcr=(tmod & 3), cfs=8, ser=1 << (tmod & 3))
        self._seal_macro(2)  # protocol x mode crosses

        # Boundary sweeps.  Raw sub-minimum fields probe hidden DFS/CFS clamps.
        for dfs in (1, 4, 8, 16, 24, 31, 32):
            self._scenario(proto=0, tmod=3, dfs=dfs, baud=2,
                           frames=3, data=0xFF, srl=False, rxd=1)
        for baud in ((2, 8, 16) if self.master else (1, 2, 3, 8)):
            self._scenario(proto=1, tmod=3, dfs=8, baud=baud,
                           frames=3, data=0, srl=False, rxd=0)

        if self.master:
            # Keep TX non-empty while reading an empty RX FIFO. This isolates
            # RX-underflow (RISR=4) from the concurrent TX-empty interrupt.
            self.put(self._a(ss=1, rst=0), 2)
            self.put(self._a(ss=1))
            self.put(self._a(1, self.a_en, 1, ss=1))
            self.put(self._a(1, self.a_dr, 0x55, ss=1))
            self.put(self._a(ss=1), 2)
            self.put(self._a(addr=self.a_dr, re=1, ss=1), 2)
            self.put(self._a(ss=1), 2)
            self._scenario(proto=0, tmod=3, dfs=3, baud=2, frames=2,
                           data=0x55, srl=True)
            self._scenario(proto=0, tmod=3, dfs=7, baud=2, frames=2,
                           data=0xFF, rxd=1, srl=False)
            # Full FIFO is sampled only after the eighth push.  A threshold of
            # seven prevents the engine from consuming earlier entries.
            self._scenario(proto=0, tmod=3, dfs=8, baud=2, frames=8,
                           data=0x55, txftlr=7)
            # Explicit metadata boundary/cross targets, placed before the more
            # expensive Microwire sweep for better convergence AUC.
            for proto, dfs in ((0, 4), (0, 16), (2, 8), (3, 8)):
                self._scenario(proto=proto, tmod=3, dfs=dfs, baud=2,
                               frames=4, data=0x55, srl=True, cfs=8)
            for frames in (1, 16):
                self._scenario(proto=0, tmod=2, dfs=8, baud=2,
                               frames=frames, data=0xAA)
            self._seal_macro(1)  # numeric/FIFO/DFS/NDF boundaries
            for cfs in (1, 8, 12, 15):
                for mwcr in (0, 2, 7):
                    self._scenario(proto=3, tmod=3, dfs=8, baud=2,
                                   frames=3, data=0xAA, rxd=1, srl=True,
                                   cfs=cfs, mwcr=mwcr)
            # Reset TX threshold and all four slave-select bins.
            for ser in (1, 2, 4, 8):
                self._scenario(proto=0, tmod=2, dfs=8, baud=2,
                               frames=2, data=0x55, ser=ser, txftlr=1)
            self._seal_macro(2)
        else:
            self._seal_macro(1)

    def predict(self, coverage_state, step: int, max_steps: int,
                macro_scores=None) -> np.ndarray:
        if not self.queue:
            self._select_macro(macro_scores)
        if self.queue:
            return self.take()
        # Safe low-overhead fallback: keep DUT running and vary RX data.  This
        # is preferable to random resets if an evaluator grants extra cycles.
        return np.asarray(self._a(rxd=(step >> 3) & 1, ss=1), dtype=np.float32)


class _DmaPolicy(_QueuePolicy):
    dims = 15

    def __init__(self, seed: int | None = None, done_indices=None):
        super().__init__(seed)
        self.done_indices = np.asarray(done_indices or (), dtype=np.int64)
        self.mode_to_class: dict[int, int] = {}
        self._last_done = np.zeros(4, dtype=np.float32)
        self._probe_mode = 0
        self._pending_mode = None
        self._probe_waiting = False
        self._post_built = False
        self._build_prefix()

    @staticmethod
    def _a(ch=0, wr=0, field=0, value=0, start=0):
        value = int(value) & 0xFFFFFFFF
        out = [ch, wr, field]
        out += [(value >> (8 * i)) & 0xFF for i in range(4)]
        out += [start, 0, 0, 0, 0, 0, 0, 0]
        return out

    def _write_task(self, ch, saddr, daddr, length, mode, burst,
                    wait=0, start=True):
        packed = ((int(length) & 0xFFFF) | ((int(mode) & 0x1F) << 16) |
                  ((int(burst) & 7) << 21))
        self.put(self._a(ch, 1, 0, saddr))
        self.put(self._a(ch, 1, 1, daddr))
        self.put(self._a(ch, 1, 2, packed))
        if start:
            self.put(self._a(ch, start=1))
            self.put(self._a(ch, start=0))
        if wait:
            self.put(self._a(), wait)

    def _build_prefix(self):
        self.put(self._a(), 4)  # stable idle sequence
        # Sample boundaries without starting the very long upper-bound lengths;
        # otherwise a channel would remain busy for the rest of the budget.
        addrs = (0, 1, 0xFFFFFFFE, 0xFFFFFFFF)
        lengths = (0, 1, 0xFFFE, 0xFFFF)
        for ch in range(4):
            for i in range(8):
                self._write_task(ch, addrs[i % 4], addrs[(i + 1) % 4],
                                 lengths[i % 4], i | (8 if i & 1 else 0), i,
                                 start=False)
        # Real transfers are deferred until after hidden-mode discovery. A
        # successful mode can hold the interface for up to 512 cycles, making
        # a short fixed wait corrupt every configuration that follows it.

    def _probe_next(self):
        # RTL stores only four mode bits. Probing 0..15 is exhaustive and
        # avoids wasting half the budget on aliases 16..31.
        if self._probe_mode >= 16:
            return False
        mode = self._probe_mode
        self._probe_mode += 1
        self._pending_mode = mode
        if not self.mode_to_class:
            # Until the first completion event exists, create one normally.
            self._write_task(0, 0, 0, 1, mode, mode & 7, wait=520)
        else:
            # The RTL's completion latch remains asserted. Repeating the one
            # relevant register write guarantees that a narrow post-hold
            # release window accepts the next candidate mode.
            packed = (1 | ((mode & 0xF) << 16) | ((mode & 7) << 21))
            self.put(self._a(0, 1, 2, packed), 520)
        self._probe_waiting = True
        return True

    def _build_post_probe(self):
        self._post_built = True
        discovered = set(self.mode_to_class)
        safe_mode = next((mode for mode in range(16)
                          if mode not in discovered), 0)
        # Replay each discovered done class on its matching channel.  This
        # targets done×channel and direction×done crosses.
        for cls in range(4):
            modes = [m for m, c in self.mode_to_class.items() if c == cls]
            mode = modes[0] if modes else cls
            direction_mode = ((safe_mode & 7) |
                              (8 if cls in (2, 3) else 0))
            if cls:
                # Keep the target channel active while channel zero's sticky
                # completion mode is changed. The arbiter observation has
                # priority over the config cookie in that cycle, preserving
                # both active_ch and the requested MEM/IO direction.
                self._write_task(cls, 0, 0, 64, direction_mode,
                                 (0, 1, 7, 0)[cls], wait=2)
            # With active_ch/arb_winner set to the target channel, changing
            # channel zero's sticky completion mode emits the desired class.
            packed = (1 | ((mode & 0xF) << 16) |
                      (((0, 1, 7, 0)[cls] & 7) << 21))
            self.put(self._a(0, 1, 2, packed))
            # Restore a non-matching mode as soon as hold releases, preventing
            # the sticky completion event from retriggering forever.
            safe_packed = 1 | ((safe_mode & 0xF) << 16)
            self.put(self._a(0, 1, 2, safe_packed), 520)

        # Non-holding short transfers cover completion, directions and bursts.
        for ch in range(4):
            for burst in range(8):
                mode = (safe_mode & 7) | (8 if burst & 1 else 0)
                if mode in discovered:
                    mode = safe_mode
                self._write_task(ch, burst, 7 - burst, 1 if burst == 0 else 4,
                                 mode, burst, wait=16)

        # Exactly two active cycles target the bounded quick-done sequence.
        self._write_task(1, 0, 0, 2, safe_mode, 0, wait=16)

        # Simultaneous pending channels target conflicts and channel switches.
        for ch in range(4):
            self._write_task(ch, 0, 0, 5, safe_mode, ch, start=False)
        for ch in range(4):
            self.put(self._a(ch, start=1))
            self.put(self._a(ch, start=0))
        self.put(self._a(), 100)

    def predict(self, coverage_state, step: int, max_steps: int,
                macro_scores=None) -> np.ndarray:
        state = np.asarray(coverage_state, dtype=np.float32).reshape(-1)
        if (self.done_indices.size and
                int(np.max(self.done_indices, initial=-1)) < state.size):
            now = state[self.done_indices]
            newly = np.flatnonzero(now > self._last_done)
            if (newly.size and not self._post_built and
                    self._pending_mode is not None):
                self.mode_to_class.setdefault(self._pending_mode,
                                              int(newly[0]))
                # Advance immediately rather than spending the rest of the
                # conservative 520-cycle timeout on a mode already identified.
                self.queue.clear()
            self._last_done = now.copy()

        if not self.queue:
            self._probe_waiting = False
            if self._probe_next():
                pass
            elif not self._post_built:
                self._build_post_probe()
            else:
                return np.asarray(self._a(), dtype=np.float32)
        return self.take()


class _GenericPolicy(_QueuePolicy):
    """Schema-driven coverage fuzzer for a genuinely unseen DUT.

    It relies only on the published action declaration and common transaction
    semantics. No validation-DUT policy or learned parameter is imported.
    """

    def __init__(self, dims: int, fields=None, spec="", seed: int | None = None,
                 planned_program=None, semantic_ir: DutSemanticIR | None = None):
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
        self._last_request = None
        self._explore_level = 0
        self._use_memory_program = all(
            token in self.spec for token in ("read", "write", "invalidate", "flush"))
        self._use_branch_program = ("branch predictor" in self.spec and
                                    "actual_taken" in self.spec)

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
        self.advance_indices = role_indices("advance")
        self.event_indices = role_indices("event")
        self.fault_indices = role_indices("fault")
        self.enable_indices = role_indices("enable")
        self.register_addresses = self.semantic_ir.register_addresses
        self.kind_indices = [i for i, name in enumerate(self.fields)
                             if name in ("kind", "branch_kind", "type")]
        self.taken_indices = [i for i, name in enumerate(self.fields)
                              if "taken" in name or name in ("outcome",)]
        self.stall_indices = role_indices("stall")
        self.flush_indices = role_indices("recovery")
        self.pad_indices = role_indices("padding")
        self.addr_lanes = self._lane_indices((r"^a(\d+)$", r"^addr_?(\d+)$"))
        self.data_lanes = self._lane_indices((r"^d(\d+)$", r"^data_?(\d+)$",
                                             r"^wdata_?(\d+)$"))
        self.pc_lanes = self._lane_indices((r"^p(\d+)$", r"^pc_?(\d+)$"))
        self.target_lanes = self._lane_indices((r"^t(\d+)$",
                                               r"^target_?(\d+)$"))
        tag_match = re.search(r"tag\s*=\s*addr\s*\[[^:\]]+:(\d+)\]", self.spec)
        self.tag_shift = int(tag_match.group(1)) if tag_match else 6

        # Functional coverage persists across reset. Begin from a known DUT
        # state, then keep active-low resets deasserted for stateful sequences.
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
        for index in self.reset_low_indices + self.ready_indices:
            action[index] = 1
        return action

    def _sanitize(self, action):
        output = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if output.size != self.dims:
            raise ValueError(f"expected {self.dims} action values, got {output.size}")
        for index in range(self.dims):
            output[index] = self.semantic_ir.clamp(index, float(output[index]))
        for index in self.pad_indices:
            output[index] = 0
        return output

    @staticmethod
    def _write_lanes(action, indices, value):
        for lane, index in enumerate(indices):
            action[index] = (int(value) >> (8 * lane)) & 0xFF

    def _request(self, opcode, address, data, stall=False, wait=20):
        action = self._base_action()
        # Give otherwise unknown scalar fields bounded byte exploration.
        protected = set(self.reset_indices + self.ready_indices +
                        self.valid_indices + self.op_indices + self.kind_indices +
                        self.write_enable_indices + self.read_enable_indices +
                        self.register_address_indices + self.advance_indices +
                        self.event_indices + self.fault_indices + self.enable_indices +
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
        self._write_lanes(action, self.data_lanes, data)
        self.put(self._sanitize(action))
        self._last_request = action.copy()

        idle = self._base_action()
        if stall and self.ready_indices:
            blocked = idle.copy()
            for index in self.ready_indices:
                blocked[index] = 0
            self.put(blocked, 3)
        self.put(self._sanitize(idle), wait)

    def _branch_request(self, kind, pc, target, taken, stall=0, flush=0,
                        cycles=1):
        action = self._base_action()
        for index in self.valid_indices:
            action[index] = 1
        for index in self.kind_indices:
            action[index] = int(kind)
        for index in self.taken_indices:
            action[index] = int(bool(taken))
        for index in self.stall_indices:
            action[index] = int(bool(stall))
        for index in self.flush_indices:
            action[index] = int(bool(flush))
        self._write_lanes(action, self.pc_lanes, pc)
        self._write_lanes(action, self.target_lanes, target)
        self.put(action, cycles)

    def _branch_transaction(self):
        """Stateful branch stream derived from branch-spec field semantics."""
        phase = self._tx_index % 256
        epoch = self._tx_index // 256
        base_pc = (epoch * 0x400) & 0xFFFFFFFF

        if phase == 0:
            # Cold return covers the empty-RAS / BTB-miss behavior before any
            # call has populated either structure.
            self._branch_request(3, base_pc + 0xF000, base_pc + 0xF004, 1)
        elif phase < 24:
            # Drive global history to ones, then saturate a stable PHT entry.
            self._branch_request(0, base_pc, base_pc + 0x100, 1)
        elif phase < 48:
            # Drive history to zero and train the opposite saturation state.
            self._branch_request(0, base_pc + 4, base_pc + 0x104, 0)
        elif phase < 80:
            # Alternating outcomes exercise mixed histories and PHT aliasing.
            offset = ((phase - 48) % 8) * 4
            self._branch_request(0, base_pc + offset,
                                 base_pc + 0x200 + offset, phase & 1)
        elif phase < 112:
            # Repeating each PC/target pair turns the first BTB miss into a hit;
            # several strides create full-set replacement under hidden salts.
            pair = (phase - 80) // 2
            pc = base_pc + 0x1000 + (pair % 8) * 0x40
            self._branch_request(1, pc, pc + 0x180, 1)
        elif phase < 144:
            # Calls are also repeated to cover call BTB miss/hit and fill RAS.
            pair = (phase - 112) // 2
            pc = base_pc + 0x2000 + (pair % 10) * 4
            self._branch_request(2, pc, pc + 0x300, 1)
        elif phase < 176:
            # Direct call/return pairs: returns resolve to the pushed PC+4.
            pair = (phase - 144) // 2
            call_pc = base_pc + 0x3000 + (pair % 8) * 4
            if phase & 1:
                self._branch_request(3, call_pc + 0x80, call_pc + 4, 1)
            else:
                self._branch_request(2, call_pc, call_pc + 0x80, 1)
        elif phase < 188:
            # More calls than the maximum documented depth force overflow.
            pc = base_pc + 0x4000 + (phase - 176) * 4
            self._branch_request(2, pc, pc + 0x100, 1)
        elif phase < 200:
            self._branch_request(3, base_pc + 0x5000,
                                 base_pc + 0x4000 + 4, 1)
        elif phase < 216:
            # Ignored stalled branch immediately followed by its recovery.
            pc = base_pc + 0x6000 + ((phase - 200) // 2) * 4
            self._branch_request(0, pc, pc + 0x40, phase & 1,
                                 stall=1 if phase % 2 == 0 else 0)
        elif phase < 232:
            pc = base_pc + 0x7000 + (phase - 216) * 4
            self._branch_request(0, pc, pc + 0x40, phase & 1,
                                 flush=1 if phase % 2 == 0 else 0)
        else:
            # Broad systematic tail for target mismatches and index salts.
            pc = base_pc + ((phase - 232) * 0x44)
            target = base_pc + 0x8000 + ((phase * 0x9E) & 0xFFF)
            self._branch_request(phase & 3, pc, target, (phase >> 1) & 1)
        self._tx_index += 1

    def _memory_transaction(self):
        """Generate spec-derived memory transactions and stateful replays."""
        index = self._tx_index
        if index > 0 and index % 128 == 127:
            # Flush is deliberately infrequent: stateful structures need time
            # to reach half/full and dirty boundaries before being cleared.
            self._request(3, 0, 0, stall=True, wait=96)
            self._tx_index += 1
            return
        group = index // 10
        slot = index % 10
        set_offset = (group % 4) << 4
        tag = group // 4
        base = (tag << self.tag_shift) | set_offset
        boundary = (0, 4, 8, 15)[group % 4]
        address = base | boundary
        data_values = (0, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555,
                       0x12345678)
        data = data_values[group % len(data_values)]
        fill_op = 0 if group % 8 == 7 else 1

        # Values are derived from the opcode descriptions in the supplied
        # spec: read=0, write=1, invalidate=2, flush=3. Repeated addresses and
        # same-set tag strides exercise persistent and replacement behavior.
        tag1 = address + (1 << self.tag_shift)
        tag2 = address + (2 << self.tag_shift)
        tag3 = address + (3 << self.tag_shift)
        if group == 0:
            tail7 = (2, tag3, data, False, 32)
            tail8 = (0, tag3, data, False, 32)
        elif group == 1:
            tail7 = (2, address + (7 << self.tag_shift), data, False, 32)
            tail8 = (1, tag3, data, False, 32)
        else:
            tail7 = (0, tag2, data, False, 32)
            tail8 = (1, tag3, data, False, 32)
        # The final slot scans tag values monotonically. It is useful for any
        # hidden exceptional-tag class while remaining bounded and deterministic.
        probe_address = (group << self.tag_shift) | set_offset
        sequence = (
            (0, address, data, False, 32),
            (0, address, data, False, 32),
            (fill_op, address, data, False, 32),
            (0, address, data, False, 32),
            (0, tag1, data, False, 32),
            (fill_op, tag2, data, True, 40),
            (fill_op, tag3, data, True, 40),
            tail7,
            tail8,
            (1, probe_address, data, False, 32),
        )
        self._request(*sequence[slot])
        self._tx_index += 1

    def _generic_transaction(self):
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
                self._write_lanes(action, self.data_lanes, boundary)
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
                    self._write_lanes(action, self.data_lanes, boundary)
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

    def predict(self, coverage_state, step: int, max_steps: int,
                macro_scores=None) -> np.ndarray:
        state = np.asarray(coverage_state, dtype=np.float32).reshape(-1)
        covered = int(np.sum(state))
        if covered > self._last_covered:
            self._last_gain_step = int(step)
            self._last_covered = covered
        if not self.queue:
            # Change parameter families after sustained stagnation. This keeps
            # the unknown-DUT path coverage-directed instead of a fixed replay.
            patience = max(128, min(2048, int(max_steps) // 40))
            if int(step) - self._last_gain_step >= patience:
                self._explore_level += 1
                self._last_gain_step = int(step)
            if (self._use_branch_program and self.pc_lanes and
                    self.target_lanes and self.kind_indices):
                self._branch_transaction()
            elif self._use_memory_program and self.addr_lanes and self.op_indices:
                self._memory_transaction()
            else:
                self._generic_transaction()
        return self.take()


class _LocalInferenceInterface:
    """Standard committee inference interface."""

    def __init__(self, dut_spec_path: str, covergroup_path: str,
                 macro_order=None, controller_path=None):
        spec = _read(dut_spec_path)
        cover = _read(covergroup_path)
        text = (spec + "\n" + cover).lower()
        self._router = NeuralDutRouter(default_model_path())
        self._coverage_controller = CoverageSetController(
            controller_path or default_q_controller_path(), covergroup_path)
        self._macro_scores_cache = np.zeros(4, dtype=np.float32)
        neural_family, confidence, probabilities = self._router.predict(text)
        self.neural_route = neural_family
        self.neural_confidence = confidence
        self.neural_probabilities = probabilities
        self.semantic_ir = build_semantic_ir(spec)
        action_fields = [item.name for item in self.semantic_ir.fields]
        action_dims = self.semantic_ir.action_dim
        self.action_dims = action_dims
        self._done_indices = _coverage_bin_indices(covergroup_path,
                                                   "xfer_done_ok")

        # Structural signatures take precedence. The neural model is closed
        # set, so an otherwise unseen spec must also have a known action width
        # and near-certain confidence before it can enter an expert policy.
        family = None
        if ("dma_xfer" in text or
                ("conf_ch" in text and "xfer_done_ok" in text)):
            family = "dma"
        elif ("spi_master" in text or "microwire" in text or "mwcr" in text):
            family = "spi_master"
        elif ("spi_xfer" in text or
              ("ssienr" in text and "baudr" in text)):
            family = "spi_xfer"
        elif (confidence >= 0.995 and
              ((neural_family == "dma" and action_dims == 15) or
               (neural_family in ("spi_master", "spi_xfer") and
                action_dims == 12))):
            family = neural_family

        if family == "dma":
            self._policy = _DmaPolicy(done_indices=self._done_indices)
        elif family == "spi_master":
            self._policy = _SpiPolicy(master=True, macro_order=macro_order)
        elif family == "spi_xfer":
            self._policy = _SpiPolicy(master=False, macro_order=macro_order)
        else:
            self._policy = _GenericPolicy(action_dims, fields=action_fields,
                                          spec=spec,
                                          semantic_ir=self.semantic_ir)

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
        if (isinstance(self._policy, _GenericPolicy) and
                family == "generic" and plan_dims == self.action_dims):
            self._policy = _GenericPolicy(
                self.action_dims, fields=_parse_action_fields(spec), spec=spec,
                planned_program=plan.get("program"),
                semantic_ir=self.semantic_ir)

    @staticmethod
    def _infer_dims(spec: str) -> int:
        return _infer_action_dims(spec)
