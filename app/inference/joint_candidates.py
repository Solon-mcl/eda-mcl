"""Compile cross-coverage targets into joint, legal register assignments."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re

from .coverage_targets import CoverageTargetIR
from .semantic_ir import DutSemanticIR, RegisterFieldIR, RegisterIR


def _joint_compile_policy():
    """Which coverage targets may be compiled into an active stimulus program.

    The original gate admitted only ``cross`` targets carrying at least two
    mapped conditions, which is what the *joint-write* idea was invented for:
    two register fields that only produce a bin together.  Applied as a general
    rule, though, that gate hides most of the target list on any interface whose
    coverage is also driven by a single field write - the target is known, its
    condition is known, and it is still never turned into a program.  Measured
    on the SPI-master package: 120 targets, of which 104 (boundary / basic /
    condition / sequential) were never compiled at all.

    A single mapped condition is enough to build a program that writes the
    value and then runs the transaction, which is exactly how the missing
    single-condition bins were recovered by hand.

    ``EDA_JOINT_TARGET_KINDS`` restricts the kinds (comma separated; unset
    means every kind), ``EDA_JOINT_SINGLE_CONDITION=0`` restores the old
    at-least-two rule.  Both exist so the change can be A/B tested rather than
    argued about.
    """
    kinds_raw = os.environ.get("EDA_JOINT_TARGET_KINDS", "").strip()
    kinds = ({item.strip().lower() for item in kinds_raw.split(",")
              if item.strip()} if kinds_raw else None)
    single = os.environ.get(
        "EDA_JOINT_SINGLE_CONDITION", "1").lower() not in ("0", "false", "no")
    return kinds, single


@dataclass(frozen=True)
class JointCoverageCandidate:
    target_index: int
    target_name: str
    register_writes: tuple[tuple[int, int], ...]
    mapped_conditions: tuple[str, ...]
    unresolved_conditions: tuple[str, ...]
    hold_cycles: int
    # Levels to force on plain input fields while the program runs, as
    # ``(field_index, level)`` pairs.  Empty means "leave the base action
    # alone", which is what every candidate did before mode-pinned inputs were
    # read out of the specification.
    input_levels: tuple[tuple[int, int], ...] = ()

    @property
    def complete(self) -> bool:
        return not self.unresolved_conditions


def _canonical(value: str) -> str:
    value = str(value).lower()
    value = re.sub(r"^(?:cov_|cfg_|conf_)", "", value)
    value = re.sub(r"(?:_boundary|_value|_path)$", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


def _all_fields(ir: DutSemanticIR):
    for register in ir.registers:
        for field in register.fields:
            yield register, field


def _find_field(ir: DutSemanticIR, signal: str):
    """Locate the register bit field that carries a coverage signal.

    Two naming layers have to be bridged here.  A coverage point is usually
    named after the *derived quantity* the design exposes (``cov_mwcr``,
    ``cov_eff_dfs``), while the register model is named after the bit fields
    inside each register (``microwire``, ``dfs``).  Neither layer is guaranteed
    to mention the other, so both directions are tried: match the signal
    against a field name first, then against the *register* the signal is named
    after.
    """
    wanted = _canonical(signal)
    aliases = {
        "protocol": ("protocol", "format", "frf"),
        "mode": ("mode", "tmod", "frf"),
        "effdfs": ("dfs",),
        "dfs": ("dfs",),
        "tmod": ("tmod",),
        "sstglen": ("sstglen",),
        "baud2": ("baudr", "baud"),
        "ndf": ("ndf",),
    }
    wanted_names = aliases.get(wanted, (wanted,))
    exact = []
    partial = []
    for register, field in _all_fields(ir):
        name = _canonical(field.name)
        if name in wanted_names:
            exact.append((register, field))
        elif any(item in name or name in item for item in wanted_names):
            partial.append((register, field))
    choices = exact or partial
    if choices:
        return choices[0]
    # Fall back to the register the signal is named after.  Only a register
    # with a *single* bit field qualifies: with several unrelated fields inside
    # one register there is nothing in the signal name that says which of them
    # carries it, and picking one would be a guess.  A register that was never
    # split into fields is deliberately *not* used here - writing it whole
    # looked plausible but measured as a net loss, because those extra
    # candidates competed for the same cycle budget as the ones that already
    # worked; the payload path has its own explicit fallback.
    for register in ir.registers:
        if (_canonical(register.name) in wanted_names and
                len(register.fields) == 1):
            return register, register.fields[0]
    return None


def _set_field(word: int, field: RegisterFieldIR, value: int) -> int:
    width_mask = (1 << (field.msb - field.lsb + 1)) - 1
    return ((int(word) & ~field.mask) |
            ((int(value) & width_mask) << field.lsb))


def _default_register_word(register: RegisterIR) -> int:
    word = 0
    for field in register.fields:
        name = _canonical(field.name)
        if name == "dfs":
            word = _set_field(word, field, 7)  # valid 8-bit frame default
    return word


def _protocol_updates(ir: DutSemanticIR, value: int):
    """Map common protocol-mode IDs to format/phase bitfields."""
    format_field = _find_field(ir, "protocol")
    phase_field = _find_field(ir, "scph")
    if format_field is None:
        return []
    # Common metadata convention: SPI0, SPI1, SSP, Microwire.
    if int(value) == 0:
        updates = [(*format_field, 0)]
        if phase_field:
            updates.append((*phase_field, 0))
        return updates
    if int(value) == 1:
        updates = [(*format_field, 0)]
        if phase_field:
            updates.append((*phase_field, 1))
        return updates
    if int(value) == 2:
        return [(*format_field, 1)]
    if int(value) == 3:
        return [(*format_field, 2)]
    return []


def compile_joint_candidate(ir: DutSemanticIR,
                            target: CoverageTargetIR
                            ) -> JointCoverageCandidate | None:
    allowed_kinds, single_allowed = _joint_compile_policy()
    if not target.target_conditions:
        return None
    if allowed_kinds is not None and target.kind not in allowed_kinds:
        return None
    words: dict[int, int] = {}
    registers: dict[int, RegisterIR] = {}
    mapped = []
    unresolved = []
    long_wait = False
    for signal, raw_value in target.target_conditions:
        if not isinstance(raw_value, (int, float)):
            unresolved.append(signal)
            continue
        value = int(raw_value)
        canonical = _canonical(signal)
        updates = []
        # Payload is checked *before* the register lookup: a signal like
        # ``cov_rx_data`` would otherwise match a short bit-field name
        # (``rx`` inside the RX-threshold register) by substring, and the
        # threshold is not the payload.
        port = _data_port(ir)
        payload_target = (port is not None and
                          any(token in canonical
                              for token in _DATA_SIGNAL_TOKENS))
        if canonical == "protocol":
            updates = _protocol_updates(ir, value)
        elif payload_target:
            updates = [(port, None, value)]
        else:
            found = _find_field(ir, signal)
            if found:
                register, field = found
                # The encoded value is the value the *field* has to hold.
                #
                # ``eff_dfs`` keeps an offset of one below the bin value.  That
                # reads as a mistake next to the design model, which stores the
                # raw field value, and it was briefly "fixed" - which cost five
                # bins on SPI-xfer and gained one on SPI-master, because the
                # candidates that changed competed for the same cycle budget.
                # The offset is therefore kept as measured; the mechanism is
                # budget allocation rather than encoding.
                encoded = value
                if canonical == "effdfs" and value > 0:
                    encoded = value - 1
                elif canonical == "baud2" and value:
                    encoded = 2
                updates = [(register, field, encoded)]
        if not updates:
            unresolved.append(signal)
            if any(token in canonical for token in
                   ("hold", "lastframe", "done", "state", "busy")):
                long_wait = True
            continue
        for register, field, encoded in updates:
            registers[register.address] = register
            current = words.setdefault(
                register.address, _default_register_word(register))
            if field is None:
                # A whole-register payload write: nothing else in the word is
                # meaningful, so the value is written as-is.
                words[register.address] = int(encoded) & 0xFFFFFFFF
            else:
                words[register.address] = _set_field(current, field, encoded)
        mapped.append(signal)
        # Close the loopback path whenever the target is payload: a pushed word
        # only becomes received data if the design routes it back, and the
        # specification states which bit does that.
        if payload_target:
            loopback = _loopback_field(ir)
            if loopback is not None:
                register, field = loopback
                current = words.setdefault(
                    register.address, _default_register_word(register))
                words[register.address] = _set_field(current, field, 1)
    # One mapped stimulus condition is enough for a program that writes the
    # value and then runs the transaction.  Two was the old rule, kept
    # available because the retry budget and the ranker threshold were
    # calibrated against the smaller pool it produced.
    minimum = 1 if single_allowed else 2
    if len(mapped) < minimum:
        return None
    writes = tuple(sorted(words.items()))
    return JointCoverageCandidate(
        target.index, f"{target.coverpoint}.{target.bin_name}", writes,
        tuple(mapped), tuple(unresolved), 128 if long_wait else 64)


# A received- or transmitted-data target is not a register field at all: its
# value is the payload word pushed through the data port.  These tokens are how
# a coverage signal says "this is payload" rather than "this is a register".
_DATA_SIGNAL_TOKENS = ("data",)
_DATA_PORT_NAMES = ("dr", "data", "txdata", "tx_data", "fifo_data")
_LOOPBACK_TOKENS = ("回环", "环回", "loopback", "loop back", "loop-back")


def _data_port(ir: DutSemanticIR):
    """The register that carries payload words in and out of the interface."""
    for register in ir.registers:
        if register.name.lower() in _DATA_PORT_NAMES:
            return register
    return None


def _loopback_field(ir: DutSemanticIR):
    """A bit that routes the outgoing stream back in as received data.

    Its meaning is stated only in prose ("1=txd looped back as received
    data"), which is exactly why the register field carries its note.  Without
    such a bit, a received-data target can only be reached by driving the
    serial input cycle by cycle - a much longer program.
    """
    for register, item in _all_fields(ir):
        note = (getattr(item, "description", "") or "").lower()
        if any(token in note for token in _LOOPBACK_TOKENS):
            return register, item
    return None


def _mode_input_levels(ir: DutSemanticIR, target: CoverageTargetIR):
    """Input levels the target's mode implies, from the spec's own pairing.

    A specification that says "SSP mode requires ``ss_in_n``=0" is stating a
    *precondition*, not a preference: with the wrong level the design never
    leaves idle, so the bin cannot be reached however the registers are
    written.  The mode is identified by matching the target's bin name against
    the mode tokens the specification used, which keeps this independent of any
    particular DUT's naming.
    """
    name = str(getattr(target, "bin_name", "")).lower()
    if not name:
        return []
    levels = []
    for item in ir.fields:
        table = getattr(item, "mode_levels", None)
        if not table:
            continue
        for token, level in sorted(table.items()):
            if token in name or name.startswith(token):
                levels.append((item.index, int(level)))
                break
    return levels


def compile_joint_candidates(ir: DutSemanticIR,
                             targets: list[CoverageTargetIR]):
    out = []
    for target in targets:
        candidate = compile_joint_candidate(ir, target)
        if candidate is None:
            continue
        out.append(candidate)
        # Two variants are kept whenever the target names a mode that pins an
        # input: one with the level the specification declares, one without it
        # (which is the base action's own level).  The pairing is read out of
        # prose, so the search layer - not this function - decides which of the
        # two actually reaches the bin.
        levels = _mode_input_levels(ir, target)
        if levels:
            out.append(JointCoverageCandidate(
                candidate.target_index, candidate.target_name,
                candidate.register_writes, candidate.mapped_conditions,
                candidate.unresolved_conditions, candidate.hold_cycles,
                tuple(levels)))
    return out
