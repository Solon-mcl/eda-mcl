"""Compile cross-coverage targets into joint, legal register assignments."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .coverage_targets import CoverageTargetIR
from .semantic_ir import DutSemanticIR, RegisterFieldIR, RegisterIR


@dataclass(frozen=True)
class JointCoverageCandidate:
    target_index: int
    target_name: str
    register_writes: tuple[tuple[int, int], ...]
    mapped_conditions: tuple[str, ...]
    unresolved_conditions: tuple[str, ...]
    hold_cycles: int

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
    return choices[0] if choices else None


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
    if target.kind != "cross" or not target.target_conditions:
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
        if canonical == "protocol":
            updates = _protocol_updates(ir, value)
        else:
            found = _find_field(ir, signal)
            if found:
                register, field = found
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
            words[register.address] = _set_field(current, field, encoded)
        mapped.append(signal)
    # A joint candidate needs at least two mapped stimulus conditions. Partial
    # candidates with one mapped input are left to normal exploration.
    if len(mapped) < 2:
        return None
    writes = tuple(sorted(words.items()))
    return JointCoverageCandidate(
        target.index, f"{target.coverpoint}.{target.bin_name}", writes,
        tuple(mapped), tuple(unresolved), 128 if long_wait else 64)


def compile_joint_candidates(ir: DutSemanticIR,
                             targets: list[CoverageTargetIR]):
    return [candidate for target in targets
            if (candidate := compile_joint_candidate(ir, target)) is not None]
