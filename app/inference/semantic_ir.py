"""DUT-agnostic semantic IR extracted from Markdown specifications.

The parser intentionally knows common interface conventions, not individual
DUT names. Its output is a conservative schema used to keep generic stimulus
within legal shapes/ranges and to expose register-style transaction structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


_NUMBER = r"(?:0x[0-9a-f]+|\d+)"


def _number(text: str) -> int:
    return int(text, 0)


def expand_action_field(raw: str) -> list[str]:
    raw = re.sub(r"[#/].*$", "", raw).strip().strip("`'\"")
    match = re.search(
        r"\b([A-Za-z_][A-Za-z_]*)(\d+)\s*\.\.\s*(?:[A-Za-z_][A-Za-z_]*)?(\d+)\b",
        raw)
    if not match:
        match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\s*:\s*(\d+)\]", raw)
    if match:
        prefix, left, right = match.group(1), int(match.group(2)), int(match.group(3))
        step = 1 if right >= left else -1
        return [f"{prefix}{index}".lower()
                for index in range(left, right + step, step)]
    identifier = re.search(r"[A-Za-z_][A-Za-z0-9_]*", raw)
    return [identifier.group(0).lower()] if identifier else []


def parse_action_fields(spec: str) -> list[str]:
    match = re.search(r"\baction\s*=\s*\[([\s\S]*?)\]", spec,
                      flags=re.IGNORECASE)
    if not match:
        return []
    fields = []
    for raw in match.group(1).split(","):
        fields.extend(expand_action_field(raw))
    return fields


def infer_action_dim(spec: str, fields=None) -> int:
    fields = fields if fields is not None else parse_action_fields(spec)
    patterns = (
        r"action\s+space\s*\(\s*(\d+)\s*dims?\s*\)",
        r"action[^\n]{0,80}?(\d+)\s*(?:dims?|dimensions?)",
        r"动作(?:空间|向量)[^\n]{0,80}?(\d+)\s*维",
    )
    declared = None
    for pattern in patterns:
        match = re.search(pattern, spec, flags=re.IGNORECASE)
        if match:
            declared = max(1, int(match.group(1)))
            break
    if fields:
        # Prefer the explicit ordered declaration. Keep a mismatch visible in
        # the IR rather than silently inventing unnamed positions here.
        return len(fields)
    return declared or 1


def _description_for(spec: str, name: str) -> str:
    escaped = re.escape(name)
    lines = []
    for line in spec.splitlines():
        if re.search(rf"(?:`|\b){escaped}(?:`|\b)", line, re.IGNORECASE):
            lines.append(line.strip())
    return " ".join(lines)


def _infer_role(name: str, description: str) -> str:
    value = name.lower()
    text = (value + " " + description).lower()
    if value.startswith("pad") or "reserved" in text or "占位" in text:
        return "padding"
    if "reset" in value or value.startswith("rst"):
        return "reset"
    if value in ("reg_we", "cfg_we", "wr_en", "write_enable") or value.endswith("_we"):
        return "write_enable"
    if value in ("reg_re", "rd_en", "read_enable") or value.endswith("_re"):
        return "read_enable"
    if value in ("reg_addr", "register_addr", "cfg_addr", "csr_addr"):
        return "register_address"
    if re.match(r"^(?:d|data_|wdata_)\d+$", value):
        return "data_lane"
    if re.match(r"^(?:a|addr_)\d+$", value):
        return "address_lane"
    if re.match(r"^(?:p|pc_)\d+$", value):
        return "pc_lane"
    if re.match(r"^(?:t|target_)\d+$", value):
        return "target_lane"
    if value in ("op", "opcode", "command", "cmd", "kind", "type"):
        return "operation"
    if "ready" in value:
        return "ready"
    if "stall" in value or "backpressure" in value:
        return "stall"
    if "flush" in value or "recover" in value:
        return "recovery"
    if "valid" in value or value in ("start", "request", "req"):
        return "request"
    if value in ("tick", "step", "advance") or "clock_enable" in value:
        return "advance"
    if any(token in value for token in ("service", "kick", "feed", "trigger")):
        return "event"
    if any(token in value for token in ("fault", "error", "inject")):
        return "fault"
    if value in ("enable", "en") or value.endswith("_enable"):
        return "enable"
    return "scalar"


def _infer_bounds(name: str, description: str, role: str) -> tuple[int, int, int]:
    text = description.lower()
    range_match = re.search(rf"({_NUMBER})\s*(?:~|\.\.|到|至)\s*({_NUMBER})", text)
    if range_match:
        low, high = _number(range_match.group(1)), _number(range_match.group(2))
        if high < low:
            low, high = high, low
        return low, high, max(1, high.bit_length())
    bit_match = re.search(r"\b(\d+)\s*(?:-?bit|bits?|位)\b", text)
    if bit_match:
        width = min(32, max(1, int(bit_match.group(1))))
        return 0, (1 << width) - 1, width
    if role in ("data_lane", "address_lane", "pc_lane", "target_lane"):
        return 0, 255, 8
    if role == "register_address":
        return 0, 255, 8
    if role == "operation":
        return 0, 3, 2
    return 0, 1, 1


def _parse_enums(description: str) -> dict[int, str]:
    output = {}
    for match in re.finditer(
            rf"\b({_NUMBER})\s*=\s*([A-Za-z_][A-Za-z0-9_ -]*?)(?=\s*[,;/，；]|$)",
            description, re.IGNORECASE):
        output[_number(match.group(1))] = match.group(2).strip().lower()
    return output


@dataclass(frozen=True)
class FieldIR:
    name: str
    index: int
    role: str
    minimum: int = 0
    maximum: int = 1
    width: int = 1
    active_low: bool = False
    enums: dict[int, str] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class RegisterIR:
    address: int
    name: str
    description: str = ""


@dataclass
class DutSemanticIR:
    action_dim: int
    fields: list[FieldIR]
    registers: list[RegisterIR]
    constraints: list[str]
    declared_dim: int | None = None

    def indices(self, *roles: str) -> list[int]:
        wanted = set(roles)
        return [item.index for item in self.fields if item.role in wanted]

    def field(self, index: int) -> FieldIR | None:
        return self.fields[index] if 0 <= index < len(self.fields) else None

    @property
    def register_addresses(self) -> list[int]:
        return sorted({item.address for item in self.registers})

    def clamp(self, index: int, value: float) -> float:
        item = self.field(index)
        if item is None:
            return float(value)
        return float(min(item.maximum, max(item.minimum, value)))


def _parse_registers(spec: str) -> list[RegisterIR]:
    found = {}
    patterns = (
        r"\bregister\s+({_NUMBER})\s*\(\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\)\s*:?\s*([^\n]*)",
        r"^\s*\|\s*({_NUMBER})\s*\|\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\|([^\n]*)$",
    )
    for pattern in patterns:
        for match in re.finditer(pattern.format(_NUMBER=_NUMBER), spec,
                                 re.IGNORECASE | re.MULTILINE):
            address = _number(match.group(1))
            found.setdefault(address, RegisterIR(
                address, match.group(2).lower(), match.group(3).strip(" |")))
    return [found[key] for key in sorted(found)]


def build_semantic_ir(spec: str) -> DutSemanticIR:
    names = parse_action_fields(spec)
    declared_match = re.search(r"action\s+space\s*\(\s*(\d+)\s*dims?\s*\)",
                               spec, re.IGNORECASE)
    declared = int(declared_match.group(1)) if declared_match else None
    dims = infer_action_dim(spec, names)
    fields = []
    for index, name in enumerate(names):
        description = _description_for(spec, name)
        role = _infer_role(name, description)
        minimum, maximum, width = _infer_bounds(name, description, role)
        active_low = (role == "reset" and
                      (name.endswith("_n") or "active-low" in description.lower() or
                       "低有效" in description))
        fields.append(FieldIR(name, index, role, minimum, maximum, width,
                              active_low, _parse_enums(description), description))
    constraints = [line.strip() for line in spec.splitlines()
                   if re.search(r"(?:<=|>=|<|>)", line) and
                   not line.lstrip().startswith(("```", "<!--"))]
    return DutSemanticIR(dims, fields, _parse_registers(spec), constraints,
                         declared_dim=declared)
