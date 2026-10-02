"""DUT-agnostic semantic IR extracted from Markdown specifications.

The parser intentionally knows common interface conventions, not individual
DUT names. Its output is a conservative schema used to keep generic stimulus
within legal shapes/ranges and to expose register-style transaction structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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


def action_declaration_lines(spec: str) -> set:
    """Line indices occupied by an ``action = [...]`` declaration.

    Such a line lists *every* field name, so treating it as a per-field
    description hands each field the whole signal list.  Measured consequence:
    enabling a description-driven rule ("description mentions asid") collapsed
    the bundled TLB DUT from 78/78 to 6/78 covered bins, because every field's
    "description" contained every other field's name.
    """
    occupied = set()
    for match in re.finditer(r"\baction\s*=\s*\[[\s\S]*?\]", spec,
                             flags=re.IGNORECASE):
        first = spec.count("\n", 0, match.start())
        last = spec.count("\n", 0, match.end())
        occupied.update(range(first, last + 1))
    return occupied


def _description_for(spec: str, name: str, skip_lines=()) -> str:
    escaped = re.escape(name)
    lines = []
    for index, line in enumerate(spec.splitlines()):
        if index in skip_lines:
            continue
        if re.search(rf"(?:`|\b){escaped}(?:`|\b)", line, re.IGNORECASE):
            lines.append(line.strip())
    return " ".join(lines)


def _infer_role(name: str, description: str) -> str:
    value = name.lower()
    text = (value + " " + description).lower()
    if value.startswith(("pad", "rsv", "spare", "dummy", "unused")) or "reserved" in text or "占位" in text:
        return "padding"
    if "reset" in value or value.startswith("rst"):
        return "reset"
    if (value in ("ch_sel", "channel_sel", "channel_select", "instance_sel") or
            (("ch" in value or "channel" in value or "instance" in value) and
             ("sel" in value or "select" in value))):
        return "instance_select"
    if (value in ("reg_we", "cfg_we", "conf_wr", "cfg_wr", "bus_wr", "wr_en",
                  "we", "wr", "write", "wren", "wr_enable", "write_enable") or
            value.endswith(("_we", "_wr", "_wren"))):
        return "write_enable"
    if (value in ("reg_re", "rd_en", "re", "rd", "oe", "oen", "read",
                  "read_enable") or value.endswith(("_re", "_rd"))):
        return "read_enable"
    if value in ("reg_addr", "register_addr", "cfg_addr", "csr_addr",
                 "conf_field", "cfg_field", "register_select",
                 "addr", "adr", "address", "reg_index"):
        return "register_address"
    if value in ("reg_wdata", "register_data", "cfg_data", "csr_wdata",
                 "write_data", "wdata", "rdata", "din", "dout", "data",
                 "dat", "payload", "word"):
        return "register_data"
    if re.match(r"^(?:d|data_?|wdata_?|din_?|dout_?|payload_?|dat_?)\d+$", value):
        return "data_lane"
    if re.match(r"^(?:a|va_?|pa_?|addr_?|vaddr_?|paddr_?|vpn_?|virt_?|phys_?)\d+$", value):
        return "address_lane"
    if re.match(r"^(?:p|pc_?)\d+$", value):
        return "pc_lane"
    if re.match(r"^(?:t|target_?)\d+$", value):
        return "target_lane"
    if value in ("op", "opcode", "command", "cmd", "kind", "type"):
        return "operation"
    if value in ("mode", "protocol", "format", "transfer_type"):
        return "mode"
    if value in ("asid", "context_id", "context", "address_space_id"):
        return "context_id"
    if value in ("priv", "privilege", "priv_mode", "mode_user", "lvl", "level", "ring"):
        return "privilege"
    if value in ("global", "global_entry", "shared", "scope", "glb", "glob", "gs"):
        return "scope"
    if (value in ("id", "tag", "txn_id", "transaction_id", "request_id",
                  "req_id", "source_id") or value.endswith(("_id", "_tag"))):
        return "transaction_id"
    if (value in ("last", "eop", "last_beat", "last_word") or
            any(token in value for token in
                ("length", "len", "count", "size", "beats", "burst_len"))):
        return "length"
    if (value == "be" or value in ("keep", "keeps", "strb_n") or
            any(token in value for token in
                ("mask", "strobe", "strb", "byte_en", "byte_enable"))):
        return "mask"
    if value in ("priority", "prio", "qos", "arbitration_class"):
        return "priority"
    if value in ("select", "selector", "source", "destination", "dest",
                 "grant", "gnt", "grant_id", "qid", "queue_id", "port",
                 "port_id", "chan", "channel"):
        return "selector"
    if value in ("push", "enqueue", "enq", "put", "tx_push"):
        return "queue_push"
    if value in ("pop", "dequeue", "deq", "get", "rx_pop"):
        return "queue_pop"
    if value in ("ack", "acknowledge", "response_ack", "irq_ack"):
        return "ack"
    if value in ("interrupt", "irq", "irq_in", "int_in", "irq_status",
                 "int_status", "irq_out", "intr"):
        return "interrupt"
    if value in ("lock", "acquire", "lock_req", "reserve"):
        return "acquire"
    if value in ("unlock", "release", "lock_release"):
        return "release"
    if value in ("credit", "credit_in", "grant_credit", "replenish"):
        return "credit"
    if any(token in value for token in
           ("wake", "wakeup", "sleep", "power_down", "powerdown", "clock_gate")):
        return "power"
    if "ready" in value or value in ("rdy", "rdyn", "rdy_out"):
        return "ready"
    if "stall" in value or "backpressure" in value or value in ("hold", "holdoff", "bp", "throttle"):
        return "stall"
    if ("flush" in value or "recover" in value or "abort" in value or
            "cancel" in value or "fence" in value or "invalidate" in value or "satp" in value or "root" in value or value in
            ("fence", "invalidate", "invalidation", "satp", "root_change")):
        return "recovery"
    if ("valid" in value or value in ("start", "request", "req", "vld",
                                      "vld_in", "launch") or
            re.match(r"^req\d*$", value) or re.match(r"^vld\d*$", value)):
        return "request"
    if (value in ("tick", "step", "advance", "clken", "clk_en", "cen") or
            "clock_enable" in value or
            (value.endswith("_en") and "clk" in value)):
        return "advance"
    if any(token in value for token in
           ("service", "kick", "feed", "trigger", "doorbell")):
        return "event"
    if any(token in value for token in ("fault", "error", "inject")):
        return "fault"
    if (value in ("enable", "en", "cs", "cs_n", "csn", "chip_select",
                  "select_n") or value.endswith(("_enable", "_csn"))):
        return "enable"
    return "scalar"


_BULLET_KEY = re.compile(
    r"^[-*+]\s+\*{0,2}`?([^`\s:：|*]+)`?\*{0,2}\s*[:：]")
# A single bullet often documents several fields at once, because they share a
# behaviour: "- `write_strobe` / `write_index`: write `data` into word 0..3".
# Treating only the first name as the key leaves every later name with an empty
# keyed description, which silently disables description-derived inference for
# exactly the fields whose names carry the least information.
_JOINT_KEY = re.compile(
    r"^[-*+]\s+(?:\*{0,2}`?[^`\s:：|*]+`?\*{0,2}\s*"
    r"(?:/|,|，|;|；|and|&|\+)\s*)+"
    r"\*{0,2}`?[^`\s:：|*]+`?\*{0,2}\s*[:：]")
_JOINT_KEY_NAME = re.compile(r"`([^`]+)`|\*{2}([^*]+)\*{2}")
_TABLE_KEY = re.compile(r"^\|\s*`?([^`\s|]+)`?\s*\|")
_RANGE_KEY = re.compile(r"^([A-Za-z_]+)(\d+)\.\.([A-Za-z_]*)(\d+)$")
_INDEXED_NAME = re.compile(r"^([A-Za-z_]+)(\d+)$")


def _key_covers(key: str, name: str) -> bool:
    """Does a bullet/table key address this field?

    Lane groups are documented as a range key (``- `vpn0..vpn3`: ...``), so a
    line keyed on the range belongs to every member of the range, not only to
    its first element.
    """
    if key.lower() == name.lower():
        return True
    span = _RANGE_KEY.match(key)
    indexed = _INDEXED_NAME.match(name)
    if not span or not indexed:
        return False
    if indexed.group(1).lower() != span.group(1).lower():
        return False
    first, last = int(span.group(2)), int(span.group(4))
    return min(first, last) <= int(indexed.group(2)) <= max(first, last)


def _joint_key_names(line: str):
    """Every field name a joint bullet key documents, in declaration order."""
    head = line.split(":", 1)[0].split("：", 1)[0]
    names = [match.group(1) or match.group(2)
             for match in _JOINT_KEY_NAME.finditer(head)]
    if len(names) < 2:
        # Unquoted style: "- a / b: ...";
        plain = re.sub(r"^[-*+]\s+", "", head)
        plain = plain.replace("*", "")
        parts = re.split(r"\s*(?:/|,|，|;|；|\band\b|&|\+)\s*", plain)
        names = [part.strip() for part in parts if part.strip()]
    return tuple(names)


def keyed_description_for(spec: str, name: str, skip_lines=()) -> str:
    """Description restricted to lines where *name* is the leading key.

    ``_description_for`` returns every line that merely mentions the field, and
    aggregate lines ("``a0..a3``: byte-address in little-endian byte lanes")
    mention several fields at once.  For semantic inference only the lines that
    are *about* this field are usable, i.e. a bullet key (``- `x`: ...``) or a
    table key (``| `x` | ... |``).
    """
    lines = []
    raw = spec.splitlines()
    for index, line in enumerate(raw):
        if index in skip_lines:
            continue
        stripped = line.strip()
        matched = False
        for pattern in (_BULLET_KEY, _TABLE_KEY):
            match = pattern.match(stripped)
            if match and _key_covers(match.group(1), name):
                matched = True
                break
        # A joint key such as "- `word_we` / `word_idx`: ..." belongs to every
        # name it lists, not just the first one.
        if not matched and _JOINT_KEY.match(stripped):
            matched = name in _joint_key_names(stripped)
        if not matched:
            continue
        # A bullet commonly wraps onto continuation lines before the next
        # bullet.  Those lines carry details the key line truncates away
        # ("Word 0 is the source address, word 1 the length, ..."), so fold
        # them in as well.
        lines.append(stripped)
        for follow in raw[index + 1:]:
            stripped_follow = follow.strip()
            if not stripped_follow:
                break
            if re.match(r"^[-*+]\s", stripped_follow) or stripped_follow.startswith("|"):
                break
            if stripped_follow.startswith("#"):
                break
            lines.append(stripped_follow)
    return " ".join(lines)


# Ordered, first-hit-wins triggers applied to the *keyed* description only.
# They exist to recover a role when the field name carries no information
# ("f0", "sig2"), which is the case that name-only inference cannot cover.
# Each trigger is a distinctive phrase rather than a single word, because the
# cost of a wrong role is a wasted candidate family, not a crash.
_DESCRIPTION_ROLE_RULES = (
    (("active-low reset", "active low reset", "external reset", "dut reset",
      "resets the", "clears the block", "复位"), "reset"),
    (("address-space identifier", "address space identifier",
      "address-space id", "address space id"), "context_id"),
    (("privilege level", "privilege", "supervisor", "user mode"), "privilege"),
    (("marked global", "global mapping", "global entry", "be marked global"),
     "scope"),
    # Deliberately vocabulary-free of family names: a maintenance/hazard
    # operation is recognised by what it does, not by the block it lives in.
    (("fence", "page-table root", "page-table-root", "flush", "invalidate"),
     "recovery"),
    (("backpressure", "ignored while asserted", "hold off", "deassert"),
     "stall"),
    (("permits one", "ready", "handshake"), "ready"),
    (("clock enable", "advances the", "advance the", "clk_en"), "advance"),
    (("request strobe", "submit a", "submits a", "assert to", "asserted to",
      "present a request", "启动"), "request"),
    (("0=load", "1=store", "0=read, 1=write", "read-modify-write",
      "opcode", "operation class"), "operation"),
    (("fault", "inject", "error"), "fault"),
    (("service-key", "service key", "doorbell", "kick", "trigger"), "event"),
    (("write data", "register write", "payload"), "register_data"),
    (("register address", "register index"), "register_address"),
    (("write enable", "write strobe", "strobe a write"), "write_enable"),
    (("read enable", "read strobe", "strobes a read", "read back"),
     "read_enable"),
    (("completion",), "interrupt"),
    (("byte-address", "byte address", "program counter", "little-endian",
      "little endian"), "address_lane"),
)


def _infer_role_from_description(keyed: str):
    text = keyed.lower()
    if not text:
        return None
    for triggers, role in _DESCRIPTION_ROLE_RULES:
        if any(trigger in text for trigger in triggers):
            return role
    return None


# A field can select *which slot* of a multi-word object a data write lands in
# ("write data into word 0..3", "selects the FIFO entry", "channel index").
# Such a field is a write-path qualifier, not an unknown scalar: the object is
# only written correctly once every slot carries its own value, so a strategy
# that sweeps values one at a time can never build a legal object.  The
# triggers describe the *act of selecting a slot*, never the name of the object
# the slots belong to.
_SLOT_SELECTOR_TRIGGERS = (
    "word 0", "word zero", "word index", "which word",
    "word select", "selected word", "written word",
    "slot index", "which slot", "slot select", "selected slot",
    "entry index", "entry select", "which entry",
    "bank select", "channel index", "channel select",
    "sub-word", "subword", "multi-word", "multi word",
)

# Slot meanings a spec can enumerate, in the order the object declares them.
# Only used to *label* a slot so the value written there can be chosen to
# exercise the design; a spec that names none still gets the full sweep.
_SLOT_MEANING_TRIGGERS = (
    ("source", "src"), ("destination", "dst"), ("length", "length"),
    ("size", "length"), ("count", "length"), ("mode", "mode"),
    ("flag", "flag"), ("address", "addr"), ("control", "ctrl"),
    ("status", "status"), ("tag", "tag"), ("data", "data"),
)


def detect_slot_selector(keyed: str):
    """Slot count and per-slot meaning labels, or ``None`` if not a selector.

    Returns ``(count, meanings)`` where ``count`` is the exclusive upper bound
    on the selector value and ``meanings`` maps a slot index to a coarse label
    (or ``None`` where the specification does not say).
    """
    text = keyed.lower()
    if not text:
        return None
    if not any(trigger in text for trigger in _SLOT_SELECTOR_TRIGGERS):
        return None
    span = re.search(rf"({_NUMBER})\s*(?:~|\.\.|到|至|-)\s*({_NUMBER})", text)
    if span:
        low, high = _number(span.group(1)), _number(span.group(2))
        count = max(low, high) + 1
    else:
        single = re.search(r"word\s*({_NUMBER})", text)
        count = _number(single.group(1)) + 1 if single else 4
    if not 2 <= count <= 32:
        return None
    meanings = {}
    for label, tag in _SLOT_MEANING_TRIGGERS:
        # Prose either spells the copula out ("word 0 is the source") or drops
        # it in a list ("word 1 the length, word 2 the destination").  Both
        # forms have to be read, or only the first slot ever gets a label.
        for match in re.finditer(
                rf"(?:word|slot|entry|index|register)?\s*({_NUMBER})\s*"
                rf"(?:(?:is|:|=)\s*)?(?:the\s+|a\s+|an\s+)?{label}\b", text):
            index = _number(match.group(1))
            if index < count:
                meanings.setdefault(index, tag)
    return count, meanings


def _infer_bounds(name: str, description: str, role: str) -> tuple[int, int, int]:
    text = description.lower()
    range_match = re.search(rf"({_NUMBER})\s*(?:~|\.\.|到|至)\s*({_NUMBER})", text)
    if range_match:
        low, high = _number(range_match.group(1)), _number(range_match.group(2))
        if high < low:
            low, high = high, low
        return low, high, max(1, high.bit_length())
    bit_match = re.search(r"\b(\d+)\s*(?:-?bits?\b|位)", text)
    if bit_match:
        width = min(32, max(1, int(bit_match.group(1))))
        return 0, (1 << width) - 1, width
    if role in ("data_lane", "address_lane", "pc_lane", "target_lane"):
        return 0, 255, 8
    if role == "register_address":
        return 0, 255, 8
    if role == "register_data":
        return 0, (1 << 32) - 1, 32
    if role in ("operation", "mode", "priority"):
        return 0, 3, 2
    if role in ("transaction_id", "selector"):
        return 0, 15, 4
    if role == "length":
        return 0, 255, 8
    if role == "mask":
        return 0, 255, 8
    return 0, 1, 1


def _parse_enums(description: str) -> dict[int, str]:
    output = {}
    for match in re.finditer(
            rf"\b({_NUMBER})\s*=\s*([A-Za-z_][A-Za-z0-9_ -]*?)(?=\s*[,;/，；.]|$)",
            description, re.IGNORECASE):
        output[_number(match.group(1))] = match.group(2).strip().lower()
    return output


def _port_domain_from_spec(spec: str, field_name: str) -> tuple[int, int] | None:
    """Read an explicit port enumeration shared by port action fields."""
    if "port" not in field_name.lower().split("_"):
        return None
    for match in re.finditer(r"\bports?\s+(?:are|:|=)\s*([^\n.]*)",
                             spec, re.IGNORECASE):
        values = [_number(item.group(1)) for item in re.finditer(
            rf"\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*({_NUMBER})\b",
            match.group(1), re.IGNORECASE)]
        if len(values) >= 2 and len(values) == len(set(values)) and max(values) <= 255:
            return min(values), max(values)
    return None


def _mesh_neighbor_probe_bound(spec: str, field_name: str) -> int | None:
    """Permit a one-hop destination probe around a declared mesh coordinate."""
    name = field_name.lower()
    match_name = re.fullmatch(r"(?:dest|destination|target)_([xy])", name)
    if not match_name or not re.search(r"\b(?:mesh|grid)\b", spec, re.IGNORECASE):
        return None
    match = re.search(r"\b(?:router|node)\s+at\s+coordinate\s*"
                      r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", spec, re.IGNORECASE)
    if not match:
        return None
    coordinate = int(match.group(1 if match_name.group(1) == "x" else 2))
    return coordinate + 1 if coordinate <= 15 else None


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
    # Non-empty only when the specification describes this field as selecting
    # a slot of a multi-word object.  Maps slot index to a coarse meaning label
    # (may be ``None``), so a write program can fill every slot.
    slot_meanings: dict = field(default_factory=dict)
    # Non-empty when the specification ties this input's level to a
    # configuration mode, e.g. "SSP mode `ss_in_n`=0".  Maps the mode token the
    # specification used (lowercased, e.g. ``ssp``) to the level the input has
    # to hold.  A transfer simply does not start when the level is wrong, so a
    # program that configures a mode without matching the level never reaches
    # the state it is aiming at.
    mode_levels: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RegisterFieldIR:
    name: str
    lsb: int
    msb: int
    # Free text the specification attached to this bit field, gathered from
    # wherever it appears (usually a prose section after the register table).
    # It is the only place a field's *meaning* is stated, so anything that has
    # to reason about a field rather than merely write it needs this.
    description: str = ""

    @property
    def mask(self) -> int:
        return ((1 << (self.msb - self.lsb + 1)) - 1) << self.lsb


@dataclass(frozen=True)
class PackedFieldIR:
    signal: str
    name: str
    lsb: int
    msb: int

    @property
    def mask(self) -> int:
        return ((1 << (self.msb - self.lsb + 1)) - 1) << self.lsb


@dataclass(frozen=True)
class RegisterIR:
    address: int
    name: str
    description: str = ""
    fields: tuple[RegisterFieldIR, ...] = ()


@dataclass(frozen=True)
class ConstraintIR:
    left: str
    operator: str
    right: str | int
    raw: str


@dataclass(frozen=True)
class DependencyIR:
    operation: str
    prerequisite: str
    relation: str
    raw: str


@dataclass(frozen=True)
class TimingConstraintIR:
    start: str
    end: str
    minimum_cycles: int
    maximum_cycles: int
    raw: str


@dataclass
class DutSemanticIR:
    action_dim: int
    fields: list[FieldIR]
    registers: list[RegisterIR]
    constraints: list[ConstraintIR]
    dependencies: list[DependencyIR] = field(default_factory=list)
    timing_constraints: list[TimingConstraintIR] = field(default_factory=list)
    packed_fields: list[PackedFieldIR] = field(default_factory=list)
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


def _register_field_notes(spec: str) -> dict:
    """Free-text notes attached to a bit field name, wherever they appear.

    A register table normally gives a coarse description of the whole register
    and leaves the per-bit meaning to a later prose section, so the sentence
    that actually explains a field lives *outside* the table.  Two shapes are
    read: a bolded name (``**SRL_TEST**：...``) and a bracketed one
    (``- `[0]=MW_MOD`：...``).
    """
    notes: dict[str, str] = {}
    for match in re.finditer(
            r"\*\*([A-Za-z_][A-Za-z0-9_]*)\*\*\s*[：:]\s*([^\n]+)", spec):
        notes.setdefault(match.group(1).lower(), match.group(2).strip())
    for match in re.finditer(
            r"`\[(\d+)(?::(\d+))?\]\s*=\s*([A-Za-z_][A-Za-z0-9_]*)`\s*[：:]"
            r"\s*([^\n]+)", spec):
        notes.setdefault(match.group(3).lower(), match.group(4).strip())
    return notes


def _parse_registers(spec: str) -> list[RegisterIR]:
    found = {}
    notes = _register_field_notes(spec)
    patterns = (
        r"\bregister\s+({_NUMBER})\s*\(\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\)\s*:?\s*([^\n]*)",
        r"^\s*\|\s*({_NUMBER})\s*\|\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\|([^\n]*)$",
    )
    for pattern in patterns:
        for match in re.finditer(pattern.format(_NUMBER=_NUMBER), spec,
                                 re.IGNORECASE | re.MULTILINE):
            address = _number(match.group(1))
            description = match.group(3).strip(" |")
            found.setdefault(address, RegisterIR(
                address, match.group(2).lower(), description,
                _fields_with_notes(description, notes)))
    range_pattern = re.compile(
        rf"^\s*\|\s*({_NUMBER})\s*(?:~|\.\.|到|至)\s*({_NUMBER})\s*\|\s*"
        r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*\|([^\n]*)$",
        re.IGNORECASE | re.MULTILINE)
    for match in range_pattern.finditer(spec):
        low, high = _number(match.group(1)), _number(match.group(2))
        if high < low:
            low, high = high, low
        # Avoid expanding malformed or enormous maps into runtime state.
        for address in range(low, min(high, low + 255) + 1):
            description = match.group(4).strip(" |")
            found.setdefault(address, RegisterIR(
                address, match.group(3).lower(), description,
                _fields_with_notes(description, notes)))
    return [found[key] for key in sorted(found)]


def _fields_with_notes(description: str, notes: dict) -> tuple[RegisterFieldIR, ...]:
    """Bit fields from the register table, annotated with their prose notes."""
    fields = _parse_register_fields(description)
    if not notes:
        return fields
    out = []
    for item in fields:
        note = notes.get(item.name)
        out.append(replace(item, description=note) if note else item)
    return tuple(out)


def _parse_register_fields(description: str) -> tuple[RegisterFieldIR, ...]:
    fields = []
    seen = set()
    for match in re.finditer(
            r"\[(\d+)(?::(\d+))?\]\s*=\s*"
            r"([A-Za-z_][A-Za-z0-9_]*)", description):
        left = int(match.group(1))
        right = int(match.group(2)) if match.group(2) is not None else left
        lsb, msb = min(left, right), max(left, right)
        name = match.group(3).lower()
        if name not in seen:
            fields.append(RegisterFieldIR(name, lsb, msb))
            seen.add(name)
    return tuple(fields)


def _parse_packed_fields(spec: str) -> list[PackedFieldIR]:
    """Extract documented packed-data layouts such as data[15:0] = len."""
    output = []
    seen = set()
    pattern = re.compile(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\[(\d+)(?::(\d+))?\]\s*=\s*([^\n]+)$",
        re.IGNORECASE | re.MULTILINE)
    for match in pattern.finditer(spec):
        signal = match.group(1).lower()
        left = int(match.group(2))
        right = int(match.group(3)) if match.group(3) is not None else left
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", match.group(4))
        tokens = [item.lower() for item in tokens
                  if not re.fullmatch(r"(?:bit|bits)\d*", item,
                                      re.IGNORECASE)]
        if not tokens:
            continue
        name = tokens[0]
        lsb, msb = min(left, right), max(left, right)
        key = (signal, name, lsb, msb)
        if key not in seen:
            output.append(PackedFieldIR(signal, name, lsb, msb))
            seen.add(key)
    return output


def _clean_symbol(value: str) -> str:
    return value.strip().strip("`'\".,:;，。；").lower()


# A specification can pin an input's level to a configuration mode.  The shape
# is prose, but a regular one: a mode token, the word "mode", and the level the
# input has to hold -- written in Chinese ("SSP 模式 `ss_in_n`=0") or in English
# ("SSP mode `ss_in_n`=0").  Several modes can share one clause joined by a
# slash ("SPI/Microwire 模式 `ss_in_n`=1"), so the token list is split apart.
_MODE_LEVEL_PATTERNS = (
    r"([A-Za-z0-9][A-Za-z0-9/\s、,]{0,32}?)\s*模式\s*`?\s*{field}\s*=\s*(\d+)",
    r"([A-Za-z0-9][A-Za-z0-9/\s,]{0,32}?)\s+mode\s*`?\s*{field}\s*=\s*(\d+)",
)
_MODE_TOKEN_SPLIT = re.compile(r"[/、,]|\s+or\s+|\s+and\s+|－|\s+")


def detect_mode_levels(spec: str, name: str) -> dict:
    """Levels an input must hold, keyed by the mode token the spec used.

    A transfer does not start when the level is wrong, so the level is not a
    cosmetic detail: a program that configures a mode without matching the
    level stops at the first precondition and never reaches the state it was
    aiming at.  Only the *pairing* is read from prose here; nothing about which
    mode is which is assumed.
    """
    escaped = re.escape(name)
    levels: dict[str, int] = {}
    for pattern in _MODE_LEVEL_PATTERNS:
        try:
            # ``replace`` instead of ``format``: the quantifiers in the pattern
            # (``{0,32}``) would otherwise be read as format fields.
            regex = re.compile(pattern.replace("{field}", escaped),
                               re.IGNORECASE)
        except re.error:
            continue
        for match in regex.finditer(spec):
            tokens, level = match.group(1), int(match.group(2))
            if level not in (0, 1):
                continue
            for token in _MODE_TOKEN_SPLIT.split(tokens):
                token = _clean_symbol(token)
                if token and 2 <= len(token) <= 20:
                    levels.setdefault(token, level)
    return levels


def _parse_structured_constraints(spec: str, field_names=()):
    constraints = []
    dependencies = []
    timings = []
    seen_constraints = set()
    seen_dependencies = set()
    known = {str(name).lower() for name in field_names}
    identifier = r"`?([A-Za-z_][A-Za-z0-9_]*)`?"
    operand = rf"(?:{_NUMBER}|`?[A-Za-z_][A-Za-z0-9_]*`?)"
    comparison = re.compile(
        rf"{identifier}\s*(<=|>=|==|=|<|>)\s*({operand})", re.IGNORECASE)
    chained = re.compile(
        rf"({operand})\s*(<=|<)\s*{identifier}\s*(<=|<)\s*({operand})",
        re.IGNORECASE)
    timing_range = re.compile(
        rf"{identifier}[^\n]{{0,80}}?(?:after|之后|后)[^\n]{{0,40}}?"
        rf"{identifier}[^\n]{{0,40}}?({_NUMBER})\s*(?:~|\.\.|到|至)\s*"
        rf"({_NUMBER})\s*(?:cycles?|拍|周期)", re.IGNORECASE)

    for raw_line in spec.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("```", "<!--")):
            continue
        for match in chained.finditer(line):
            low, low_op, middle, high_op, high = match.groups()
            for left, op, right in ((middle, ">=" if low_op == "<=" else ">", low),
                                    (middle, high_op, high)):
                key = (_clean_symbol(left), op, _clean_symbol(right))
                if key not in seen_constraints:
                    value = _number(right) if re.fullmatch(_NUMBER, right,
                                                           re.IGNORECASE) else _clean_symbol(right)
                    constraints.append(ConstraintIR(key[0], op, value, line))
                    seen_constraints.add(key)
        for match in comparison.finditer(line):
            left, op, right = match.groups()
            key = (_clean_symbol(left), op, _clean_symbol(right))
            if key in seen_constraints:
                continue
            value = _number(right) if re.fullmatch(_NUMBER, right,
                                                   re.IGNORECASE) else _clean_symbol(right)
            constraints.append(ConstraintIR(key[0], op, value, line))
            seen_constraints.add(key)

        dependency_patterns = (
            (rf"{identifier}\s+(?:requires?|needs?)\s+{identifier}", "requires"),
            (rf"{identifier}[^\n]{{0,40}}?(?:only when|仅当|只有在)\s+{identifier}",
             "condition"),
            (rf"{identifier}[^\n]{{0,40}}?(?:after|之后|后)\s+{identifier}",
             "after"),
            (rf"{identifier}[^\n]{{0,40}}?(?:before|之前|前)\s+{identifier}",
             "before"),
        )
        for pattern, relation in dependency_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if not match:
                continue
            operation = _clean_symbol(match.group(1))
            prerequisite = _clean_symbol(match.group(2))
            actual_relation = relation
            if relation == "condition":
                actual_relation = ("write_condition" if
                                   re.search(r"(?:write|写|(?:^|_)w(?:r|e)(?:_|\b))",
                                             f"{operation} {line}", re.IGNORECASE)
                                   else "requires")
            # Free-form prose yields spurious pairs such as
            # "accesses requires the".  Keep a dependency only when at least
            # one side is a declared action field, which is the only case the
            # executor can actually act on.
            if known and operation not in known and prerequisite not in known:
                continue
            key = (operation, prerequisite, actual_relation)
            if key not in seen_dependencies:
                dependencies.append(DependencyIR(operation, prerequisite,
                                                 actual_relation, line))
                seen_dependencies.add(key)

        match = timing_range.search(line)
        if match:
            end, start, low, high = match.groups()
            timings.append(TimingConstraintIR(
                _clean_symbol(start), _clean_symbol(end),
                _number(low), _number(high), line))
    return constraints, dependencies, timings


# The complete role vocabulary produced by _infer_role.  Single source of truth
# for anybody that has to *validate* a role from outside (the LLM enricher), so
# a second, drifting copy of this list cannot appear.
SEMANTIC_ROLES = frozenset((
    "ack", "acquire", "address_lane", "advance", "context_id", "credit",
    "data_lane", "enable", "event", "fault", "instance_select", "interrupt",
    "length", "mask", "mode", "operation", "padding", "pc_lane", "power",
    "priority", "privilege", "queue_pop", "queue_push", "read_enable", "ready",
    "recovery", "register_address", "register_data", "release", "request",
    "reset", "scalar", "scope", "selector", "stall", "target_lane",
    "transaction_id", "write_enable",
))


def build_semantic_ir(spec: str, role_overrides=None,
                      override_all_roles=False) -> DutSemanticIR:
    names = parse_action_fields(spec)
    declared_match = re.search(r"action\s+space\s*\(\s*(\d+)\s*dims?\s*\)",
                               spec, re.IGNORECASE)
    declared = int(declared_match.group(1)) if declared_match else None
    dims = infer_action_dim(spec, names)
    skip_lines = action_declaration_lines(spec)
    overrides = {str(key).strip().lower(): str(value).strip().lower()
                 for key, value in (role_overrides or {}).items()}
    fields = []
    for index, name in enumerate(names):
        description = _description_for(spec, name, skip_lines)
        keyed = keyed_description_for(spec, name, skip_lines)
        role = _infer_role(name, description)
        if role == "scalar":
            # Name-only inference failed; the spec prose may still say what the
            # signal is.  Only the keyed lines are trusted here, since mixed
            # lines mention several fields at once.
            role = _infer_role_from_description(keyed) or role
        # The selector names identify bit positions even when their prose
        # mentions faults; they are not Boolean fault-injection triggers.
        if re.fullmatch(r"(?:fault_)?bit_?[a-z0-9]+", name.lower()):
            role = "scalar"
        override = overrides.get(name.lower())
        if override is not None and override in SEMANTIC_ROLES:
            # External role hints fill gaps; they do not second-guess a role the
            # local rules already resolved, and can never turn a field into
            # padding (padding must stay zeroed).
            if override != "padding" and (role == "scalar" or
                                          override_all_roles):
                role = override
        minimum, maximum, width = _infer_bounds(name, description, role)
        # The action declaration can name a scalar input while a nearby prose
        # sentence defines its shared port enumeration. An explicit enum on the
        # field itself is equally authoritative, including its final item.
        enums = _parse_enums(description)
        if enums and role in ("scalar", "operation", "mode", "priority", "selector"):
            minimum = min(minimum, min(enums))
            maximum = max(maximum, max(enums))
            width = max(width, maximum.bit_length())
        port_domain = _port_domain_from_spec(spec, name)
        if port_domain is not None and role == "scalar":
            minimum = min(minimum, port_domain[0])
            maximum = max(maximum, port_domain[1])
            width = max(width, maximum.bit_length())
        neighbor = _mesh_neighbor_probe_bound(spec, name)
        if neighbor is not None and role == "scalar":
            maximum = max(maximum, neighbor)
            width = max(width, maximum.bit_length())
        # A pair of fault-site selectors may be documented only as bit
        # numbers, while the word width appears elsewhere in the spec. Their
        # values are positions within that word, not one-bit Boolean flags.
        if (role == "scalar" and re.fullmatch(
                r"(?:fault_)?bit_?[a-z0-9]+", name.lower()) and
                re.search(r"\bbit\b|bit_|位", description.lower())):
            word_widths = [int(value) for value in re.findall(
                r"\b(\d+)\s*-\s*bit\b", spec.lower())]
            if word_widths:
                positions = min(64, max(word_widths))
                maximum = max(maximum, positions - 1)
                width = max(width, maximum.bit_length())
        active_low = (name.endswith("_n") or
                      "active-low" in description.lower() or
                      "低有效" in description)
        slot = detect_slot_selector(keyed) if role == "scalar" else None
        slot_meanings = {}
        if slot is not None:
            count, labels = slot
            # The declared range is authoritative for how many slots exist; the
            # prose only labels them.  Keep the two consistent so a program
            # never writes a slot the field cannot express.
            maximum = max(int(maximum), count - 1)
            minimum = min(int(minimum), 0)
            slot_meanings = {index: labels.get(index)
                             for index in range(count)}
        # Only single-bit inputs are considered: a mode pins a level, and a
        # multi-bit field is not a level.  The pairing is extracted even when
        # the role rules could not classify the field, because an unclassified
        # input is exactly the one a program would otherwise never drive.
        mode_levels = {}
        if not slot_meanings and int(maximum) <= 1:
            mode_levels = detect_mode_levels(spec, name)
        fields.append(FieldIR(name, index, role, minimum, maximum, width,
                              active_low, enums,
                              description, slot_meanings, mode_levels))
    constraints, dependencies, timings = _parse_structured_constraints(
        spec, names)
    return DutSemanticIR(dims, fields, _parse_registers(spec), constraints,
                         dependencies, timings, _parse_packed_fields(spec),
                         declared_dim=declared)
