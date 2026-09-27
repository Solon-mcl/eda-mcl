#!/usr/bin/env python3
"""Derive naming/style variants of one DUT spec for name-invariance checks.

The generator is deliberately kept in the repository: the variants are a
measurement instrument, so how they were produced has to be reproducible.
Four variants are written next to this file:

``tlb_shortnames.md``
    Same semantics, short names and business synonyms (``valid`` -> ``vld``,
    ``vpn`` -> ``va``, ``pad`` -> ``rsv`` ...).  Checks whether the role
    vocabulary covers the abbreviations real specifications use.

``tlb_prose.md``
    Field names unchanged, but the action table is rewritten as free prose.
    Checks whether structure recovery survives a narrative description.

``tlb_opaque.md``  (adversarial lower bound)
    Every occurrence of a field name is replaced by ``f<N>``, *including the
    prose*.  This destroys the vocabulary in the description as well, so it is
    a hard floor rather than a realistic case.

``tlb_opaque_realistic.md``  (realistic case)
    Signal names become opaque (``f<N>``) but the prose keeps its vocabulary:
    only the declaration block and the backticked field keys are rewritten.
    An unseen DUT with undocumented names still documents what the signals do,
    so this is the case the description fallback exists for.

Run:  python3 synthesis/make_spec_variants.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "validation_duts" / "tlb_mmu_validation" /
          "tlb_mmu_validation" / "dut" / "dut_spec.md")
OUT = ROOT / "synthesis" / "spec_variants"

# Original declaration order of the TLB action vector.
FIELDS = ["valid", "vpn0", "vpn1", "vpn2", "vpn3", "op", "priv", "asid",
          "global", "fence", "flush", "satp", "stall", "reset_n", "pad0",
          "pad1"]

SHORT_FORMS = [
    ("valid", "vld"), ("vpn0", "va0"), ("vpn1", "va1"), ("vpn2", "va2"),
    ("vpn3", "va3"), ("op", "cmd"), ("priv", "lvl"), ("asid", "tag"),
    ("global", "glb"), ("fence", "tlbfence"), ("flush", "tlbflush"),
    ("satp", "ptroot"), ("stall", "hold"), ("reset_n", "rst_n"),
    ("pad0", "rsv0"), ("pad1", "rsv1"),
]


def _read_source() -> str:
    return SOURCE.read_text(encoding="utf-8")


def _replace_word(text: str, old: str, new: str) -> str:
    return re.sub(r"\b%s\b" % re.escape(old), new, text)


def make_shortnames(source: str) -> str:
    text = source
    for old, new in SHORT_FORMS:
        text = _replace_word(text, old, new)
    return text.replace(
        "# tlb_mmu_validation DUT specification",
        "# shortened-name variant of the same translation DUT")


def make_prose(source: str) -> str:
    start = source.index("## 2. Action space")
    end = source.index("## 3. Translation behavior")
    block = """## 2. Stimulus description

The stimulus vector drives a memory request interface. One line submits a
request, four following lines carry the little-endian virtual page number, one
selects the operation class (load, store, fetch or atomic read-modify-write),
one is the privilege level, one is the address-space identifier, and one asks
for a global mapping. Three maintenance lines drop local entries, drop every
entry, and change the page-table root; a backpressure line makes a submitted
request be ignored while it is held; and a reset line clears the block. Two
further lines are unused.

```text
action = [valid, vpn0, vpn1, vpn2, vpn3, op, priv, asid, global,
          fence, flush, satp, stall, reset_n, pad0, pad1]
```

"""
    return source[:start] + block + source[end:]


def make_opaque(source: str) -> str:
    text = source
    for index, name in enumerate(FIELDS):
        text = _replace_word(text, name, "f%d" % index)
    return text.replace(
        "# tlb_mmu_validation DUT specification",
        "# opaque-name variant of the same translation DUT")


def make_opaque_realistic(source: str) -> str:
    """Opaque signal names, but the prose keeps its vocabulary."""
    rename = {name: "f%d" % index for index, name in enumerate(FIELDS)}

    declaration = re.search(r"\baction\s*=\s*\[[\s\S]*?\]", source,
                            flags=re.IGNORECASE)
    text = source
    if declaration:
        block = declaration.group(0)
        rewritten = block
        for old, new in rename.items():
            rewritten = _replace_word(rewritten, old, new)
        text = text[:declaration.start()] + rewritten + text[declaration.end():]
    for old, new in rename.items():
        escaped = re.escape(old)
        # bullet key:  - `name`: ...
        text = re.sub(r"(?m)^([-*+]\s+)(`?)%s(`?)(\s*[:：])" % escaped,
                      lambda m, n=new: "%s`%s`%s" % (m.group(1), n, m.group(4)),
                      text)
        # table key:  | `name` | ...
        text = re.sub(r"(?m)^(\|\s*)(`?)%s(`?)(\s*\|)" % escaped,
                      lambda m, n=new: "%s`%s`%s" % (m.group(1), n, m.group(4)),
                      text)
    # lane groups are documented as a range key, e.g. `vpn0..vpn3`
    for old, new in rename.items():
        for other, other_new in rename.items():
            if other == old:
                continue
            text = re.sub(r"`%s\.\.%s`" % (re.escape(old), re.escape(other)),
                          "`%s..%s`" % (new, other_new), text)
    return text.replace(
        "# tlb_mmu_validation DUT specification",
        "# opaque-signal variant (prose intact) of the same translation DUT")


def main() -> int:
    source = _read_source()
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tlb_shortnames.md": make_shortnames(source),
        "tlb_prose.md": make_prose(source),
        "tlb_opaque.md": make_opaque(source),
        "tlb_opaque_realistic.md": make_opaque_realistic(source),
    }
    for name, text in outputs.items():
        path = OUT / name
        path.write_text(text, encoding="utf-8")
        print("wrote", path.relative_to(ROOT), len(text), "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
