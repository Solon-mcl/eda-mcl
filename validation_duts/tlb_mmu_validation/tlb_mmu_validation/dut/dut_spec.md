# tlb_mmu_validation DUT specification

## 1. Overview

This held-out DUT models an ASID-tagged translation lookaside buffer backed by
a synthetic page-table walk. Each accepted memory request either hits the TLB,
walks the page table and fills a way, or fails translation; the response
reports a single cumulative bin vector.

It is deliberately unlike the public serial/DMA controllers, the cache
controller validation DUT, the branch predictor, and the watchdog supervisor.
Coverage depends on permission semantics, address-space scoping, global pages,
hardware accessed/dirty bits, locked pages, and page-table-root changes.

Hidden evaluation parameters:

- `tlb_sets` (4 or 8): set count; the low index bits are salted.
- `tlb_ways` (2 or 4): associativity.
- `replace_xor` (0/1): perturbs the victim way of a full set.
- `asid_bits` (1..2): effective ASID width, so identifiers may collapse.
- `walk_salt` (0..15): XOR perturbation of every walked PTE field.
- `lock_enable` (0/1): enables locked pages that walk but never fill the TLB.

## 2. Action space (16 dims)

```text
action = [valid, vpn0, vpn1, vpn2, vpn3, op, priv, asid, global,
          fence, flush, satp, stall, reset_n, pad0, pad1]
```

- `valid`: submit a memory request.
- `vpn0..vpn3`: little-endian virtual page number.
- `op`: 0=load, 1=store, 2=fetch, 3=atomic read-modify-write.
- `priv`: 0=user, 1=supervisor.
- `asid`: address-space identifier, masked by the hidden ASID width.
- `global`: request the filled entry be marked global.
- `fence`: address-space-scoped TLB fence; local entries are dropped, global
  entries survive.
- `flush`: full TLB flush; every entry including global ones is dropped.
- `satp`: page-table-root change. Flips the hidden walk salt generation, clears
  the tracked accessed-bit history, and flushes the whole TLB.
- `stall`: front-end backpressure; a valid request is ignored while asserted.
- `reset_n`: active-low DUT reset. Functional coverage is retained by the
  harness across DUT-only reset pulses.

Scope pulses take effect on the same cycle as the request they accompany, so a
single action can fence and then access.

## 3. Translation behavior

A request is looked up in the salted set. A way matches when the VPN is equal
and either the entry is global or its ASID matches under the hidden width.
A same-VPN entry that fails only the ASID test is reported as an ASID conflict.

On a miss the page table is walked. The walked PTE is invalid for one in seven
salted indices (page fault), may be locked (walk succeeds but nothing is
installed), and otherwise yields read/write/execute/user permission bits plus a
physical page number. Store and atomic accesses need write permission; fetch
needs execute permission; user accesses need the user bit. Invalid PTEs take
priority over permission faults, which take priority over ASID conflicts.

Successful translations set the accessed bit, and store/atomic accesses set the
dirty bit. Coverage includes access classes, both privilege modes, every TLB
lookup and walk outcome, the full fault taxonomy, accessed/dirty transitions,
fence and flush classes, entry scope, functional crosses, and eight temporal
sequences for refill-then-hit, victim replacement, ASID aliasing, fence
eviction, global survival, fault recovery, stall recovery, and page-table-root
change.
