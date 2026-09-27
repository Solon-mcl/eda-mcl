# opaque-name variant of the same translation DUT

## 1. Overview

This held-out DUT models an ASID-tagged translation lookaside buffer backed by
a synthetic page-table walk. Each accepted memory request either hits the TLB,
walks the page table and fills a way, or fails translation; the response
reports a single cumulative bin vector.

It is deliberately unlike the public serial/DMA controllers, the cache
controller validation DUT, the branch predictor, and the watchdog supervisor.
Coverage depends on permission semantics, address-space scoping, f8 pages,
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
action = [f0, f1, f2, f3, f4, f5, f6, f7, f8,
          f9, f10, f11, f12, f13, f14, f15]
```

- `f0`: submit a memory request.
- `f1..f4`: little-endian virtual page number.
- `f5`: 0=load, 1=store, 2=fetch, 3=atomic read-modify-write.
- `f6`: 0=user, 1=supervisor.
- `f7`: address-space identifier, masked by the hidden ASID width.
- `f8`: request the filled entry be marked f8.
- `f9`: address-space-scoped TLB f9; local entries are dropped, f8
  entries survive.
- `f10`: full TLB f10; every entry including f8 ones is dropped.
- `f11`: page-table-root change. Flips the hidden walk salt generation, clears
  the tracked accessed-bit history, and flushes the whole TLB.
- `f12`: front-end backpressure; a f0 request is ignored while asserted.
- `f13`: active-low DUT reset. Functional coverage is retained by the
  harness across DUT-only reset pulses.

Scope pulses take effect on the same cycle as the request they accompany, so a
single action can f9 and then access.

## 3. Translation behavior

A request is looked up in the salted set. A way matches when the VPN is equal
and either the entry is f8 or its ASID matches under the hidden width.
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
f9 and f10 classes, entry scope, functional crosses, and eight temporal
sequences for refill-then-hit, victim replacement, ASID aliasing, f9
eviction, f8 survival, fault recovery, f12 recovery, and page-table-root
change.
