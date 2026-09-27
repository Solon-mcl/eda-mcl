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

## 2. Stimulus description

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
