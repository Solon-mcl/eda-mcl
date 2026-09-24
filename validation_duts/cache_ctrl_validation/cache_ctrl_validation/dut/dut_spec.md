# cache_ctrl_validation DUT specification

## 1. Overview

This held-out DUT is a two-way, four-set, write-back data-cache controller with
16-byte cache lines.  It accepts one CPU transaction at a time and models
lookup, dirty eviction, refill, response, targeted invalidation, and full-cache
flush.  This differs deliberately from the public SPI and DMA DUTs: stimuli are
atomic memory transactions, address aliasing drives behavior, and the important
coverage depends on persistent cache contents and replacement history.

Hidden evaluation parameters are `replacement_xor` (which perturbs victim
selection), `memory_latency` (1..8 cycles), and `poison_tag` (a tag whose refill
raises the error class).  Agents see only the cumulative coverage vector.

## 2. Action space (16 dims)

```text
action = [req_valid, opcode, a0, a1, a2, a3, d0, d1, d2, d3,
          mem_ready, reset_n, pad0, pad1, pad2, pad3]
```

- `req_valid`: request strobe; accepted only in IDLE.
- `opcode`: 0=read, 1=write, 2=invalidate addressed line, 3=flush all lines.
- `a0..a3`: byte-address in little-endian byte lanes.
- `d0..d3`: write-data in little-endian byte lanes.
- `mem_ready`: permits one writeback/refill/flush memory beat.  Low values
  create backpressure.
- `reset_n`: active-low reset. Functional coverage is not reset by the harness.
- remaining dimensions are reserved.

Address decomposition is `offset=addr[3:0]`, `set=addr[5:4]`, and
`tag=addr[31:6]`. Requests must be separated by idle cycles if the previous
request has not completed.

## 3. Observable behavior

The controller traverses IDLE, LOOKUP, WRITEBACK, REFILL, RESPOND, and
FLUSH_SCAN. Reads and writes may hit either way. A miss chooses an invalid way
first, otherwise an LRU-derived victim; dirty victims are written back before
refill. A write miss performs write allocate. Invalidate drops one matching
line. Flush scans every set/way and writes dirty lines before invalidating them.

Coverage contains state and operation bins, address/data boundaries, cache
occupancy, hit/miss/eviction classes, operation/state and set/way crosses, and
multi-cycle sequences such as cold-miss-to-refill, conflict eviction, dirty
writeback, write-then-read hit, backpressure recovery, and complete flush.

## 4. Validation intent

The reference `greedy` policy is an oracle-like reachability check, not a model
to train on. The `random` policy is a fair baseline. The main submission's
generic path should be evaluated without importing this package's policy.
