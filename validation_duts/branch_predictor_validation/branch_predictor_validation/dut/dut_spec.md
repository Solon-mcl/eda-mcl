# branch_predictor_validation DUT specification

## 1. Overview

This held-out DUT models a hybrid CPU branch predictor: a 16-entry gshare
pattern-history table (PHT), a two-set/two-way branch target buffer (BTB), and
a bounded return-address stack (RAS). Each accepted branch resolves in the same
transaction, updates predictor state, and exposes only cumulative bin feedback.

It is deliberately unlike the public serial/DMA controllers and the cache
validation DUT. Coverage depends on learned history, destructive table aliasing,
saturating counters, target prediction, and stack depth.

Hidden evaluation parameters:

- `history_bits` (4..6): effective global-history length.
- `index_salt` (0..15): XOR perturbation of PHT and BTB indices.
- `replacement_xor` (0/1): perturbs the full-set BTB victim.
- `ras_depth` (4..8): physical return-stack capacity.

## 2. Action space (16 dims)

```text
action = [valid, p0, p1, p2, p3, kind, actual_taken,
          t0, t1, t2, t3, stall, flush, reset_n, pad0, pad1]
```

- `valid`: submit a resolved control-flow instruction.
- `p0..p3`: little-endian program counter.
- `kind`: 0=conditional, 1=unconditional jump, 2=call, 3=return.
- `actual_taken`: resolved direction.
- `t0..t3`: little-endian resolved target.
- `stall`: front-end backpressure; a valid request is ignored while asserted.
- `flush`: pipeline recovery pulse; clears global history but retains tables.
- `reset_n`: active-low predictor reset. Functional coverage is retained by the
  harness across DUT-only reset pulses.

## 3. Prediction behavior

Conditional direction uses the MSB of a two-bit saturating PHT counter indexed
by PC, global history, and the hidden salt. Other branch kinds predict taken,
except an empty-stack return. Taken predictions use the BTB target; returns use
the RAS top. Calls push `PC+4`; returns pop when a value exists.

Coverage includes branch/outcome classes, all PHT counter states, BTB hit/miss
and replacement, history patterns, RAS boundaries, stall/flush behavior,
functional crosses, and eight temporal sequences for saturation, aliasing,
mispredict recovery, call-return matching, overflow, and flush recovery.
