# ecc_memory_validation DUT specification

## 1. Overview

This held-out DUT is an eight-word, 32-bit SECDED-protected memory with
explicit fault injection and a background scrubber. Reads distinguish clean,
single-bit-correctable, and multi-bit-uncorrectable words. Correction takes a
hidden number of cycles and can be stalled; scrubbing visits words in a hidden
salted stride order.

It differs from all existing validation DUTs by centering coverage on data
integrity, latent faults, correction latency, and maintenance traffic.

Hidden evaluation parameters:

- `syndrome_xor` (0..31): remaps injected bit positions and scrub origin.
- `scrub_stride` (odd value 1,3,5,7): permutation of the eight-word scan.
- `correction_latency` (1..6): cycles between SBE detection and repair.
- `zero_on_dbe` (0/1): whether an uncorrectable read returns zero.
- `poison_word` (0..7): word that promotes two injected faults to a poison DBE.

## 2. Action space (16 dims)

```text
action = [write, read, address, d0, d1, d2, d3,
          inject, bit_a, bit_b, scrub, stall, reset_n, pad0, pad1, pad2]
```

- `write` stores a word and clears all latent errors at the address.
- `read` starts a clean, correctable, or uncorrectable read.
- `inject` inserts one error at `bit_a`; a different `bit_b` inserts a second.
  Bit numbers are interpreted after the hidden syndrome remap.
- `scrub` services the current hidden scrub address and advances the scan.
- `stall` pauses correction and scrub progress.
- `reset_n` is active low; harness coverage persists across DUT-only reset.

## 3. Coverage intent

The 72 bins cover operation states/results, all addresses, data and syndrome
classes, clean/SBE/DBE populations, scrub positions, correction and scrub
backpressure, functional crosses, and eight temporal sequences: clean
write/read, SBE correction, DBE detection, stalled correction recovery, scrub
repair, scrub DBE observation, overwrite repair, and a full scrub wrap.
