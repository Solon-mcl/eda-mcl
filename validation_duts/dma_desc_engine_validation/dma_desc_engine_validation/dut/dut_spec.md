# dma_desc_engine_validation DUT specification

## 1. Overview

This held-out DUT is a descriptor-chain DMA engine.  It never moves payload
bytes: it walks a linked list of descriptors in memory, validates each one
before issuing the burst, and stops on the first invalid descriptor.

It differs deliberately from the public SPI and DMA DUTs and from every other
held-out DUT.  The public DMA transfers one programmed region; the cache
controller caches lines; the TLB translates addresses; the branch predictor
predicts control flow; the watchdog watches time.  Here coverage is driven by
pointer chasing through a descriptor list, descriptor legality checks, the
completion handshake, the retry budget, word writes over a descriptor, the
mid-transfer abort, and the chain limit.

Hidden evaluation parameters:

- `desc_base`: base address of the descriptor region.
- `corrupt_xor` (0/1): perturbs the address the engine uses to find the next
  descriptor, so a chain that walks correctly under the default layout can
  still fail to fetch.
- `chain_limit` (2..6): descriptors accepted before the engine stops on limit.
- `ack_latency` (1..4): cycles allowed for the completion acknowledgement.
- `retry_limit` (1..3): acknowledgements withheld before retry exhaustion.

## 2. Action space (16 dims)

```text
action = [start, fetch_valid, word_we, word_idx,
          p0, p1, p2, p3, d0, d1, d2, d3, ack, abort, reset_n, pad0]
```

- `start`: arm the engine in IDLE.  Each start rotates the descriptor base by
  64 bytes, so consecutive chains live at different addresses.
- `fetch_valid`: latch the descriptor at `desc_ptr` into the engine.
- `word_we` / `word_idx`: write `data` into descriptor word 0..3 before the
  engine latches it.  Word 0 is the source address, word 1 the length, word 2
  the destination address, word 3 the mode and address flag.
- `p0..p3`: `desc_ptr` in little-endian byte lanes.
- `d0..d3`: `data` in little-endian byte lanes.
- `ack`: sequence completion.  `abort`: stop the chain mid-transfer.
- `reset_n`: active-low reset.  The harness retains functional coverage across
  DUT-only reset pulses.

A `fetch_valid` for a pointer that is not the head of the current chain does
not latch a descriptor; it reports a fetch miss.

## 3. Engine behaviour

The engine traverses IDLE, FETCH, VALIDATE, ISSUE, WAIT_ACK, DONE, ERROR, and
ABORT.  VALIDATE raises exactly one failure class for an illegal descriptor:
zero or too-low source/destination, unaligned addresses, a length that pushes
either end past the 64 KB window, zero or oversized length, an unsupported
mode, an unsupported address flag.  A failed check moves the engine to ERROR
without completing the burst.

Mode 0 issues the whole burst.  Mode 1 requires a completion acknowledgement
within the hidden latency, otherwise the retry budget is spent and the engine
falls into ERROR with the retry-exhausted class.  Mode 3 issues a single
word per cycle.  A completed descriptor retires: the destination advances by
the length, the chain counter increments, and the engine stops on the chain
limit or the end-of-list flag.

Coverage contains state and result bins, the twelve descriptor failure classes,
word index, dirty and handshake boundaries, chain occupancy, state/result and
state/failure crosses, and eight multi-cycle sequences: start, chain limit,
mid-burst abort, check-failure stop, burst stride, error recovery, done
handshake, and reset recovery.

## 4. Validation intent

The reference `greedy` policy is a reachability check, not a model to train on.
The `random` policy is a fair baseline.  The main submission's generic path
must be evaluated without importing this package's policy.
