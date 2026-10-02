# noc_router_validation DUT specification

## 1. Overview

This held-out DUT models a five-port mesh Network-on-Chip router at coordinate
(1,1). Ports are LOCAL=0, NORTH=1, EAST=2, SOUTH=3, and WEST=4. Each input has
a bounded FIFO. Head flits request an output, credit availability gates
forwarding, and one winner per output is selected when requests conflict.

VC0 may adapt around congestion; the hidden escape VC always uses deterministic
dimension-order routing. This makes the DUT structurally different from the
existing buses, memories, predictors, and safety controllers.

Hidden evaluation parameters:

- `xy_order` (0/1): deterministic X-first versus Y-first routing.
- `priority_seed` (0..4): initial cyclic arbitration priority.
- `escape_vc` (0/1): VC forced onto deterministic routing.
- `congestion_threshold` (1..3): congestion required before adaptive detour.
- `buffer_depth` (2..5): per-input FIFO depth.

## 2. Action space (16 dims)

```text
action = [valid, in_port, dest_x, dest_y, vc, flit_type,
          d0, d1, d2, d3, credit_mask, congestion_mask,
          router_stall, reset_n, pad0, pad1]
```

- `flit_type`: 0=head, 1=body, 2=tail, 3=single-flit packet.
- `credit_mask`: bit per output; one means the downstream accepts a flit.
- `congestion_mask`: bit per output indicating congestion.
- `router_stall`: freezes arbitration/draining but still permits enqueue.
- `reset_n`: active-low reset. Harness coverage persists across DUT reset.

## 3. Coverage intent

The 75 bins cover all ports, both VCs, all flit types, deterministic/adaptive/
escape/local routing, enqueue/forward/backpressure/drop/conflict outcomes,
buffer levels, credit and congestion classes, arbitration multiplicity,
functional port/VC/flit crosses, and fourteen temporal sequences for normal
forwarding, credit recovery, conflict resolution, adaptive detour, escape
routing, buffer overflow, head-body-tail formation, local delivery, sustained
credit blocking, five-way conflict draining, adaptive-to-escape correlation,
full-buffer recovery, congestion-dependent route flipping, and tail-flit
credit recovery.
