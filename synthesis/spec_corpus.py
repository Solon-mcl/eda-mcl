"""Synthetic specification corpus for parsing-generalization checks.

Every entry is a made-up DUT spec written in a different interface idiom and
naming convention.  The corpus exists to measure two things that the seven
bundled DUTs cannot measure, because they all share one author and one spec
style:

1. does the parser recover the intended field role from the name alone?
2. does the policy still produce at least one stimulus candidate?

``expect_roles`` lists only unambiguous expectations; a value may be a string
or a tuple of acceptable roles.  ``min_candidates`` is a lower bound on the
number of candidates the generic search layer must produce.
"""

CORPUS = [
    # ---------------- register-mapped peripherals ----------------
    dict(
        id="reg_full_names", style="register",
        spec="""## Register peripheral
action = [reset_n, write_enable, reg_addr, reg_wdata, reg_re, irq]
Write registers through reg_addr / reg_wdata with write_enable.
reg_re reads back. irq reports completion. reset_n is active low.
""",
        expect_roles={"reset_n": "reset", "write_enable": "write_enable",
                      "reg_addr": "register_address", "reg_wdata": "register_data",
                      "reg_re": "read_enable", "irq": "interrupt"},
        min_candidates=1,
    ),
    dict(
        id="reg_short_names", style="register",
        spec="""## Bus peripheral (short names)
action = [rst_n, we, addr, wdata, re, irq]
rst_n clears the block. we strobes a write to addr with wdata.
re strobes a read. irq pulses when the transfer completes.
""",
        expect_roles={"rst_n": "reset", "we": "write_enable",
                      "addr": "register_address", "wdata": "register_data",
                      "re": "read_enable", "irq": "interrupt"},
        min_candidates=1,
    ),
    dict(
        id="reg_chip_select", style="register",
        spec="""## Chip-select style bus
action = [cs_n, wr, adr, din, dout]
cs_n selects the device. wr low means write. adr is the byte address.
din carries write data, dout carries read data.
""",
        expect_roles={"cs_n": ("enable", "write_enable"),
                      "wr": "write_enable", "adr": "register_address",
                      "din": "register_data", "dout": "register_data"},
        min_candidates=1,
    ),
    dict(
        id="reg_packed_data", style="register",
        spec="""## Packed config port
action = [cs, we, addr, data]
data[15:0] = len
data[20:16] = mode
data[23:21] = burst
Write to addr to program length, mode and burst in one word.
""",
        expect_roles={"we": "write_enable", "addr": "register_address"},
        min_candidates=1,
    ),
    dict(
        id="reg_numbered_lanes", style="register",
        spec="""## Multi-lane register port
action = [we, addr0, addr1, data0, data1]
we strobes. addr0/addr1 are the low and high halves of the register address.
data0/data1 are the low and high halves of the write data.
""",
        expect_roles={"we": "write_enable", "addr0": "address_lane",
                      "addr1": "address_lane", "data0": "data_lane",
                      "data1": "data_lane"},
        min_candidates=1,
    ),
    dict(
        id="reg_enable_start", style="register",
        spec="""## Config and start block
action = [enable, start, mode, cfg_addr, cfg_data]
enable powers the block. mode selects the operating mode (0..3).
Write cfg_addr with cfg_data, then pulse start.
""",
        expect_roles={"enable": "enable", "start": "request",
                      "mode": "mode", "cfg_addr": "register_address",
                      "cfg_data": "register_data"},
        min_candidates=1,
    ),

    # ---------------- streaming / handshake ----------------
    dict(
        id="stream_valid_ready", style="stream",
        spec="""## Streaming interface
action = [valid, ready, data, last, keep]
valid asserts a beat, ready is the sink handshake. data is the payload,
last marks the final beat, keep is the byte mask.
""",
        expect_roles={"valid": "request", "ready": "ready",
                      "last": "length", "keep": "mask"},
        min_candidates=1,
    ),
    dict(
        id="stream_abbrev", style="stream",
        spec="""## Streaming interface (abbreviated names)
action = [vld, rdy, dat, sop, eop]
vld marks a valid beat, rdy is the handshake back. dat is the payload word.
sop and eop mark start and end of packet.
""",
        expect_roles={"vld": "request", "rdy": "ready"},
        min_candidates=1,
    ),
    dict(
        id="stream_stall", style="stream",
        spec="""## Request-grant with backpressure
action = [req, gnt, stall, payload]
Raise req to request a transfer. gnt grants it. The sink may assert stall
to hold the transfer off; payload carries the data word.
""",
        expect_roles={"req": "request", "stall": "stall"},
        min_candidates=1,
    ),
    dict(
        id="stream_prose_only", style="stream",
        spec="""This module moves packets between two clock domains.
It exposes a request line, an opcode, a sixteen bit payload and a ready
handshake. Raise request to submit work; the module may hold you off with
backpressure until it has room.
action = [request, opcode, payload, ready]
""",
        expect_roles={"request": "request", "opcode": "operation",
                      "ready": "ready"},
        min_candidates=1,
    ),
    dict(
        id="stream_fifo", style="stream",
        spec="""## FIFO port
action = [push, pop, full, empty, data]
push enqueues data, pop dequeues. full and empty are status flags.
""",
        expect_roles={"push": "queue_push", "pop": "queue_pop"},
        min_candidates=1,
    ),

    # ---------------- queue / credit / arbitration ----------------
    dict(
        id="queue_credit", style="queue",
        spec="""## Credit-based queue
action = [push, pop, credit, qid, data]
push enqueues a word, pop dequeues. credit is the number of free slots
(0..7). qid selects the queue (0..3). data is the payload.
""",
        expect_roles={"push": "queue_push", "pop": "queue_pop",
                      "credit": "credit"},
        min_candidates=1,
    ),
    dict(
        id="queue_channel_sel", style="queue",
        spec="""## Channel multiplexed queue
action = [enq, deq, chan_sel, credit_in, din, dout]
enq enqueues into the channel chosen by chan_sel, deq removes.
credit_in reports the free credits. din/dout are the data ports.
""",
        expect_roles={"enq": "queue_push", "deq": "queue_pop",
                      "chan_sel": "instance_select", "credit_in": "credit"},
        min_candidates=1,
    ),
    dict(
        id="arbiter_priority", style="queue",
        spec="""## Arbiter
action = [req0, req1, req2, grant, prio]
The three req lines request the shared resource. grant is the arbitration
result. prio sets the priority class of the requester.
""",
        expect_roles={"grant": "selector", "prio": "priority"},
        min_candidates=1,
    ),
    dict(
        id="doorbell_service", style="queue",
        spec="""## Doorbell and service window
action = [doorbell, ack, service, fault]
Ring the doorbell to request service. ack acknowledges it. service marks
the service window. fault injects an error.
""",
        expect_roles={"doorbell": "event", "ack": "ack",
                      "service": "event", "fault": "fault"},
        min_candidates=1,
    ),

    # ---------------- address / memory / translation ----------------
    dict(
        id="mem_lanes", style="memory",
        spec="""## Memory port
action = [valid, a0, a1, a2, a3, d0, d1, d2, d3, wr]
valid starts the access. a0..a3 carry the address bytes, d0..d3 the data
bytes. wr selects write direction.
""",
        expect_roles={"valid": "request", "a0": "address_lane",
                      "d3": "data_lane"},
        min_candidates=1,
    ),
    dict(
        id="tlb_translation", style="memory",
        spec="""## TLB translation port
action = [valid, vpn0, vpn1, op, priv, asid, global, fence, flush, satp]
valid starts a translation. vpn0/vpn1 are the virtual page number bytes.
op selects load or store. priv is the privilege level. asid is the address
space id. global marks a global mapping. fence, flush and satp change the
translation context.
""",
        expect_roles={"valid": "request", "vpn0": "address_lane",
                      "op": "operation", "priv": "privilege",
                      "asid": "context_id", "global": "scope",
                      "fence": "recovery", "flush": "recovery",
                      "satp": "recovery"},
        min_candidates=1,
    ),
    dict(
        id="cache_port", style="memory",
        spec="""## Cache maintenance port
action = [valid, addr, tag, set, we, flush, invalidate]
valid starts the access. addr is the line address, tag and set are its
fields. we selects write. flush and invalidate maintain the array.
""",
        expect_roles={"valid": "request", "we": "write_enable",
                      "flush": "recovery", "invalidate": "recovery"},
        min_candidates=1,
    ),
    dict(
        id="branch_port", style="memory",
        spec="""## Branch prediction port
action = [valid, pc0, pc1, target0, target1, kind, taken, stall]
valid marks a prediction request. pc0/pc1 carry the program counter bytes,
target0/target1 the branch target bytes. kind selects the branch class,
taken is the outcome. stall holds the pipeline.
""",
        expect_roles={"valid": "request", "pc0": "pc_lane",
                      "target0": "target_lane", "taken": "scalar"},
        min_candidates=1,
    ),

    # ---------------- control / state machine ----------------
    dict(
        id="fsm_control", style="control",
        spec="""## State machine control
action = [clk_en, state, next_state, tick, reset_n]
clk_en gates the clock. state and next_state are the FSM registers.
tick advances one step. reset_n is active low.
""",
        expect_roles={"tick": "advance", "reset_n": "reset"},
        min_candidates=1,
    ),
    dict(
        id="watchdog_style", style="control",
        spec="""## Watchdog with service window
action = [we, reg_addr, wdata, advance, service, fault, rst_n]
Write reg_addr with wdata. advance steps the internal timer. service kicks
the watchdog inside the legal window. fault injects an error. rst_n resets.
""",
        expect_roles={"we": "write_enable", "reg_addr": "register_address",
                      "wdata": "register_data", "advance": "advance",
                      "service": "event", "fault": "fault", "rst_n": "reset"},
        min_candidates=1,
    ),
    dict(
        id="irq_controller", style="control",
        spec="""## Interrupt controller
action = [irq_en, irq_ack, irq_status, mask, prio]
irq_en enables the interrupt line globally. irq_ack acknowledges a pending
interrupt. irq_status reports it. mask and prio configure it.
""",
        expect_roles={"irq_ack": "ack", "mask": "mask", "prio": "priority"},
        min_candidates=1,
    ),

    # ---------------- degenerate / hostile ----------------
    dict(
        id="minimal_no_desc", style="degenerate",
        spec="""## Opaque module
action = [a, b, c, d]
""",
        expect_roles={},
        min_candidates=1,
    ),
    dict(
        id="params_abcd", style="degenerate",
        spec="""## Parameter block
action = [a0, a1, a2, a3]
Four independent unsigned parameter bytes, no further meaning.
""",
        expect_roles={},
        min_candidates=1,
    ),
    dict(
        id="opaque_names", style="degenerate",
        spec="""## Vendor specific block
action = [foo, bar, baz, qux]
""",
        expect_roles={},
        min_candidates=1,
    ),
    dict(
        id="numbered_only", style="degenerate",
        spec="""## Wide port
action = [p0, p1, p2, p3, p4, p5]
Six unsigned bytes.
""",
        expect_roles={},
        min_candidates=1,
    ),
]
