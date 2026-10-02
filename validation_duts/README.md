# Held-out validation DUTs

This directory is intentionally separate from `public_duts`: its contents are
for out-of-distribution validation and must not be used to train or tune the
stimulus generator.

`cache_ctrl_validation/cache_ctrl_validation` is a transaction-level, 2-way
set-associative write-back cache controller.  It keeps the public-DUT package
contract (`dut`, `local_sim.py`, `coverage_simulator.py`, `harness.py`, and
`inference_interface.py`) while changing the protocol, state machine, action
schema, and coverage topology substantially.

Quick check:

```bash
python tools/run_experiments.py --dut cache_ctrl_validation --backend local \
  --agent greedy --steps 2000 --interval 200

python tools/run_experiments.py --dut branch_predictor_validation --backend local \
  --agent greedy --steps 700 --interval 100

python tools/run_experiments.py --dut watchdog_safety_validation --backend local \
  --agent greedy --steps 300 --interval 50

python tools/run_experiments.py --dut tlb_mmu_validation --backend local \
  --agent greedy --steps 1500 --interval 100

python tools/run_experiments.py --dut dma_desc_engine_validation --backend local \
  --agent greedy --steps 1500 --interval 100

python tools/run_experiments.py --dut ecc_memory_validation --backend local \
  --agent greedy --steps 500 --interval 50

python tools/run_experiments.py --dut noc_router_validation --backend local \
  --agent greedy --steps 500 --interval 50
```

The package is local-simulation-first.  The RTL and SystemVerilog covergroup
are readable reference artifacts; unlike the three image DUTs, no precompiled
Verilator binary is shipped.

`branch_predictor_validation/branch_predictor_validation` is a stateful hybrid
branch predictor combining a gshare PHT, two-way BTB, and return-address stack.
Its hidden indexing and history parameters make it useful for validating
feedback-driven exploration of aliasing and long-lived microarchitectural state.

`watchdog_safety_validation/watchdog_safety_validation` is a windowed watchdog
and safety supervisor. It emphasizes long temporal windows, ordered secret-key
service, configuration locking, timeout reset, and multi-fault escalation.

`tlb_mmu_validation/tlb_mmu_validation` is an ASID-tagged TLB and synthetic
page-table walker with permission faults, global mappings, fences, and hidden
replacement/index parameters.

`dma_desc_engine_validation/dma_desc_engine_validation` is a linked descriptor
engine with legality checks, retry/ack timing, abort, and hidden chain limits.

`ecc_memory_validation/ecc_memory_validation` is an ECC-protected memory with
fault injection, delayed correction, uncorrectable-error policy, and a salted
background scrub order.

`noc_router_validation/noc_router_validation` is a five-port virtual-channel
mesh router with buffered arbitration, credit backpressure, adaptive detours,
and a deterministic escape VC.
