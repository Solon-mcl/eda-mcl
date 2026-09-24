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
