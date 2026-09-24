// SystemVerilog coverage reference; coverage_meta.json is the executable mirror.
module watchdog_safety_covergroup(
    input logic clk, input logic reset_n,
    input logic [2:0] state, event_state, result,
    input logic [1:0] counter_class, fault_class,
    input logic enabled, cfg_locked, safety_locked, key_phase,
    input logic seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8
);
    covergroup cg @(posedge clk);
        state_cp: coverpoint state { bins all[] = {[0:5]}; }
        result_cp: coverpoint result { bins all[] = {[0:7]}; }
        counter_cp: coverpoint counter_class { bins all[] = {[0:3]}; }
        fault_cp: coverpoint fault_class { bins all[] = {[0:3]}; }
        enable_cp: coverpoint enabled { bins off={0}; bins on={1}; }
        cfg_lock_cp: coverpoint cfg_locked { bins off={0}; bins on={1}; }
        safety_lock_cp: coverpoint safety_locked { bins off={0}; bins on={1}; }
        key_cp: coverpoint key_phase { bins first={0}; bins second={1}; }
        state_result: cross event_state, result;
        seq1_cp: coverpoint seq1 { bins hit={1}; }
        seq2_cp: coverpoint seq2 { bins hit={1}; }
        seq3_cp: coverpoint seq3 { bins hit={1}; }
        seq4_cp: coverpoint seq4 { bins hit={1}; }
        seq5_cp: coverpoint seq5 { bins hit={1}; }
        seq6_cp: coverpoint seq6 { bins hit={1}; }
        seq7_cp: coverpoint seq7 { bins hit={1}; }
        seq8_cp: coverpoint seq8 { bins hit={1}; }
    endgroup
    cg coverage = new;
endmodule
