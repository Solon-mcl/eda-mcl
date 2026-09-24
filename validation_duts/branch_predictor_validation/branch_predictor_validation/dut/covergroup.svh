// SystemVerilog functional-coverage reference; metadata is the local mirror.
module branch_predictor_covergroup(
    input logic clk, input logic rst_n,
    input logic [1:0] kind, outcome, btb_result, pht_before,
    input logic [1:0] ghr_class,
    input logic [3:0] ras_level,
    input logic actual_taken, stalled, flushed,
    input logic seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8
);
    covergroup cg @(posedge clk);
        kind_cp: coverpoint kind { bins all[] = {[0:3]}; }
        outcome_cp: coverpoint outcome { bins all[] = {[0:3]}; }
        btb_cp: coverpoint btb_result { bins all[] = {[0:3]}; }
        pht_cp: coverpoint pht_before { bins all[] = {[0:3]}; }
        history_cp: coverpoint ghr_class { bins all[] = {[0:3]}; }
        ras_cp: coverpoint ras_level {
            bins empty={0}; bins one={1}; bins half={2}; bins full={4};
        }
        stall_cp: coverpoint stalled { bins running={0}; bins stalled={1}; }
        flush_cp: coverpoint flushed { bins normal={0}; bins flush={1}; }
        kind_outcome: cross kind_cp, outcome_cp;
        pht_actual: cross pht_cp, actual_taken;
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
