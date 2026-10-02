module ecc_memory_covergroup(
    input logic clk, input logic reset_n,
    input logic [2:0] state, op, address,
    input logic [3:0] result,
    input logic [1:0] error_class, syndrome_class, data_class, scrub_class,
    input logic stalled,
    input logic seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8
);
    covergroup cg @(posedge clk);
        state_cp: coverpoint state { bins all[] = {[0:5]}; }
        result_cp: coverpoint result { bins all[] = {[0:10]}; }
        error_cp: coverpoint error_class { bins all[] = {[0:3]}; }
        address_cp: coverpoint address { bins all[] = {[0:7]}; }
        syndrome_cp: coverpoint syndrome_class { bins all[] = {[0:3]}; }
        data_cp: coverpoint data_class { bins all[] = {[0:3]}; }
        scrub_cp: coverpoint scrub_class { bins first={0}; bins middle={1}; bins last={2}; }
        stall_cp: coverpoint stalled { bins run={0}; bins wait_state={1}; }
        op_result: cross op, result;
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
