// Functional-coverage reference. coverage_meta.json is the executable mirror.
module cache_ctrl_covergroup(
    input logic clk, input logic rst_n,
    input logic [2:0] state, input logic [1:0] op,
    input logic [1:0] set_idx, input logic way,
    input logic [3:0] result, input logic [3:0] offset,
    input logic [1:0] data_class,
    input logic [3:0] valid_count, input logic [3:0] dirty_count,
    input logic stalled,
    input logic seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8
);
    covergroup cg @(posedge clk);
        state_cp: coverpoint state { bins all[] = {[0:5]}; }
        op_cp: coverpoint op { bins all[] = {[0:3]}; }
        set_cp: coverpoint set_idx { bins all[] = {[0:3]}; }
        way_cp: coverpoint way { bins all[] = {[0:1]}; }
        result_cp: coverpoint result { bins all[] = {[0:7]}; }
        offset_cp: coverpoint offset {
            bins first={0}; bins word1={4}; bins word2={8}; bins last={15};
        }
        data_cp: coverpoint data_class { bins all[] = {[0:3]}; }
        valid_cp: coverpoint valid_count {
            bins empty={0}; bins one={1}; bins half={4}; bins full={8};
        }
        dirty_cp: coverpoint dirty_count {
            bins none={0}; bins one={1}; bins half={4}; bins full={8};
        }
        stall_cp: coverpoint stalled { bins flowing={0}; bins blocked={1}; }
        set_way: cross set_cp, way_cp;
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
