module noc_router_covergroup(
    input logic clk, input logic reset_n,
    input logic [2:0] input_port, output_port,
    input logic vc, input logic [1:0] flit_type, route_mode,
    input logic [2:0] result,
    input logic [1:0] queue_class, credit_class, congestion_class, arb_class,
    input logic seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8,
    input logic seq9, seq10, seq11, seq12, seq13, seq14
);
    covergroup cg @(posedge clk);
        in_cp: coverpoint input_port { bins all[] = {[0:4]}; }
        out_cp: coverpoint output_port { bins all[] = {[0:4]}; }
        vc_cp: coverpoint vc { bins all[] = {[0:1]}; }
        flit_cp: coverpoint flit_type { bins all[] = {[0:3]}; }
        route_cp: coverpoint route_mode { bins all[] = {[0:3]}; }
        result_cp: coverpoint result { bins all[] = {[0:7]}; }
        queue_cp: coverpoint queue_class { bins all[] = {[0:3]}; }
        credit_cp: coverpoint credit_class { bins all[] = {[0:2]}; }
        congestion_cp: coverpoint congestion_class { bins all[] = {[0:2]}; }
        arb_cp: coverpoint arb_class { bins all[] = {[0:2]}; }
        in_out: cross in_cp, out_cp;
        seq1_cp: coverpoint seq1 { bins hit={1}; }
        seq2_cp: coverpoint seq2 { bins hit={1}; }
        seq3_cp: coverpoint seq3 { bins hit={1}; }
        seq4_cp: coverpoint seq4 { bins hit={1}; }
        seq5_cp: coverpoint seq5 { bins hit={1}; }
        seq6_cp: coverpoint seq6 { bins hit={1}; }
        seq7_cp: coverpoint seq7 { bins hit={1}; }
        seq8_cp: coverpoint seq8 { bins hit={1}; }
        seq9_cp: coverpoint seq9 { bins hit={1}; }
        seq10_cp: coverpoint seq10 { bins hit={1}; }
        seq11_cp: coverpoint seq11 { bins hit={1}; }
        seq12_cp: coverpoint seq12 { bins hit={1}; }
        seq13_cp: coverpoint seq13 { bins hit={1}; }
        seq14_cp: coverpoint seq14 { bins hit={1}; }
    endgroup
    cg coverage = new;
endmodule
