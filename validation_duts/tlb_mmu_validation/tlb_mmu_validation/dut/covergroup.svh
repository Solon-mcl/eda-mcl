// SystemVerilog functional-coverage reference; metadata is the local mirror.
module tlb_mmu_covergroup(
    input logic clk, input logic rst_n,
    input logic [1:0] op, priv, tlb_result, walk_result, hazard, result,
    input logic dirty_before, accessed_clear, fence_class, stalled,
    input logic entry_global,
    input logic seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8
);
    covergroup cg @(posedge clk);
        op_cp: coverpoint op { bins all[] = {[0:3]}; }
        priv_cp: coverpoint priv { bins all[] = {[0:1]}; }
        tlb_cp: coverpoint tlb_result { bins all[] = {[0:3]}; }
        walk_cp: coverpoint walk_result { bins all[] = {[0:3]}; }
        hazard_cp: coverpoint hazard { bins all[] = {[0:3]}; }
        result_cp: coverpoint result { bins all[] = {[0:3]}; }
        dirty_cp: coverpoint dirty_before { bins clean={0}; bins dirty={1}; }
        accessed_cp: coverpoint accessed_clear {
            bins already_set={0}; bins cold_clear={1};
        }
        fence_cp: coverpoint fence_class {
            bins none={0}; bins asid_fence={1}; bins full_flush={2};
        }
        stall_cp: coverpoint stalled { bins running={0}; bins stalled={1}; }
        scope_cp: coverpoint entry_global { bins local={0}; bins global={1}; }
        op_result: cross op_cp, result_cp;
        priv_hazard: cross priv_cp, hazard_cp;
        op_dirty: cross op_cp, dirty_cp;
        lookup_walk: cross tlb_cp, walk_cp;
        fence_lookup: cross fence_cp, tlb_cp;
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
