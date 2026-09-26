// Readable reference RTL for tlb_mmu_validation.  Like the other held-out
// packages this file is a specification artifact: no Verilator binary ships
// with the package, so local_sim.py stays the authoritative model.
//
// Salted ASID-tagged TLB plus a synthetic page-table walk.  Coverage outputs
// mirror the signals consumed by dut/coverage_meta.json.
//
// Reference simplification: the Python model keeps an unbounded set of pages
// whose accessed bit is already set; the RTL approximates it with a 16-entry
// recent-walk bitmap indexed by vpn[3:0].
module tlb_mmu_top #(
    parameter integer TLB_SETS = 4,
    parameter integer TLB_WAYS = 2,
    parameter REPLACE_XOR = 1'b1,
    parameter [1:0] ASID_BITS = 2'd2,
    parameter [3:0] WALK_SALT = 4'd5,
    parameter LOCK_ENABLE = 1'b1
) (
    input wire clk, input wire rst_n,
    input wire valid, input wire [31:0] vpn, input wire [1:0] op,
    input wire priv, input wire [1:0] asid, input wire global_req,
    input wire fence, input wire flush, input wire satp, input wire stall,
    output reg [1:0] cov_op, output reg cov_priv,
    output reg [1:0] cov_tlb_result, output reg [1:0] cov_walk_result,
    output reg [1:0] cov_hazard, output reg [1:0] cov_result,
    output reg cov_dirty_before, output reg cov_accessed_clear,
    output reg [1:0] cov_fence_class, output reg cov_stalled,
    output reg cov_entry_global
);
    localparam [1:0] OP_LOAD = 2'd0, OP_STORE = 2'd1, OP_FETCH = 2'd2,
                     OP_ATOMIC = 2'd3;
    localparam [1:0] TLB_HIT = 2'd0, TLB_REFILL = 2'd1, TLB_REPLACE = 2'd2,
                     TLB_NOFILL = 2'd3;
    localparam [1:0] WALK_IDLE = 2'd0, WALK_OK = 2'd1, WALK_PTE_BAD = 2'd2,
                     WALK_LOCKED = 2'd3;
    localparam [1:0] HZ_NONE = 2'd0, HZ_PAGE = 2'd1, HZ_PROT = 2'd2,
                     HZ_ASID = 2'd3;

    localparam integer SET_BITS = (TLB_SETS == 8) ? 3 : 2;
    localparam [3:0] SET_MASK = (TLB_SETS == 8) ? 4'h7 : 4'h3;

    reg [31:0] tlb_vpn    [0:TLB_SETS-1][0:TLB_WAYS-1];
    reg [1:0]  tlb_asid   [0:TLB_SETS-1][0:TLB_WAYS-1];
    reg        tlb_valid  [0:TLB_SETS-1][0:TLB_WAYS-1];
    reg        tlb_global [0:TLB_SETS-1][0:TLB_WAYS-1];
    reg        pte_dirty  [0:TLB_SETS-1][0:TLB_WAYS-1];
    reg        pte_access [0:TLB_SETS-1][0:TLB_WAYS-1];
    reg        victim_way [0:TLB_SETS-1];
    reg        satp_gen;
    reg        walk_seen  [0:15];

    integer i, j;

    // ---- salted indexing and the synthetic walk ----------------------
    wire [1:0]  asid_mask = (ASID_BITS == 2'd2) ? 2'b11 : 2'b01;
    wire [3:0]  set_index = (vpn[3:0] ^ WALK_SALT) & SET_MASK;
    wire [31:0] seed = vpn ^ {28'd0, WALK_SALT} ^ (satp_gen ? 32'h5D : 32'd0);
    wire        pte_valid  = (seed % 7) != 0;
    wire        pte_locked = LOCK_ENABLE && seed[3];
    wire        pte_w = seed[2];
    wire        pte_x = seed[1];
    wire        pte_u = seed[4];
    wire [19:0] pte_pfn = 20'(((seed * 32'h9E3779B1) ^
                               (32'd0 + WALK_SALT) << 7));
    wire [3:0]  walk_index = vpn[3:0];
    wire        pte_accessed = walk_seen[walk_index];

    // ---- lookup -------------------------------------------------------
    reg        hit;
    reg [1:0]  hit_way;
    reg        alias_seen;
    reg        hit_global;
    reg        free_found;
    reg [1:0]  free_way;
    reg        victim_way_eff;

    always @* begin
        hit = 1'b0; alias_seen = 1'b0; hit_way = 2'd0; hit_global = 1'b0;
        free_found = 1'b0; free_way = 2'd0;
        for (i = 0; i < TLB_WAYS; i = i + 1) begin
            if (tlb_valid[set_index][i] && tlb_vpn[set_index][i] == vpn) begin
                if (tlb_global[set_index][i] ||
                    ((tlb_asid[set_index][i] & asid_mask) == (asid & asid_mask))) begin
                    if (!hit) begin
                        hit = 1'b1;
                        hit_way = i[1:0];
                        hit_global = tlb_global[set_index][i];
                    end
                end else begin
                    alias_seen = 1'b1;
                end
            end
            if (!tlb_valid[set_index][i] && !free_found) begin
                free_found = 1'b1;
                free_way = i[1:0];
            end
        end
        victim_way_eff = (free_found ? free_way
                                      : victim_way[set_index] ^ REPLACE_XOR);
    end

    wire        walk_ok = !hit && pte_valid && !pte_locked;
    wire        page_fault = !hit && !pte_valid;
    wire        use_entry_dirty = hit ? pte_dirty[set_index][hit_way]
                                      : 1'b0;
    wire        use_entry_access = hit ? pte_access[set_index][hit_way]
                                       : pte_accessed;
    wire        perm_w = hit ? 1'b1 : pte_w;
    wire        perm_x = hit ? 1'b1 : pte_x;
    wire        perm_u = hit ? 1'b1 : pte_u;
    wire        prot_fault = !page_fault &&
                             ((op == OP_STORE  && !perm_w) ||
                              (op == OP_ATOMIC && !perm_w) ||
                              (op == OP_FETCH  && !perm_x) ||
                              (!priv && !perm_u));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            satp_gen <= 1'b0;
            cov_op <= 2'd0; cov_priv <= 1'b0; cov_tlb_result <= TLB_HIT;
            cov_walk_result <= WALK_IDLE; cov_hazard <= HZ_NONE;
            cov_result <= 2'd0; cov_dirty_before <= 1'b0;
            cov_accessed_clear <= 1'b0; cov_fence_class <= 2'd0;
            cov_stalled <= 1'b0; cov_entry_global <= 1'b0;
            for (i = 0; i < 16; i = i + 1) walk_seen[i] <= 1'b0;
            for (i = 0; i < TLB_SETS; i = i + 1) begin
                victim_way[i] <= 1'b0;
                for (j = 0; j < TLB_WAYS; j = j + 1) begin
                    tlb_valid[i][j] <= 1'b0; tlb_global[i][j] <= 1'b0;
                    tlb_vpn[i][j] <= 32'd0; tlb_asid[i][j] <= 2'd0;
                    pte_dirty[i][j] <= 1'b0; pte_access[i][j] <= 1'b0;
                end
            end
        end else begin
            cov_fence_class <= 2'd0;
            cov_stalled <= stall;
            cov_op <= op;
            cov_priv <= priv;

            if (flush || satp) begin
                cov_fence_class <= 2'd2;
                if (satp) begin
                    satp_gen <= ~satp_gen;
                    for (i = 0; i < 16; i = i + 1) walk_seen[i] <= 1'b0;
                end
                for (i = 0; i < TLB_SETS; i = i + 1)
                    for (j = 0; j < TLB_WAYS; j = j + 1)
                        tlb_valid[i][j] <= 1'b0;
            end else if (fence) begin
                cov_fence_class <= 2'd1;
                for (i = 0; i < TLB_SETS; i = i + 1)
                    for (j = 0; j < TLB_WAYS; j = j + 1)
                        if (tlb_valid[i][j] && !tlb_global[i][j])
                            tlb_valid[i][j] <= 1'b0;
            end

            if (valid && !stall) begin
                cov_dirty_before <= use_entry_dirty;
                cov_accessed_clear <= ~use_entry_access;
                cov_entry_global <= hit && hit_global;
                cov_tlb_result <= hit ? TLB_HIT :
                                  (walk_ok ? (free_found ? TLB_REFILL
                                                         : TLB_REPLACE)
                                           : TLB_NOFILL);
                cov_walk_result <= hit ? WALK_IDLE :
                                   (!pte_valid ? WALK_PTE_BAD :
                                    (pte_locked ? WALK_LOCKED : WALK_OK));
                cov_hazard <= page_fault ? HZ_PAGE :
                              (prot_fault ? HZ_PROT :
                               (alias_seen ? HZ_ASID : HZ_NONE));
                cov_result <= page_fault ? 2'd2 :
                              (prot_fault ? 2'd3 : (hit ? 2'd0 : 2'd1));

                if (!page_fault && !prot_fault) walk_seen[walk_index] <= 1'b1;

                if (hit) begin
                    pte_access[set_index][hit_way] <= 1'b1;
                    if (op == OP_STORE || op == OP_ATOMIC)
                        pte_dirty[set_index][hit_way] <= 1'b1;
                end else if (walk_ok) begin
                    tlb_valid[set_index][victim_way_eff] <= 1'b1;
                    tlb_global[set_index][victim_way_eff] <= global_req;
                    tlb_vpn[set_index][victim_way_eff] <= vpn;
                    tlb_asid[set_index][victim_way_eff] <= asid;
                    pte_access[set_index][victim_way_eff] <= 1'b1;
                    pte_dirty[set_index][victim_way_eff] <=
                        (op == OP_STORE || op == OP_ATOMIC);
                    if (!free_found) victim_way[set_index] <= ~victim_way_eff;
                end
            end
        end
    end
endmodule
