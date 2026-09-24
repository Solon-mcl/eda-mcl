// Synthesizable reference RTL: salted gshare + 2x2 BTB + bounded RAS.
module branch_predictor_top #(
    parameter [2:0] HISTORY_BITS = 3'd4,
    parameter [3:0] INDEX_SALT = 4'd9,
    parameter REPLACEMENT_XOR = 1'b1,
    parameter integer RAS_DEPTH = 4
) (
    input wire clk, input wire rst_n,
    input wire valid, input wire [31:0] pc, input wire [1:0] kind,
    input wire actual_taken, input wire [31:0] actual_target,
    input wire stall, input wire flush,
    output reg [1:0] cov_kind, output reg [1:0] cov_outcome,
    output reg [1:0] cov_btb_result, output reg [1:0] cov_pht_before,
    output reg [5:0] cov_ghr, output reg [3:0] cov_ras_level,
    output reg cov_stalled, output reg cov_flushed
);
    localparam COND=0, CALL=2, RET=3;
    reg [1:0] pht [0:15];
    reg btb_valid [0:1][0:1];
    reg [28:0] btb_tag [0:1][0:1];
    reg [31:0] btb_target [0:1][0:1];
    reg btb_victim [0:1];
    reg [31:0] ras [0:RAS_DEPTH-1];
    integer ras_sp;
    integer i, s, w;

    wire [5:0] history_mask = (6'h3f >> (6-HISTORY_BITS));
    wire [3:0] pht_idx = (pc[5:2] ^ cov_ghr[3:0] ^ INDEX_SALT);
    wire btb_set = pc[2] ^ INDEX_SALT[0];
    wire [28:0] pc_tag = pc[31:3];
    wire hit0 = btb_valid[btb_set][0] && btb_tag[btb_set][0] == pc_tag;
    wire hit1 = btb_valid[btb_set][1] && btb_tag[btb_set][1] == pc_tag;
    wire hit = hit0 || hit1;
    wire hit_way = hit1;
    wire victim_way = !btb_valid[btb_set][0] ? 1'b0 :
                      (!btb_valid[btb_set][1] ? 1'b1 :
                       (btb_victim[btb_set] ^ REPLACEMENT_XOR));
    reg pred_taken;
    reg [31:0] pred_target;

    always @* begin
        if (kind == COND) pred_taken = pht[pht_idx][1];
        else if (kind == RET) pred_taken = ras_sp != 0;
        else pred_taken = 1'b1;
        if (kind == RET)
            pred_target = ras_sp != 0 ? ras[ras_sp-1] : 0;
        else
            pred_target = hit ? btb_target[btb_set][hit_way] : 0;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cov_kind <= 0; cov_outcome <= 0; cov_btb_result <= 0;
            cov_pht_before <= 1; cov_ghr <= 0; cov_ras_level <= 0;
            cov_stalled <= 0; cov_flushed <= 0; ras_sp <= 0;
            for (i=0; i<16; i=i+1) pht[i] <= 1;
            for (s=0; s<2; s=s+1) begin
                btb_victim[s] <= 0;
                for (w=0; w<2; w=w+1) begin
                    btb_valid[s][w] <= 0; btb_tag[s][w] <= 0;
                    btb_target[s][w] <= 0;
                end
            end
            for (i=0; i<RAS_DEPTH; i=i+1) ras[i] <= 0;
        end else begin
            cov_stalled <= stall; cov_flushed <= flush; cov_btb_result <= 0;
            if (flush) cov_ghr <= 0;
            if (valid && !stall) begin
                cov_kind <= kind; cov_pht_before <= pht[pht_idx];
                cov_ras_level <= ras_sp[3:0];
                cov_btb_result <= hit ? 1 : 2;
                if (pred_taken != actual_taken) cov_outcome <= 2;
                else if (!actual_taken) cov_outcome <= 0;
                else if (pred_target == actual_target) cov_outcome <= 1;
                else cov_outcome <= 3;

                if (kind == COND) begin
                    if (actual_taken && pht[pht_idx] != 3) pht[pht_idx] <= pht[pht_idx]+1;
                    else if (!actual_taken && pht[pht_idx] != 0) pht[pht_idx] <= pht[pht_idx]-1;
                end
                if (kind == RET && ras_sp != 0) ras_sp <= ras_sp-1;
                else if (kind == CALL && actual_taken) begin
                    if (ras_sp < RAS_DEPTH) begin ras[ras_sp] <= pc+4; ras_sp <= ras_sp+1; end
                    else begin
                        for (i=0; i<RAS_DEPTH-1; i=i+1) ras[i] <= ras[i+1];
                        ras[RAS_DEPTH-1] <= pc+4;
                    end
                end

                if (actual_taken) begin
                    if (!hit) begin
                        btb_valid[btb_set][victim_way] <= 1;
                        btb_tag[btb_set][victim_way] <= pc_tag;
                        if (btb_valid[btb_set][0] && btb_valid[btb_set][1])
                            cov_btb_result <= 3;
                        btb_victim[btb_set] <= !victim_way;
                    end
                    btb_target[btb_set][hit ? hit_way : victim_way] <= actual_target;
                end
                cov_ghr <= ((cov_ghr << 1) | {5'b0, actual_taken}) & history_mask;
            end
        end
    end
endmodule
