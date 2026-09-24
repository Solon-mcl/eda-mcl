// Compact synthesizable reference RTL for cache_ctrl_validation.
// The Python model is the executable local-validation backend.
module cache_ctrl_top #(
    parameter [7:0] MEM_LATENCY = 8'd3,
    parameter REPLACEMENT_XOR = 1'b0,
    parameter [25:0] POISON_TAG = 26'h2a
) (
    input wire clk, input wire rst_n,
    input wire req_valid, input wire [1:0] opcode,
    input wire [31:0] addr, input wire [31:0] wdata,
    input wire mem_ready,
    output wire req_ready, output reg resp_valid, output reg [31:0] resp_rdata,
    output reg [2:0] cov_state, output reg [1:0] cov_op,
    output reg [1:0] cov_set, output reg cov_way,
    output reg [3:0] cov_result, output reg cov_stalled,
    output reg [3:0] cov_offset
);
    localparam IDLE=0, LOOKUP=1, WRITEBACK=2, REFILL=3, RESPOND=4, FLUSH=5;
    localparam OP_WRITE=1, OP_INVALIDATE=2, OP_FLUSH=3;
    reg valid [0:3][0:1];
    reg dirty [0:3][0:1];
    reg [25:0] tags [0:3][0:1];
    reg [31:0] data [0:3][0:1];
    reg lru [0:3];
    reg [27:0] saved_line;
    reg [31:0] saved_data;
    reg [7:0] wait_count;
    reg [2:0] flush_slot;
    integer s, w;
    wire [1:0] addr_set = saved_line[1:0];
    wire [25:0] addr_tag = saved_line[27:2];
    wire hit0 = valid[addr_set][0] && tags[addr_set][0] == addr_tag;
    wire hit1 = valid[addr_set][1] && tags[addr_set][1] == addr_tag;
    wire victim = !valid[addr_set][0] ? 1'b0 :
                  (!valid[addr_set][1] ? 1'b1 : (lru[addr_set] ^ REPLACEMENT_XOR));
    assign req_ready = cov_state == IDLE;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cov_state <= IDLE; cov_op <= 0; cov_set <= 0; cov_way <= 0;
            cov_result <= 0; cov_stalled <= 0; resp_valid <= 0; resp_rdata <= 0;
            cov_offset <= 0; wait_count <= 0; flush_slot <= 0;
            saved_line <= 0; saved_data <= 0;
            for (s=0; s<4; s=s+1) begin
                lru[s] <= 0;
                for (w=0; w<2; w=w+1) begin
                    valid[s][w] <= 0; dirty[s][w] <= 0; tags[s][w] <= 0; data[s][w] <= 0;
                end
            end
        end else begin
            resp_valid <= 0; cov_stalled <= 0;
            case (cov_state)
                IDLE: if (req_valid) begin
                    cov_op <= opcode; saved_line <= addr[31:4]; saved_data <= wdata;
                    cov_offset <= addr[3:0];
                    cov_result <= 0;
                    if (opcode == OP_FLUSH) begin flush_slot <= 0; cov_state <= FLUSH; end
                    else cov_state <= LOOKUP;
                end
                LOOKUP: begin
                    cov_set <= addr_set;
                    if (hit0 || hit1) begin
                        cov_way <= hit1; lru[addr_set] <= !hit1;
                        if (cov_op == OP_INVALIDATE) begin
                            valid[addr_set][hit1] <= 0; dirty[addr_set][hit1] <= 0;
                            cov_result <= 5;
                        end else begin
                            cov_result <= 1;
                            resp_rdata <= data[addr_set][hit1];
                            if (cov_op == OP_WRITE) begin
                                dirty[addr_set][hit1] <= 1;
                                data[addr_set][hit1] <= saved_data;
                            end
                        end
                        cov_state <= RESPOND;
                    end else if (cov_op == OP_INVALIDATE) begin
                        cov_result <= 2; cov_state <= RESPOND;
                    end else begin
                        cov_way <= victim;
                        if (valid[addr_set][victim] && dirty[addr_set][victim]) begin
                            cov_result <= 4; cov_state <= WRITEBACK;
                        end else begin
                            cov_result <= valid[addr_set][victim] ? 3 : 2;
                            cov_state <= REFILL;
                        end
                        wait_count <= MEM_LATENCY;
                    end
                end
                WRITEBACK: if (!mem_ready) cov_stalled <= 1; else if (wait_count > 1)
                    wait_count <= wait_count-1; else begin
                        dirty[addr_set][cov_way] <= 0; wait_count <= MEM_LATENCY;
                        cov_state <= REFILL;
                    end
                REFILL: if (!mem_ready) cov_stalled <= 1; else if (wait_count > 1)
                    wait_count <= wait_count-1; else begin
                        valid[addr_set][cov_way] <= 1; tags[addr_set][cov_way] <= addr_tag;
                        dirty[addr_set][cov_way] <= cov_op == OP_WRITE;
                        data[addr_set][cov_way] <= cov_op == OP_WRITE ? saved_data : 0;
                        resp_rdata <= cov_op == OP_WRITE ? saved_data : 0;
                        lru[addr_set] <= !cov_way;
                        if (addr_tag == POISON_TAG) cov_result <= 7;
                        cov_state <= RESPOND;
                    end
                RESPOND: begin resp_valid <= 1; cov_state <= IDLE; end
                FLUSH: begin
                    cov_set <= flush_slot[2:1]; cov_way <= flush_slot[0];
                    if (valid[flush_slot[2:1]][flush_slot[0]] &&
                        dirty[flush_slot[2:1]][flush_slot[0]] && !mem_ready)
                        cov_stalled <= 1;
                    else begin
                        if (dirty[flush_slot[2:1]][flush_slot[0]]) cov_result <= 6;
                        valid[flush_slot[2:1]][flush_slot[0]] <= 0;
                        dirty[flush_slot[2:1]][flush_slot[0]] <= 0;
                        if (flush_slot == 7) cov_state <= RESPOND;
                        else flush_slot <= flush_slot + 1;
                    end
                end
            endcase
        end
    end
endmodule
