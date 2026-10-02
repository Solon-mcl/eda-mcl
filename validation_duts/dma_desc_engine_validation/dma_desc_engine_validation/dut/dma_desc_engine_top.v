// Synthesizable reference RTL: descriptor-chain DMA engine with checks.
module dma_desc_engine_top #(
    parameter [31:0] DESC_BASE = 32'h0000_0100,
    parameter CORRUPT_XOR = 1'b0,
    parameter integer CHAIN_LIMIT = 4,
    parameter integer ACK_LATENCY = 2,
    parameter integer RETRY_LIMIT = 3
) (
    input wire clk, input wire rst_n,
    input wire start, input wire fetch_valid, input wire word_we,
    input wire [1:0] word_idx, input wire [31:0] desc_ptr,
    input wire [31:0] wdata, input wire ack, input wire abort,
    output reg [3:0] cov_state, output reg [3:0] cov_result,
    output reg [3:0] cov_fail_class, output reg [1:0] cov_word_index,
    output reg cov_dirty, output reg [1:0] cov_ack_wait_class,
    output reg [1:0] cov_chain_class, output reg [1:0] cov_checks
);
    localparam IDLE=0, FETCH=1, VALIDATE=2, ISSUE=3,
               WAIT_ACK=4, ERROR=5, DONE=6, ABORT=7;

    reg [3:0] state;
    reg [31:0] descriptor [0:3];
    reg [31:0] next_ptr;
    reg [15:0] words_left;
    reg [15:0] ack_count;
    reg [2:0] retries;
    reg [3:0] chain_len;
    reg [1:0] checks;
    reg dirty;
    reg [31:0] base_reg;
    integer i;

    wire [31:0] src = descriptor[0];
    wire [31:0] length = descriptor[1];
    wire [31:0] dst = descriptor[2];
    wire [31:0] flags = descriptor[3];
    wire [1:0] mode = flags[1:0];
    wire aflag = flags[31];

    reg [3:0] fail_class;
    reg [3:0] result;
    reg [1:0] word_index;

    always @* begin
        fail_class = 4'd0;
        if (src == 32'd0) fail_class = 4'd1;
        else if (src < 32'd16) fail_class = 4'd2;
        else if ((src & 32'd3) || (dst & 32'd3)) fail_class = 4'd9;
        else if ((src + length) >= 32'h1_0000) fail_class = 4'd3;
        else if (dst == 32'd0) fail_class = 4'd4;
        else if (dst < 32'd16) fail_class = 4'd5;
        else if ((dst + length) >= 32'h1_0000) fail_class = 4'd6;
        else if (mode != 2'd0 && mode != 2'd1 && mode != 2'd3) fail_class = 4'd10;
        else if (aflag) fail_class = 4'd11;
        else if (length == 32'd0) fail_class = 4'd7;
        else if (length > 32'd4096) fail_class = 4'd8;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; result <= 0; word_index <= 0; dirty <= 1'b0;
            base_reg <= DESC_BASE; words_left <= 0; ack_count <= 0;
            retries <= 0; chain_len <= 0; checks <= 0; next_ptr <= 0;
            for (i = 0; i < 4; i = i + 1) descriptor[i] <= 0;
        end else begin
            result <= 0;
            cov_state <= state;
            cov_result <= result;
            cov_fail_class <= fail_class;
            cov_word_index <= word_index;
            cov_dirty <= dirty;
            cov_chain_class <= chain_len[1:0];
            cov_checks <= checks;

            case (state)
            IDLE: begin
                dirty <= 1'b0;
                if (start) begin
                    base_reg <= base_reg + 32'h40;
                    chain_len <= 0; retries <= 0; checks <= 0;
                    descriptor[0] <= 32'h240;
                    descriptor[1] <= 32'h40;
                    descriptor[2] <= 32'h800;
                    descriptor[3] <= 32'h0;
                    state <= FETCH;
                    result <= 4'd1;
                end
            end
            FETCH: begin
                if (word_we) begin
                    word_index <= word_idx;
                    descriptor[word_idx] <= wdata;
                    dirty <= 1'b1;
                    result <= 4'd8;
                end else if (fetch_valid) begin
                    word_index <= 0;
                    next_ptr <= desc_ptr + 32'h10;
                    state <= VALIDATE;
                end
            end
            VALIDATE: begin
                checks <= checks + 1'b1;
                if (fail_class != 4'd0) begin
                    result <= 4'd3;
                    state <= ERROR;
                end else begin
                    result <= 4'd4;
                    words_left <= length[15:0];
                    ack_count <= ACK_LATENCY;
                    if (mode == 2'd1) state <= WAIT_ACK;
                    else state <= ISSUE;
                end
            end
            ISSUE: begin
                if (abort) begin
                    result <= 4'd2;
                    state <= ABORT;
                end else if (words_left != 0) begin
                    words_left <= words_left - 1'b1;
                    result <= 4'd10;
                    if (words_left == 1) begin
                        chain_len <= chain_len + 1'b1;
                        if (chain_len + 1 >= CHAIN_LIMIT) begin
                            result <= 4'd11;
                            state <= DONE;
                        end else begin
                            result <= 4'd11;
                            state <= IDLE;
                        end
                    end
                end
            end
            WAIT_ACK: begin
                if (abort) begin
                    result <= 4'd2;
                    state <= ABORT;
                end else if (ack) begin
                    chain_len <= chain_len + 1'b1;
                    if (chain_len + 1 >= CHAIN_LIMIT) begin
                        result <= 4'd11;
                        state <= DONE;
                    end else begin
                        result <= 4'd11;
                        state <= IDLE;
                    end
                end else begin
                    ack_count <= (ack_count == 0) ? 0 : ack_count - 1'b1;
                    if (ack_count == 0) begin
                        if (retries + 1 >= RETRY_LIMIT) begin
                            result <= 4'd3;
                            state <= ERROR;
                        end else begin
                            retries <= retries + 1'b1;
                            ack_count <= ACK_LATENCY;
                        end
                    end
                end
            end
            ERROR: begin
                if (next_ptr != 0) begin
                    state <= DONE;
                    result <= 4'd11;
                end
            end
            DONE: begin
                result <= 4'd6;
                state <= IDLE;
            end
            ABORT: begin
                result <= 4'd2;
                state <= IDLE;
            end
            endcase

            cov_ack_wait_class <= (state == WAIT_ACK) ? 2'd1 : 2'd0;
        end
    end
endmodule
