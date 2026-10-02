// Synthesizable transaction-level reference RTL for ecc_memory_validation.
module ecc_memory_top #(
    parameter [4:0] SYNDROME_XOR = 5'd5,
    parameter [2:0] SCRUB_STRIDE = 3'd3,
    parameter [2:0] CORRECTION_LATENCY = 3'd3,
    parameter ZERO_ON_DBE = 1'b1,
    parameter [2:0] POISON_WORD = 3'd6
) (
    input wire clk, input wire reset_n,
    input wire write, input wire read, input wire [2:0] address,
    input wire [31:0] write_data,
    input wire inject, input wire [5:0] bit_a, input wire [5:0] bit_b,
    input wire scrub, input wire stall,
    output reg [31:0] read_data,
    output reg [2:0] cov_state, output reg [3:0] cov_result,
    output reg [2:0] cov_op, output reg [1:0] cov_error_class,
    output reg [2:0] cov_address, output reg [1:0] cov_syndrome_class,
    output reg [1:0] cov_data_class, output reg [1:0] cov_scrub_class,
    output reg cov_stalled
);
    localparam IDLE=0, ST_WRITE=1, ST_READ=2, ST_CORRECT=3,
               ST_UNCORRECTABLE=4, ST_SCRUB=5;
    reg [31:0] memory [0:7];
    reg [1:0] error_count [0:7];
    reg [2:0] scrub_cursor, pending_addr;
    reg [2:0] correction_left;
    reg correction_pending;
    integer i;
    function automatic [2:0] scrub_address(input [2:0] cursor);
        reg [2:0] scaled;
        begin
            case (SCRUB_STRIDE)
                3'd1: scaled = cursor;
                3'd3: scaled = (cursor << 1) + cursor;
                3'd5: scaled = (cursor << 2) + cursor;
                default: scaled = -cursor;
            endcase
            scrub_address = scaled + SYNDROME_XOR[2:0];
        end
    endfunction
    function automatic [1:0] syndrome2(input [1:0] a, input [1:0] b);
        reg [1:0] av, bv;
        begin
            av = a ^ SYNDROME_XOR[1:0];
            bv = b ^ SYNDROME_XOR[1:0];
            syndrome2 = av + bv + SYNDROME_XOR[1:0];
        end
    endfunction
    wire [2:0] scrub_addr = scrub_address(scrub_cursor);

    always @* begin
        if (memory[cov_address] == 0) cov_data_class = 0;
        else if (memory[cov_address] == 32'hffffffff) cov_data_class = 1;
        else if (memory[cov_address] == 32'haaaaaaaa) cov_data_class = 2;
        else cov_data_class = 3;
        if (scrub_cursor == 0) cov_scrub_class = 0;
        else if (scrub_cursor == 7) cov_scrub_class = 2;
        else cov_scrub_class = 1;
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            read_data <= 0; cov_state <= IDLE; cov_result <= 0; cov_op <= 0;
            cov_error_class <= 0; cov_address <= 0; cov_syndrome_class <= 0;
            cov_stalled <= 0; scrub_cursor <= 0; pending_addr <= 0;
            correction_left <= 0; correction_pending <= 0;
            for (i=0; i<8; i=i+1) begin memory[i] <= 0; error_count[i] <= 0; end
        end else begin
            cov_state <= IDLE; cov_result <= 0; cov_op <= 0; cov_stalled <= 0;
            if (correction_pending) begin
                cov_state <= ST_CORRECT; cov_op <= 2; cov_address <= pending_addr;
                cov_error_class <= error_count[pending_addr];
                if (stall) cov_stalled <= 1;
                else if (correction_left > 1) correction_left <= correction_left-1;
                else begin
                    error_count[pending_addr] <= 0; cov_error_class <= 0;
                    correction_pending <= 0; correction_left <= 0; cov_result <= 4;
                end
            end else if (write) begin
                cov_state <= ST_WRITE; cov_op <= 1; cov_result <= 1;
                cov_address <= address; memory[address] <= write_data;
                error_count[address] <= 0; cov_error_class <= 0;
            end else if (inject) begin
                cov_op <= 3; cov_address <= address;
                cov_syndrome_class <= syndrome2(bit_a[1:0], bit_b[1:0]);
                if (bit_a == bit_b && error_count[address] == 0) begin
                    error_count[address] <= 1; cov_error_class <= 1; cov_result <= 6;
                end else begin
                    error_count[address] <= 2;
                    cov_error_class <= address == POISON_WORD ? 3 : 2;
                    cov_result <= 7;
                end
            end else if (read) begin
                cov_op <= 2; cov_address <= address;
                cov_error_class <= (address == POISON_WORD && error_count[address] >= 2) ?
                                   3 : error_count[address];
                if (error_count[address] == 0) begin
                    cov_state <= ST_READ; cov_result <= 2; read_data <= memory[address];
                end else if (error_count[address] == 1) begin
                    cov_state <= ST_CORRECT; cov_result <= 3; read_data <= memory[address];
                    correction_pending <= 1; pending_addr <= address;
                    correction_left <= CORRECTION_LATENCY;
                end else begin
                    cov_state <= ST_UNCORRECTABLE; cov_result <= 5;
                    read_data <= ZERO_ON_DBE ? 0 : memory[address];
                end
            end else if (scrub) begin
                cov_state <= ST_SCRUB; cov_op <= 4; cov_address <= scrub_addr;
                cov_error_class <= (scrub_addr == POISON_WORD && error_count[scrub_addr] >= 2) ?
                                   3 : error_count[scrub_addr];
                if (stall) cov_stalled <= 1;
                else begin
                    if (error_count[scrub_addr] == 0) cov_result <= 8;
                    else if (error_count[scrub_addr] == 1) begin
                        cov_result <= 9; error_count[scrub_addr] <= 0;
                    end else cov_result <= 10;
                    scrub_cursor <= scrub_cursor+1;
                end
            end
        end
    end
endmodule
