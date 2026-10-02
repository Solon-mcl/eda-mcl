// Synthesizable transaction-level reference RTL for noc_router_validation.
module noc_router_top #(
    parameter XY_ORDER = 1'b1,
    parameter [2:0] PRIORITY_SEED = 3'd2,
    parameter ESCAPE_VC = 1'b1,
    parameter [2:0] CONGESTION_THRESHOLD = 3'd1,
    parameter [2:0] BUFFER_DEPTH = 3'd3
) (
    input wire clk, input wire reset_n,
    input wire valid, input wire [2:0] in_port,
    input wire [1:0] dest_x, input wire [1:0] dest_y,
    input wire vc, input wire [1:0] flit_type, input wire [31:0] payload,
    input wire [4:0] credit_mask, input wire [4:0] congestion_mask,
    input wire router_stall,
    output reg [2:0] cov_input_port, output reg [2:0] cov_output_port,
    output reg cov_vc, output reg [1:0] cov_flit_type,
    output reg [1:0] cov_route_mode, output reg [2:0] cov_result,
    output reg [1:0] cov_queue_class, output reg [1:0] cov_arb_class,
    output reg [31:0] cov_payload
);
    localparam LOCAL=0, NORTH=1, EAST=2, SOUTH=3, WEST=4;
    reg [2:0] count [0:4];
    reg [1:0] qx [0:4], qy [0:4], qtype [0:4];
    reg qvc [0:4];
    reg [31:0] qpayload [0:4];
    integer i, k, port, chosen, contenders;
    reg [2:0] chosen_out;
    reg [1:0] chosen_mode;

    function automatic [2:0] bit_count5(input [4:0] value);
        integer n;
        begin
            bit_count5 = 0;
            for (n=0; n<5; n=n+1)
                bit_count5 = bit_count5 + {2'b0, value[n]};
        end
    endfunction
    function automatic [2:0] route_port(
        input [1:0] x, input [1:0] y, input route_vc,
        input [4:0] congestion
    );
        reg [2:0] primary, alternate;
        reg x_needed, y_needed;
        begin
            x_needed = x != 1; y_needed = y != 1;
            if (!x_needed && !y_needed) route_port = LOCAL;
            else begin
                if (XY_ORDER) begin
                    primary = x_needed ? (x > 1 ? EAST : WEST) : (y > 1 ? NORTH : SOUTH);
                    alternate = y_needed ? (y > 1 ? NORTH : SOUTH) : primary;
                end else begin
                    primary = y_needed ? (y > 1 ? NORTH : SOUTH) : (x > 1 ? EAST : WEST);
                    alternate = x_needed ? (x > 1 ? EAST : WEST) : primary;
                end
                if (route_vc != ESCAPE_VC && congestion[primary] &&
                    bit_count5(congestion) >= CONGESTION_THRESHOLD &&
                    x_needed && y_needed)
                    route_port = alternate;
                else route_port = primary;
            end
        end
    endfunction
    function automatic [1:0] route_kind(
        input [1:0] x, input [1:0] y, input route_vc,
        input [4:0] congestion
    );
        reg [2:0] deterministic, selected;
        begin
            if (x == 1 && y == 1) route_kind = 3;
            else if (route_vc == ESCAPE_VC) route_kind = 2;
            else begin
                deterministic = route_port(x, y, ESCAPE_VC, congestion);
                selected = route_port(x, y, route_vc, congestion);
                route_kind = selected != deterministic ? 1 : 0;
            end
        end
    endfunction

    always @* begin
        chosen = -1; chosen_out = LOCAL; chosen_mode = 0; contenders = 0;
        for (k=0; k<5; k=k+1) begin
            port = ({29'b0, PRIORITY_SEED} + k) % 5;
            if (count[port] != 0 && chosen < 0) begin
                chosen = port;
                chosen_out = route_port(qx[port], qy[port], qvc[port], congestion_mask);
                chosen_mode = route_kind(qx[port], qy[port], qvc[port], congestion_mask);
            end
        end
        if (chosen >= 0)
            for (k=0; k<5; k=k+1)
                if (count[k] != 0 &&
                    route_port(qx[k], qy[k], qvc[k], congestion_mask) == chosen_out)
                    contenders = contenders + 1;
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            cov_input_port <= 0; cov_output_port <= 0; cov_vc <= 0;
            cov_flit_type <= 3; cov_route_mode <= 0; cov_result <= 0;
            cov_queue_class <= 0; cov_arb_class <= 0; cov_payload <= 0;
            for (i=0; i<5; i=i+1) begin
                count[i] <= 0; qx[i] <= 1; qy[i] <= 1; qvc[i] <= 0;
                qtype[i] <= 3; qpayload[i] <= 0;
            end
        end else begin
            cov_result <= 0;
            cov_arb_class <= contenders == 0 ? 0 : (contenders == 1 ? 1 : 2);
            if (router_stall) cov_result <= 7;
            else if (chosen >= 0) begin
                cov_input_port <= chosen[2:0]; cov_output_port <= chosen_out;
                cov_vc <= qvc[chosen]; cov_flit_type <= qtype[chosen];
                cov_route_mode <= chosen_mode; cov_payload <= qpayload[chosen];
                if (!credit_mask[chosen_out]) cov_result <= 4;
                else begin
                    count[chosen] <= count[chosen]-1;
                    cov_result <= chosen_out == LOCAL ? 3 : (contenders > 1 ? 6 : 2);
                end
            end
            if (valid) begin
                cov_input_port <= in_port; cov_vc <= vc; cov_flit_type <= flit_type;
                cov_payload <= payload;
                if (count[in_port] >= BUFFER_DEPTH) cov_result <= 5;
                else begin
                    count[in_port] <= count[in_port]+1;
                    qx[in_port] <= dest_x; qy[in_port] <= dest_y;
                    qvc[in_port] <= vc; qtype[in_port] <= flit_type;
                    qpayload[in_port] <= payload; cov_result <= 1;
                end
            end
            if (count[cov_input_port] == 0) cov_queue_class <= 0;
            else if (count[cov_input_port] == 1) cov_queue_class <= 1;
            else if (count[cov_input_port] >= BUFFER_DEPTH) cov_queue_class <= 3;
            else cov_queue_class <= 2;
        end
    end
endmodule
