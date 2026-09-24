// Synthesizable reference RTL for the held-out windowed safety watchdog.
module watchdog_safety_top #(
    parameter [7:0] SERVICE_KEY_A = 8'ha5,
    parameter [7:0] SERVICE_KEY_B = 8'h5a,
    parameter [3:0] KEY_GAP = 4'd3,
    parameter [2:0] ESCALATION_LIMIT = 3'd3,
    parameter [3:0] RESET_HOLD = 4'd2
) (
    input wire clk, input wire reset_n,
    input wire reg_we, input wire [1:0] reg_addr, input wire [31:0] reg_wdata,
    input wire tick, input wire service, input wire fault_inject,
    output reg [2:0] cov_state, output reg [2:0] cov_event_state,
    output reg [2:0] cov_result, output wire [1:0] cov_counter_class,
    output wire [1:0] cov_fault_class,
    output reg cov_enabled, output reg cov_cfg_locked,
    output reg cov_safety_locked, output reg cov_key_phase,
    output reg [31:0] cov_last_wdata
);
    localparam DISABLED=0, CLOSED=1, OPEN=2, PRETIMEOUT=3,
               RESET_PENDING=4, SAFETY_LOCKED=5;
    reg [7:0] window_tick, timeout_tick, counter;
    reg [2:0] fault_count;
    reg [3:0] key_left, pending_left;

    assign cov_counter_class = counter == 0 ? 0 :
                               counter < window_tick ? 1 :
                               counter == window_tick ? 2 :
                               counter >= timeout_tick-2 ? 3 : 1;
    assign cov_fault_class = cov_safety_locked ? 3 :
                             (fault_count >= 2 ? 2 : fault_count[1:0]);
    wire service_fault = service && !cov_safety_locked &&
                         ((!cov_key_phase && reg_wdata[7:0] != SERVICE_KEY_A) ||
                          (cov_key_phase &&
                           (key_left == 0 || reg_wdata[7:0] != SERVICE_KEY_B ||
                            (cov_state != OPEN && cov_state != PRETIMEOUT))));

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            cov_state <= DISABLED; cov_event_state <= DISABLED; cov_result <= 0;
            cov_enabled <= 0; cov_cfg_locked <= 0; cov_safety_locked <= 0;
            cov_key_phase <= 0; window_tick <= 3; timeout_tick <= 8;
            counter <= 0; fault_count <= 0; key_left <= 0; pending_left <= 0;
            cov_last_wdata <= 0;
        end else begin
            cov_result <= 0;
            cov_event_state <= cov_state;
            if (reg_we || service) cov_last_wdata <= reg_wdata;
            if (cov_key_phase && !service) begin
                if (key_left <= 1) begin cov_key_phase <= 0; key_left <= 0; end
                else key_left <= key_left-1;
            end

            if (reg_we) begin
                if (cov_state != DISABLED || cov_cfg_locked || cov_safety_locked)
                    cov_result <= 2;
                else begin
                    case (reg_addr)
                        0: begin
                            cov_enabled <= reg_wdata[0];
                            cov_cfg_locked <= cov_cfg_locked | reg_wdata[1];
                            if (reg_wdata[0]) begin cov_state <= CLOSED; counter <= 0; end
                        end
                        1: begin
                            window_tick <= reg_wdata[7:0] < 2 ? 2 : reg_wdata[7:0];
                            if (timeout_tick <= reg_wdata[7:0]+1)
                                timeout_tick <= reg_wdata[7:0]+2;
                        end
                        2: timeout_tick <= reg_wdata[7:0] < window_tick+2 ?
                                          window_tick+2 : reg_wdata[7:0];
                        default: cov_result <= 2;
                    endcase
                    if (reg_addr != 3) cov_result <= 1;
                end
            end

            if (service && !cov_safety_locked) begin
                if (!cov_key_phase) begin
                    if (reg_wdata[7:0] == SERVICE_KEY_A) begin
                        cov_key_phase <= 1; key_left <= KEY_GAP;
                    end else begin
                        cov_result <= 6; fault_count <= fault_count+1;
                    end
                end else begin
                    cov_key_phase <= 0; key_left <= 0;
                    if (key_left == 0 || reg_wdata[7:0] != SERVICE_KEY_B) begin
                        cov_result <= 6; fault_count <= fault_count+1;
                    end else if (cov_state == CLOSED) begin
                        cov_result <= 4; fault_count <= fault_count+1;
                    end else if (cov_state == OPEN || cov_state == PRETIMEOUT) begin
                        cov_result <= cov_state == OPEN ? 3 : 5;
                        counter <= 0; cov_state <= CLOSED;
                    end else begin
                        cov_result <= 6; fault_count <= fault_count+1;
                    end
                end
            end

            if (fault_inject && !cov_safety_locked) fault_count <= fault_count+1;
            if ((fault_inject || service_fault) &&
                fault_count+1 >= ESCALATION_LIMIT) begin
                fault_count <= ESCALATION_LIMIT; cov_safety_locked <= 1;
                cov_enabled <= 0; cov_state <= SAFETY_LOCKED;
                cov_event_state <= SAFETY_LOCKED; cov_result <= 7;
            end

            if (cov_state == RESET_PENDING) begin
                if (pending_left <= 1) begin
                    pending_left <= 0; cov_state <= DISABLED; cov_enabled <= 0;
                end else pending_left <= pending_left-1;
            end else if (tick && cov_enabled && !cov_safety_locked) begin
                counter <= counter+1;
                if (counter+1 >= timeout_tick) begin
                    cov_state <= RESET_PENDING; cov_event_state <= RESET_PENDING;
                    cov_enabled <= 0; pending_left <= RESET_HOLD;
                end else if (counter+1 >= timeout_tick-2) cov_state <= PRETIMEOUT;
                else if (counter+1 >= window_tick) cov_state <= OPEN;
                else cov_state <= CLOSED;
            end
        end
    end
endmodule
