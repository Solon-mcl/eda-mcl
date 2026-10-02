// SystemVerilog functional-coverage reference; metadata is the local mirror.
module dma_desc_engine_covergroup(
    input logic clk, input logic rst_n,
    input logic [3:0] state, event_state,
    input logic [3:0] result,
    input logic [3:0] fail_class,
    input logic [1:0] word_index,
    input logic dirty,
    input logic [1:0] ack_wait_class, chain_class, checks,
    input logic seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8
);
    covergroup cg @(posedge clk);
        state_cp: coverpoint state { bins all[] = {[0:7]}; }
        event_cp: coverpoint event_state { bins all[] = {[0:7]}; }
        result_cp: coverpoint result {
            bins wanted[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11};
        }
        fail_cp: coverpoint fail_class { bins all[] = {[0:12]}; }
        word_cp: coverpoint word_index { bins all[] = {[0:3]}; }
        dirty_cp: coverpoint dirty { bins clean={0}; bins dirty={1}; }
        ack_cp: coverpoint ack_wait_class { bins all[] = {[0:2]}; }
        chain_cp: coverpoint chain_class { bins all[] = {[0:3]}; }
        check_cp: coverpoint checks { bins all[] = {[0:3]}; }
        // Only the reaches of these crosses that the engine can actually
        // produce are binned; the metadata mirrors the same list.
        state_result: cross event_cp, result_cp {
            bins idle_start = binsof(event_cp) intersect {0} && binsof(result_cp) intersect {1};
            bins fetch_word_write = binsof(event_cp) intersect {1} && binsof(result_cp) intersect {8};
            bins fetch_miss = binsof(event_cp) intersect {1} && binsof(result_cp) intersect {5};
            bins validate_pass = binsof(event_cp) intersect {2} && binsof(result_cp) intersect {4};
            bins validate_fail = binsof(event_cp) intersect {2} && binsof(result_cp) intersect {3};
            bins issue_beat = binsof(event_cp) intersect {3} && binsof(result_cp) intersect {10};
            bins issue_abort = binsof(event_cp) intersect {3} && binsof(result_cp) intersect {2};
            bins wait_abort = binsof(event_cp) intersect {4} && binsof(result_cp) intersect {2};
            bins wait_retire = binsof(event_cp) intersect {4} && binsof(result_cp) intersect {11};
            bins error_clear = binsof(event_cp) intersect {5} && binsof(result_cp) intersect {0};
            bins done_clear = binsof(event_cp) intersect {6} && binsof(result_cp) intersect {6};
        }
        state_fail: cross event_cp, fail_cp {
            bins fetch_miss = binsof(event_cp) intersect {1} && binsof(fail_cp) intersect {0};
            bins wait_retry = binsof(event_cp) intersect {4} && binsof(fail_cp) intersect {12};
            bins validate_low_src = binsof(event_cp) intersect {2} && binsof(fail_cp) intersect {2};
            bins validate_unaligned = binsof(event_cp) intersect {2} && binsof(fail_cp) intersect {9};
            bins validate_mode = binsof(event_cp) intersect {2} && binsof(fail_cp) intersect {10};
            bins validate_flag = binsof(event_cp) intersect {2} && binsof(fail_cp) intersect {11};
            bins validate_zero_len = binsof(event_cp) intersect {2} && binsof(fail_cp) intersect {7};
            bins validate_zero_src = binsof(event_cp) intersect {2} && binsof(fail_cp) intersect {1};
            bins validate_high_dst = binsof(event_cp) intersect {2} && binsof(fail_cp) intersect {6};
        }
        chain_state: cross chain_cp, state_cp {
            bins none_idle = binsof(chain_cp) intersect {0} && binsof(state_cp) intersect {0};
            bins one_idle = binsof(chain_cp) intersect {1} && binsof(state_cp) intersect {0};
            bins mid_idle = binsof(chain_cp) intersect {2} && binsof(state_cp) intersect {0};
            bins limit_idle = binsof(chain_cp) intersect {3} && binsof(state_cp) intersect {0};
            bins none_fetch = binsof(chain_cp) intersect {0} && binsof(state_cp) intersect {1};
            bins mid_wait = binsof(chain_cp) intersect {2} && binsof(state_cp) intersect {4};
            bins limit_done = binsof(chain_cp) intersect {3} && binsof(state_cp) intersect {6};
        }
        word_dirty: cross word_cp, dirty_cp {
            bins w0_dirty = binsof(word_cp) intersect {0} && binsof(dirty_cp) intersect {1};
            bins w1_dirty = binsof(word_cp) intersect {1} && binsof(dirty_cp) intersect {1};
            bins w0_clean = binsof(word_cp) intersect {0} && binsof(dirty_cp) intersect {0};
        }
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
