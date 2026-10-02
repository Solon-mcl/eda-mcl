#!/usr/bin/env python3
"""Cycle-level buffered five-port NoC router model."""

LOCAL, NORTH, EAST, SOUTH, WEST = range(5)


class NocRouterValidation:
    def __init__(self, xy_order=1, priority_seed=2, escape_vc=1,
                 congestion_threshold=1, buffer_depth=3):
        self.xy_order = bool(xy_order)
        self.priority_seed = int(priority_seed) % 5
        self.escape_vc = int(escape_vc) & 1
        self.congestion_threshold = min(3, max(1, int(congestion_threshold)))
        self.buffer_depth = min(5, max(2, int(buffer_depth)))
        self.reset()

    def reset(self):
        self.queues = [[] for _ in range(5)]
        self.rr = [self.priority_seed] * 5
        self.input_port = self.output_port = LOCAL
        self.vc = 0
        self.flit_type = 3
        self.route_mode = 0
        self.result = 0
        self.queue_class = 0
        self.credit_class = 0
        self.congestion_class = 0
        self.arb_class = 0
        self._seq = {}
        self._credit_wait = None
        self._credit_wait_cycles = {}
        self._conflict_loser = None
        self._three_conflict = None
        self._overflow_recovery = None
        self._adaptive_sig = None
        self._blocked_tail = None
        self._packet_phase = [0] * 5
        self._packet_sig = [None] * 5
        self.forward_valid = False
        self.activity_valid = False
        self.fwd_input_port = self.fwd_output_port = LOCAL
        self.fwd_vc = 0
        self.fwd_flit_type = 3
        self.fwd_route_mode = 0
        self.fwd_result = 0

    def _clear_pulses(self):
        self.result = 0
        self.arb_class = 0
        self.forward_valid = False
        self.activity_valid = False
        for key in list(self._seq):
            self._seq[key] = False

    @staticmethod
    def _popcount(value):
        return bin(int(value) & 0x1F).count("1")

    def _route(self, packet, congestion_mask):
        dx, dy = packet["x"], packet["y"]
        if dx == 1 and dy == 1:
            return LOCAL, 3
        x_port = EAST if dx > 1 else WEST
        y_port = NORTH if dy > 1 else SOUTH
        x_needed, y_needed = dx != 1, dy != 1
        if self.xy_order:
            primary = x_port if x_needed else y_port
            alternate = y_port if y_needed else primary
        else:
            primary = y_port if y_needed else x_port
            alternate = x_port if x_needed else primary
        if packet["vc"] == self.escape_vc:
            return primary, 2
        congested = bool(congestion_mask & (1 << primary))
        enough = self._popcount(congestion_mask) >= self.congestion_threshold
        if congested and enough and x_needed and y_needed and alternate != primary:
            return alternate, 1
        return primary, 0

    def _queue_class(self, port):
        level = len(self.queues[port])
        if level == 0:
            return 0
        if level == 1:
            return 1
        if level >= self.buffer_depth:
            return 3
        return 2

    def step(self, action):
        self._clear_pulses()
        if not int(action.get("reset_n", 1)):
            self.reset()
            return

        credits = int(action.get("credit_mask", 0)) & 0x1F
        congestion = int(action.get("congestion_mask", 0)) & 0x1F
        credit_count = self._popcount(credits)
        congestion_count = self._popcount(congestion)
        self.credit_class = 0 if credit_count == 0 else (2 if credit_count == 5 else 1)
        self.congestion_class = 0 if congestion_count == 0 else (1 if congestion_count == 1 else 2)

        recovery_pending = self._overflow_recovery is not None
        recovery_progress = False
        if action.get("router_stall"):
            self.result = 7
        else:
            requests = [[] for _ in range(5)]
            route_modes = {}
            for port in range(5):
                if self.queues[port]:
                    out, mode = self._route(self.queues[port][0], congestion)
                    requests[out].append(port)
                    route_modes[(port, out)] = mode
            max_contenders = max((len(items) for items in requests), default=0)
            self.arb_class = 0 if max_contenders == 0 else (1 if max_contenders == 1 else 2)
            for out, contenders in enumerate(requests):
                if not contenders:
                    continue
                if not (credits & (1 << out)):
                    self.result = 4
                    wait_key = (contenders[0], out)
                    self._credit_wait = wait_key
                    self._credit_wait_cycles[wait_key] = (
                        self._credit_wait_cycles.get(wait_key, 0) + 1)
                    if (self.queues[contenders[0]][0]["type"] == 2 and
                            credits == 0):
                        if (self._blocked_tail and
                                self._blocked_tail[0] == wait_key):
                            self._blocked_tail = (wait_key,
                                                  self._blocked_tail[1] + 1)
                        else:
                            self._blocked_tail = (wait_key, 1)
                    continue
                order = sorted(contenders, key=lambda p: (p - self.rr[out]) % 5)
                winner = order[0]
                packet = self.queues[winner].pop(0)
                packet_sig = (winner, packet["x"], packet["y"],
                              packet["payload"] & 0xFF)
                self.input_port, self.output_port = winner, out
                self.vc, self.flit_type = packet["vc"], packet["type"]
                self.route_mode = route_modes[(winner, out)]
                self.result = 3 if out == LOCAL else (6 if len(contenders) > 1 else 2)
                self.forward_valid = True
                self.activity_valid = True
                self.fwd_input_port, self.fwd_output_port = winner, out
                self.fwd_vc, self.fwd_flit_type = packet["vc"], packet["type"]
                self.fwd_route_mode, self.fwd_result = self.route_mode, self.result
                self.rr[out] = (winner + 1) % 5
                self._seq["seq1"] = True
                if self._credit_wait == (winner, out):
                    self._seq["seq2"] = True
                    self._credit_wait = None
                if self._credit_wait_cycles.pop((winner, out), 0) >= 32:
                    self._seq["seq9"] = True
                if (self._blocked_tail and
                        self._blocked_tail[0] == (winner, out) and
                        self._blocked_tail[1] >= 8 and packet["type"] == 2 and
                        credits == 31):
                    self._seq["seq14"] = True
                    self._blocked_tail = None
                if self._conflict_loser == winner:
                    self._seq["seq3"] = True
                    self._conflict_loser = None
                if len(contenders) > 1:
                    self._conflict_loser = order[1]
                if len(contenders) >= 5 and self._three_conflict is None:
                    self._three_conflict = (out, set(contenders))
                if self._three_conflict and self._three_conflict[0] == out:
                    self._three_conflict[1].discard(winner)
                    if not self._three_conflict[1]:
                        self._seq["seq10"] = True
                        self._three_conflict = None
                if self.route_mode == 1:
                    self._seq["seq4"] = True
                    self._adaptive_sig = packet_sig
                if self.route_mode == 2 and (congestion & (1 << out)):
                    self._seq["seq5"] = True
                if self.route_mode == 2 and self._adaptive_sig == packet_sig:
                    self._seq["seq11"] = True
                if self.route_mode == 0 and self._adaptive_sig == packet_sig:
                    self._seq["seq13"] = True
                if out == LOCAL:
                    self._seq["seq8"] = True
                if (self._overflow_recovery and
                        self._overflow_recovery[0] == winner and
                        credits == 31 and congestion == 0 and
                        not action.get("valid")):
                    self._overflow_recovery[1] -= 1
                    recovery_progress = True
                    if self._overflow_recovery[1] == 0:
                        self._seq["seq12"] = True
                        self._overflow_recovery = None

        if recovery_pending and not recovery_progress:
            self._overflow_recovery = None

        if action.get("valid"):
            port = int(action.get("in_port", 0)) % 5
            self.input_port = port
            self.activity_valid = True
            self.vc = int(action.get("vc", 0)) & 1
            self.flit_type = int(action.get("flit_type", 0)) & 3
            if len(self.queues[port]) >= self.buffer_depth:
                self.result = 5
                self._seq["seq6"] = True
                self._overflow_recovery = [port, len(self.queues[port])]
            else:
                packet = {"x": int(action.get("dest_x", 1)) % 3,
                          "y": int(action.get("dest_y", 1)) % 3,
                          "vc": self.vc, "type": self.flit_type,
                          "payload": int(action.get("payload", 0)) & 0xFFFFFFFF}
                self.queues[port].append(packet)
                self.result = 1
                phase = self._packet_phase[port]
                signature = (packet["x"], packet["y"], packet["vc"],
                             packet["payload"] & 0xFF)
                if self.flit_type == 0:
                    self._packet_phase[port] = 1
                    self._packet_sig[port] = signature
                elif (self.flit_type == 1 and phase == 1 and
                      self._packet_sig[port] == signature):
                    self._packet_phase[port] = 2
                elif (self.flit_type == 2 and phase == 2 and
                      self._packet_sig[port] == signature):
                    self._packet_phase[port] = 0
                    self._packet_sig[port] = None
                    self._seq["seq7"] = True
                elif self.flit_type == 3:
                    self._packet_phase[port] = 0
                    self._packet_sig[port] = None
                else:
                    self._packet_phase[port] = 0
                    self._packet_sig[port] = None
            self.queue_class = self._queue_class(port)
        else:
            self.queue_class = self._queue_class(self.input_port)

    def read_signals(self):
        out = {"input_port": self.input_port, "output_port": self.output_port,
               "vc": self.vc, "flit_type": self.flit_type,
               "vc_role": int(self.vc == self.escape_vc),
               "route_mode": self.route_mode, "result": self.result,
               "queue_class": self.queue_class, "credit_class": self.credit_class,
               "congestion_class": self.congestion_class, "arb_class": self.arb_class,
               "activity_valid": int(self.activity_valid),
               "forward_valid": int(self.forward_valid),
               "event_input_port": self.input_port,
               "event_vc": self.vc, "event_flit_type": self.flit_type,
               "fwd_input_port": self.fwd_input_port,
               "fwd_output_port": self.fwd_output_port,
               "fwd_vc_role": int(self.fwd_vc == self.escape_vc),
               "fwd_flit_type": self.fwd_flit_type,
               "fwd_route_mode": self.fwd_route_mode,
               "fwd_result": self.fwd_result}
        out.update({f"seq{i}": int(self._seq.get(f"seq{i}", False))
                    for i in range(1, 15)})
        return out


if __name__ == "__main__":
    dut = NocRouterValidation()
    print("noc_router_validation local_sim loaded", dut.read_signals())
