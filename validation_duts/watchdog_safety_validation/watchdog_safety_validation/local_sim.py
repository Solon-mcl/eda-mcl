#!/usr/bin/env python3
"""Cycle-level model of a windowed watchdog and safety supervisor."""

DISABLED, CLOSED, OPEN, PRETIMEOUT, RESET_PENDING, SAFETY_LOCKED = range(6)


class WatchdogSafetyValidation:
    def __init__(self, service_key_a=0xA5, service_key_b=0x5A,
                 key_gap=3, escalation_limit=3, reset_hold=2):
        self.service_key_a = int(service_key_a) & 0xFF
        self.service_key_b = int(service_key_b) & 0xFF
        self.key_gap = min(8, max(1, int(key_gap)))
        self.escalation_limit = min(5, max(2, int(escalation_limit)))
        self.reset_hold = min(6, max(1, int(reset_hold)))
        self.reset()

    def reset(self):
        self.window = 3
        self.timeout = 8
        self.enabled = False
        self.cfg_locked = False
        self.safety_locked = False
        self.state = DISABLED
        self.event_state = DISABLED
        self.counter = 0
        self.fault_count = 0
        self.key_phase = 0
        self.key_left = 0
        self.pending_left = 0
        self.result = 0
        self._seq = {}

    def _clear_pulses(self):
        self.result = 0
        for key in list(self._seq):
            self._seq[key] = False

    def _add_fault(self):
        self.fault_count += 1
        if self.fault_count >= self.escalation_limit:
            self.fault_count = self.escalation_limit
            self.safety_locked = True
            self.enabled = False
            self.state = SAFETY_LOCKED
            self.event_state = SAFETY_LOCKED
            self.result = 7
            self._seq["seq6"] = True

    def _write_config(self, address, data):
        if self.state != DISABLED or self.cfg_locked or self.safety_locked:
            self.result = 2
            if self.safety_locked:
                self._seq["seq7"] = True
            return
        address &= 3
        if address == 0:
            self.enabled = bool(data & 1)
            self.cfg_locked = self.cfg_locked or bool(data & 2)
            if self.enabled:
                self.counter = 0
                self.state = CLOSED
                self._seq["seq1"] = True
        elif address == 1:
            self.window = min(253, max(2, int(data) & 0xFFFF))
            if self.timeout <= self.window + 1:
                self.timeout = self.window + 2
        elif address == 2:
            self.timeout = min(255, max(self.window + 2, int(data) & 0xFFFF))
        else:
            self.result = 2
            return
        self.result = 1

    def _service(self, key):
        key &= 0xFF
        if self.key_phase == 0:
            if key == self.service_key_a:
                self.key_phase = 1
                self.key_left = self.key_gap
                return
            self.result = 6
            self._add_fault()
            return
        valid_second = self.key_left > 0 and key == self.service_key_b
        self.key_phase = 0
        self.key_left = 0
        if not valid_second:
            self.result = 6
            self._add_fault()
            return
        if self.state == CLOSED:
            self.result = 4
            self._seq["seq4"] = True
            self._add_fault()
        elif self.state == OPEN:
            self.result = 3
            self._seq["seq3"] = True
            self.counter = 0
            self.state = CLOSED
        elif self.state == PRETIMEOUT:
            self.result = 5
            self._seq["seq3"] = True
            self.counter = 0
            self.state = CLOSED
        else:
            self.result = 6
            self._add_fault()

    def step(self, request):
        self._clear_pulses()
        if not int(request.get("reset_n", 1)):
            was_locked = self.safety_locked
            self.reset()
            if was_locked:
                self._seq["seq8"] = True
            return

        self.event_state = self.state

        if self.key_phase and not request.get("service"):
            self.key_left -= 1
            if self.key_left <= 0:
                self.key_phase = 0

        if request.get("reg_we"):
            self._write_config(int(request.get("reg_addr", 0)),
                               int(request.get("data", 0)))

        if request.get("service") and not self.safety_locked:
            self._service(int(request.get("data", 0)))

        if request.get("fault_inject") and not self.safety_locked:
            self._add_fault()

        if self.state == RESET_PENDING:
            self.pending_left -= 1
            if self.pending_left <= 0:
                self.enabled = False
                self.state = DISABLED
            self.event_state = self.state
            return

        if request.get("tick") and self.enabled and not self.safety_locked:
            previous = self.state
            self.counter += 1
            if self.counter >= self.timeout:
                self.state = RESET_PENDING
                self.pending_left = self.reset_hold
                self.enabled = False
                self._seq["seq5"] = True
            elif self.counter >= self.timeout - 2:
                self.state = PRETIMEOUT
            elif self.counter >= self.window:
                self.state = OPEN
                if previous == CLOSED:
                    self._seq["seq2"] = True
            else:
                self.state = CLOSED
            self.event_state = self.state

    def _counter_class(self):
        if self.counter == 0:
            return 0
        if self.counter < self.window:
            return 1
        if self.counter == self.window:
            return 2
        if self.counter >= self.timeout - 2:
            return 3
        return 1

    def read_signals(self):
        fault_class = 3 if self.safety_locked else min(2, self.fault_count)
        signals = {
            "state": self.state, "event_state": self.event_state,
            "result": self.result,
            "counter_class": self._counter_class(),
            "fault_class": fault_class,
            "enabled": int(self.enabled), "cfg_locked": int(self.cfg_locked),
            "safety_locked": int(self.safety_locked), "key_phase": self.key_phase,
        }
        signals.update({f"seq{i}": int(self._seq.get(f"seq{i}", False))
                        for i in range(1, 9)})
        return signals


if __name__ == "__main__":
    dut = WatchdogSafetyValidation()
    print("watchdog_safety_validation local_sim loaded", dut.read_signals())
