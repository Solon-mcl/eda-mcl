#!/usr/bin/env python3
"""Cycle-level abstract SECDED memory and background scrubber."""

IDLE, WRITE, READ, CORRECT, UNCORRECTABLE, SCRUB = range(6)


class EccMemoryValidation:
    WORDS = 8

    def __init__(self, syndrome_xor=5, scrub_stride=3,
                 correction_latency=3, zero_on_dbe=True, poison_word=6):
        self.syndrome_xor = int(syndrome_xor) & 31
        stride = int(scrub_stride) & 7
        self.scrub_stride = stride if stride in (1, 3, 5, 7) else 3
        self.correction_latency = min(6, max(1, int(correction_latency)))
        self.zero_on_dbe = bool(zero_on_dbe)
        self.poison_word = int(poison_word) & 7
        self.reset()

    def reset(self):
        self.memory = [0] * self.WORDS
        self.errors = [set() for _ in range(self.WORDS)]
        self.state = IDLE
        self.result = 0
        self.op = 0
        self.address = 0
        self.last_data = 0
        self.read_data = 0
        self.scrub_cursor = 0
        self.scrub_count = 0
        self.pending_addr = None
        self.pending_left = 0
        self.stalled = 0
        self._pending_stalled = False
        self._written = set()
        self._seq = {}

    def _clear_pulses(self):
        self.state = IDLE
        self.result = 0
        self.op = 0
        self.stalled = 0
        for key in list(self._seq):
            self._seq[key] = False

    def _error_class(self, address=None):
        address = self.address if address is None else int(address) & 7
        count = len(self.errors[address])
        if count >= 2 and address == self.poison_word:
            return 3
        return min(2, count)

    def _syndrome_class(self):
        return (sum(self.errors[self.address]) + self.syndrome_xor) & 3

    def _finish_correction(self):
        self.errors[self.pending_addr].clear()
        self.address = self.pending_addr
        self.pending_addr = None
        self.state = CORRECT
        self.op = 2
        self.result = 4
        self._seq["seq2"] = True
        if self._pending_stalled:
            self._seq["seq4"] = True
        self._pending_stalled = False

    def step(self, request):
        self._clear_pulses()
        if not int(request.get("reset_n", 1)):
            self.reset()
            return

        if self.pending_addr is not None:
            self.state = CORRECT
            self.op = 2
            if request.get("stall"):
                self.stalled = 1
                self._pending_stalled = True
                return
            self.pending_left -= 1
            if self.pending_left <= 0:
                self._finish_correction()
            return

        address = int(request.get("address", 0)) & 7
        data = int(request.get("data", 0)) & 0xFFFFFFFF

        if request.get("write"):
            self.state, self.op, self.result = WRITE, 1, 1
            self.address, self.last_data = address, data
            if self.errors[address]:
                self._seq["seq7"] = True
            self.memory[address] = data
            self.errors[address].clear()
            self._written.add(address)
            return

        if request.get("inject"):
            self.op = 3
            self.address = address
            bit_a = (int(request.get("bit_a", 0)) ^ self.syndrome_xor) % 33
            bit_b = (int(request.get("bit_b", 0)) ^ self.syndrome_xor) % 33
            self.errors[address].add(bit_a)
            if int(request.get("bit_a", 0)) != int(request.get("bit_b", 0)):
                self.errors[address].add(bit_b)
            self.result = 7 if len(self.errors[address]) >= 2 else 6
            return

        if request.get("read"):
            self.op = 2
            self.address = address
            count = len(self.errors[address])
            if count == 0:
                self.state, self.result = READ, 2
                self.read_data = self.memory[address]
                if address in self._written:
                    self._seq["seq1"] = True
            elif count == 1:
                self.state, self.result = CORRECT, 3
                self.read_data = self.memory[address]
                self.pending_addr = address
                self.pending_left = self.correction_latency
            else:
                self.state, self.result = UNCORRECTABLE, 5
                self.read_data = 0 if self.zero_on_dbe else self.memory[address]
                self._seq["seq3"] = True
            return

        if request.get("scrub"):
            self.state, self.op = SCRUB, 4
            self.address = ((self.scrub_cursor * self.scrub_stride) +
                            self.syndrome_xor) & 7
            if request.get("stall"):
                self.stalled = 1
                return
            count = len(self.errors[self.address])
            if count == 0:
                self.result = 8
            elif count == 1:
                self.result = 9
                self.errors[self.address].clear()
                self._seq["seq5"] = True
            else:
                self.result = 10
                self._seq["seq6"] = True
            self.scrub_cursor = (self.scrub_cursor + 1) & 7
            self.scrub_count += 1
            if self.scrub_cursor == 0:
                self._seq["seq8"] = True

    def _data_class(self):
        value = self.last_data & 0xFFFFFFFF
        if value == 0:
            return 0
        if value == 0xFFFFFFFF:
            return 1
        if value == 0xAAAAAAAA:
            return 2
        return 3

    def _scrub_class(self):
        if self.scrub_cursor == 0:
            return 0
        if self.scrub_cursor == 7:
            return 2
        return 1

    def read_signals(self):
        out = {
            "state": self.state, "result": self.result, "op": self.op,
            "error_class": self._error_class(), "address": self.address,
            "syndrome_class": self._syndrome_class(),
            "data_class": self._data_class(), "scrub_class": self._scrub_class(),
            "stalled": self.stalled,
        }
        out.update({f"seq{i}": int(self._seq.get(f"seq{i}", False))
                    for i in range(1, 9)})
        return out


if __name__ == "__main__":
    dut = EccMemoryValidation()
    print("ecc_memory_validation local_sim loaded", dut.read_signals())
