#!/usr/bin/env python3
"""Cycle-level Python model of a tiny two-way write-back cache controller."""

IDLE, LOOKUP, WRITEBACK, REFILL, RESPOND, FLUSH_SCAN = range(6)
READ, WRITE, INVALIDATE, FLUSH = range(4)


class CacheCtrlValidation:
    SETS = 4
    WAYS = 2

    def __init__(self, replacement_xor=0, memory_latency=3, poison_tag=0x2A):
        self.replacement_xor = int(replacement_xor) & 1
        self.memory_latency = max(1, int(memory_latency))
        self.poison_tag = int(poison_tag) & 0x3FFFFFF
        self.reset()

    def reset(self):
        self.valid = [[False] * self.WAYS for _ in range(self.SETS)]
        self.dirty = [[False] * self.WAYS for _ in range(self.SETS)]
        self.tags = [[0] * self.WAYS for _ in range(self.SETS)]
        self.data = [[0] * self.WAYS for _ in range(self.SETS)]
        self.lru = [0] * self.SETS
        self.state = IDLE
        self.op = READ
        self.addr = self.wdata = 0
        self.set_idx = self.way = self.offset = 0
        self.tag = 0
        self.result = 0
        self.stalled = 0
        self.countdown = 0
        self.flush_slot = 0
        self._after_writeback = REFILL
        self._seq = {}
        self._filled = set()
        self._written = set()
        self._invalidated = set()
        self._stall_seen = False

    def _decode(self):
        self.offset = self.addr & 0xF
        self.set_idx = (self.addr >> 4) & 3
        self.tag = (self.addr >> 6) & 0x3FFFFFF

    def _find(self):
        for way in range(self.WAYS):
            if self.valid[self.set_idx][way] and self.tags[self.set_idx][way] == self.tag:
                return way
        return None

    def _victim(self):
        for way in range(self.WAYS):
            if not self.valid[self.set_idx][way]:
                return way
        return self.lru[self.set_idx] ^ self.replacement_xor

    def _begin_wait(self, next_state):
        self.countdown = self.memory_latency
        self._after_writeback = next_state

    def _clear_seq_pulses(self):
        for key in list(self._seq):
            self._seq[key] = False

    def step(self, req):
        self._clear_seq_pulses()
        self.stalled = 0
        if not int(req.get("reset_n", 1)):
            self.reset()
            return

        mem_ready = bool(req.get("mem_ready", 1))
        if self.state == IDLE:
            if req.get("req_valid"):
                self.op = int(req.get("opcode", 0)) & 3
                self.addr = int(req.get("addr", 0)) & 0xFFFFFFFF
                self.wdata = int(req.get("wdata", 0)) & 0xFFFFFFFF
                self._decode()
                self.result = 0
                if self.op == FLUSH:
                    self.flush_slot = 0
                    self.state = FLUSH_SCAN
                else:
                    self.state = LOOKUP
            return

        if self.state == LOOKUP:
            hit_way = self._find()
            if self.op == INVALIDATE:
                if hit_way is None:
                    self.result = 2
                else:
                    self.way = hit_way
                    self.result = 5
                    self.valid[self.set_idx][hit_way] = False
                    self.dirty[self.set_idx][hit_way] = False
                    self._invalidated.add(self.addr >> 4)
                self.state = RESPOND
                return
            if hit_way is not None:
                self.way = hit_way
                self.result = 1
                self.lru[self.set_idx] = 1 - hit_way
                if self.op == WRITE:
                    self.data[self.set_idx][hit_way] = self.wdata
                    self.dirty[self.set_idx][hit_way] = True
                    self._written.add(self.addr >> 4)
                elif (self.addr >> 4) in self._filled:
                    self._seq["seq2"] = True
                if self.op == READ and (self.addr >> 4) in self._written:
                    self._seq["seq3"] = True
                self.state = RESPOND
                return

            self.result = 2
            self.way = self._victim()
            if self.valid[self.set_idx][self.way]:
                if self.dirty[self.set_idx][self.way]:
                    self.result = 4
                    self._seq["seq5"] = True
                    self.state = WRITEBACK
                    self._begin_wait(REFILL)
                else:
                    self.result = 3
                    self.state = REFILL
                    self._begin_wait(RESPOND)
                self._seq["seq4"] = True
            else:
                self.state = REFILL
                self._begin_wait(RESPOND)
                self._seq["seq1"] = True
            return

        if self.state in (WRITEBACK, REFILL):
            if not mem_ready:
                self.stalled = 1
                self._stall_seen = True
                return
            if self._stall_seen:
                self._seq["seq6"] = True
                self._stall_seen = False
            self.countdown -= 1
            if self.countdown > 0:
                return
            if self.state == WRITEBACK:
                self.dirty[self.set_idx][self.way] = False
                self.state = self._after_writeback
                self._begin_wait(RESPOND)
                return
            self.valid[self.set_idx][self.way] = True
            self.tags[self.set_idx][self.way] = self.tag
            self.data[self.set_idx][self.way] = self.wdata if self.op == WRITE else 0
            self.dirty[self.set_idx][self.way] = self.op == WRITE
            self.lru[self.set_idx] = 1 - self.way
            line = self.addr >> 4
            self._filled.add(line)
            if self.op == WRITE:
                self._written.add(line)
            if line in self._invalidated:
                self._seq["seq8"] = True
            if self.tag == self.poison_tag:
                self.result = 7
            self.state = RESPOND
            return

        if self.state == RESPOND:
            self.state = IDLE
            return

        # FLUSH_SCAN: dirty entries consume memory beats; clean/invalid entries
        # advance immediately. One slot is exposed per cycle for coverage.
        self.set_idx, self.way = divmod(self.flush_slot, self.WAYS)
        if self.valid[self.set_idx][self.way] and self.dirty[self.set_idx][self.way]:
            self.result = 6
            if not mem_ready:
                self.stalled = 1
                self._stall_seen = True
                return
            self.dirty[self.set_idx][self.way] = False
        self.valid[self.set_idx][self.way] = False
        self.flush_slot += 1
        if self.flush_slot >= self.SETS * self.WAYS:
            self._seq["seq7"] = True
            self.state = RESPOND

    @staticmethod
    def _data_class(value):
        value &= 0xFFFFFFFF
        if value == 0:
            return 0
        if value == 0xFFFFFFFF:
            return 1
        if value == 0xAAAAAAAA:
            return 2
        return 3

    def read_signals(self):
        signals = {
            "state": self.state, "op": self.op, "set_idx": self.set_idx,
            "way": self.way, "result": self.result, "offset": self.offset,
            "data_class": self._data_class(self.wdata), "stalled": self.stalled,
            "valid_count": sum(map(sum, self.valid)),
            "dirty_count": sum(map(sum, self.dirty)),
        }
        signals.update({f"seq{i}": int(self._seq.get(f"seq{i}", False))
                        for i in range(1, 9)})
        return signals


if __name__ == "__main__":
    dut = CacheCtrlValidation()
    print("cache_ctrl_validation local_sim loaded", dut.read_signals())
