#!/usr/bin/env python3
"""Cycle-level model of a salted gshare + BTB + RAS branch predictor."""

COND, JUMP, CALL, RETURN = range(4)


class BranchPredictorValidation:
    def __init__(self, history_bits=4, index_salt=9, replacement_xor=1,
                 ras_depth=4):
        self.history_bits = min(6, max(4, int(history_bits)))
        self.index_salt = int(index_salt) & 0xF
        self.replacement_xor = int(replacement_xor) & 1
        self.ras_depth = min(8, max(4, int(ras_depth)))
        self.reset()

    def reset(self):
        self.pht = [1] * 16
        self.pht_owner = [None] * 16
        self.ghr = 0
        self.btb_valid = [[False, False] for _ in range(2)]
        self.btb_tag = [[0, 0] for _ in range(2)]
        self.btb_target = [[0, 0] for _ in range(2)]
        self.btb_victim = [0, 0]
        self.ras = []
        self.kind = COND
        self.outcome = 0
        self.btb_result = 0
        self.pht_before = 1
        self.actual_taken = 0
        self.ras_level = 0
        self.stalled = 0
        self.flushed = 0
        self._seq = {}
        self._prev_mispredict = False
        self._stall_seen = False
        self._flush_seen = False

    @property
    def history_mask(self):
        return (1 << self.history_bits) - 1

    def _pht_index(self, pc):
        return ((pc >> 2) ^ self.ghr ^ self.index_salt) & 0xF

    def _btb_coords(self, pc):
        return ((pc >> 2) ^ self.index_salt) & 1, pc >> 3

    def _btb_find(self, pc):
        set_idx, tag = self._btb_coords(pc)
        for way in range(2):
            if self.btb_valid[set_idx][way] and self.btb_tag[set_idx][way] == tag:
                return set_idx, way
        return set_idx, None

    def _btb_allocate(self, pc, target):
        set_idx, way = self._btb_find(pc)
        replaced = False
        if way is None:
            for candidate in range(2):
                if not self.btb_valid[set_idx][candidate]:
                    way = candidate
                    break
            else:
                way = self.btb_victim[set_idx] ^ self.replacement_xor
                replaced = True
        self.btb_valid[set_idx][way] = True
        self.btb_tag[set_idx][way] = pc >> 3
        self.btb_target[set_idx][way] = target & 0xFFFFFFFF
        self.btb_victim[set_idx] = 1 - way
        return replaced

    def _ghr_class(self):
        value = self.ghr & self.history_mask
        mask = self.history_mask
        alt_a = 0xAAAAAAAA & mask
        alt_5 = 0x55555555 & mask
        if value == 0:
            return 0
        if value == mask:
            return 1
        if value in (alt_a, alt_5):
            return 2
        return 3

    def _ras_level_class(self):
        if self.ras_level == 0:
            return 0
        if self.ras_level == 1:
            return 1
        if self.ras_level >= self.ras_depth:
            return 4
        if self.ras_level >= (self.ras_depth + 1) // 2:
            return 2
        return 3

    def _clear_pulses(self):
        for key in list(self._seq):
            self._seq[key] = False
        self.stalled = 0
        self.flushed = 0
        self.btb_result = 0

    def step(self, request):
        self._clear_pulses()
        if not int(request.get("reset_n", 1)):
            self.reset()
            return
        if request.get("flush"):
            self.ghr = 0
            self.flushed = 1
            self._flush_seen = True
        if request.get("stall"):
            self.stalled = 1
            self._stall_seen = True
            return
        if not request.get("valid"):
            return

        pc = int(request.get("pc", 0)) & 0xFFFFFFFF
        target = int(request.get("target", 0)) & 0xFFFFFFFF
        self.kind = int(request.get("kind", 0)) & 3
        self.actual_taken = int(bool(request.get("actual_taken")))
        self.ras_level = len(self.ras)

        pht_idx = self._pht_index(pc)
        self.pht_before = self.pht[pht_idx]
        owner = self.pht_owner[pht_idx]
        if owner is not None and owner != pc:
            self._seq["seq3"] = True
        self.pht_owner[pht_idx] = pc

        btb_set, btb_way = self._btb_find(pc)
        btb_hit = btb_way is not None
        self.btb_result = 1 if btb_hit else 2
        btb_prediction = self.btb_target[btb_set][btb_way] if btb_hit else 0

        if self.kind == COND:
            pred_taken = self.pht_before >= 2
            pred_target = btb_prediction
        elif self.kind == RETURN:
            pred_taken = bool(self.ras)
            pred_target = self.ras[-1] if self.ras else 0
        else:
            pred_taken = True
            pred_target = btb_prediction

        if int(pred_taken) != self.actual_taken:
            self.outcome = 2
        elif not self.actual_taken:
            self.outcome = 0
        elif pred_target == target:
            self.outcome = 1
        else:
            self.outcome = 3

        if self._prev_mispredict and self.outcome in (0, 1):
            self._seq["seq4"] = True
        self._prev_mispredict = self.outcome in (2, 3)
        if self._stall_seen:
            self._seq["seq7"] = True
            self._stall_seen = False
        if self._flush_seen:
            self._seq["seq8"] = True
            self._flush_seen = False

        if self.kind == COND:
            old = self.pht_before
            self.pht[pht_idx] = min(3, old + 1) if self.actual_taken else max(0, old - 1)
            if self.pht[pht_idx] == 3:
                self._seq["seq1"] = True
            if self.pht[pht_idx] == 0:
                self._seq["seq2"] = True

        if self.kind == RETURN:
            if self.ras:
                expected = self.ras.pop()
                if self.actual_taken and target == expected and self.outcome == 1:
                    self._seq["seq5"] = True
        elif self.kind == CALL and self.actual_taken:
            if len(self.ras) >= self.ras_depth:
                self.ras.pop(0)
                self._seq["seq6"] = True
            self.ras.append((pc + 4) & 0xFFFFFFFF)

        if self.actual_taken:
            if self._btb_allocate(pc, target):
                self.btb_result = 3

        self.ghr = ((self.ghr << 1) | self.actual_taken) & self.history_mask

    def read_signals(self):
        signals = {
            "kind": self.kind, "outcome": self.outcome,
            "btb_result": self.btb_result, "pht_before": self.pht_before,
            "ghr_class": self._ghr_class(), "ras_level": self._ras_level_class(),
            "actual_taken": self.actual_taken, "stalled": self.stalled,
            "flushed": self.flushed,
        }
        signals.update({f"seq{i}": int(self._seq.get(f"seq{i}", False))
                        for i in range(1, 9)})
        return signals


if __name__ == "__main__":
    dut = BranchPredictorValidation()
    print("branch_predictor_validation local_sim loaded", dut.read_signals())
