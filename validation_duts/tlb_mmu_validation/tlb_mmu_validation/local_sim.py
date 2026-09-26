#!/usr/bin/env python3
"""Behavioural model of an ASID-tagged TLB backed by a synthetic page walk.

Unlike the public serial/DMA controllers, the cache controller, the branch
predictor, and the watchdog supervisor, coverage here is driven by address
translation: permission semantics, ASID scoping, global pages, hardware
accessed/dirty bits, locked pages, and page-table-root changes.
"""

LOAD, STORE, FETCH, ATOMIC = range(4)
IDLE = 4


class TlbMmuValidation:
    def __init__(self, tlb_sets=4, tlb_ways=2, replace_xor=1, asid_bits=2,
                 walk_salt=5, lock_enable=1):
        self.tlb_sets = min(8, max(4, int(tlb_sets)))
        self.tlb_ways = min(4, max(2, int(tlb_ways)))
        self.replace_xor = int(replace_xor) & 1
        self.asid_bits = min(2, max(1, int(asid_bits)))
        self.walk_salt = int(walk_salt) & 0xF
        self.lock_enable = int(lock_enable) & 1
        self.reset()

    @property
    def asid_mask(self):
        return (1 << self.asid_bits) - 1

    def reset(self):
        self.tlb = [[self._empty_entry() for _ in range(self.tlb_ways)]
                    for _ in range(self.tlb_sets)]
        self.victim = [0] * self.tlb_sets
        self.satp_delta = 0
        self.hit_pages = set()

        self.vpn = 0
        self.op = LOAD
        self.priv = 1
        self.asid = 0
        self.tlb_result = IDLE
        self.walk_result = 0
        self.hazard = IDLE
        self.result = IDLE
        self.dirty_before = IDLE
        self.accessed_clear = IDLE
        self.entry_global = IDLE
        self.fence_class = 0
        self.stalled = 0
        self.asid_alias = 0

        self._seq = {}
        self._prev_refill = False
        self._refill_vpn = None
        self._prev_fault = False
        self._stall_seen = False
        self._fence_seen = False
        self._fence_scope = 0
        self._satp_seen = False

    @staticmethod
    def _empty_entry():
        return {"valid": False, "vpn": 0, "asid": 0, "global": 0,
                "pte": {"valid": False, "locked": False, "pfn": 0, "r": 0,
                        "w": 0, "x": 0, "u": 0, "dirty": 0, "accessed": 0}}

    # ------------------------------------------------------------------
    # Address translation
    # ------------------------------------------------------------------
    def _set_index(self, vpn):
        return (vpn ^ self.walk_salt) & (self.tlb_sets - 1)

    def _walk(self, vpn):
        """Synthetic page-table walk; the hidden salt perturbs every field."""
        seed = (vpn ^ self.walk_salt ^ (self.satp_delta * 0x5D)) & 0xFFFFFFFF
        return {
            "valid": (seed % 7) != 0,
            "locked": bool(self.lock_enable and (seed & 0x8)),
            "pfn": ((seed * 0x9E3779B1) ^ (self.walk_salt << 7)) & 0xFFFFF,
            "r": 1,
            "w": (seed >> 2) & 1,
            "x": (seed >> 1) & 1,
            "u": (seed >> 4) & 1,
            "dirty": 0,
            "accessed": 1 if vpn in self.hit_pages else 0,
        }

    def _install(self, set_idx, vpn, pte, want_global):
        entry = {"valid": True, "vpn": vpn, "asid": self.asid,
                 "global": want_global, "pte": dict(pte)}
        ways = self.tlb[set_idx]
        for way in range(self.tlb_ways):
            if not ways[way]["valid"]:
                ways[way] = entry
                return False
        victim = self.victim[set_idx] ^ self.replace_xor
        ways[victim] = entry
        self.victim[set_idx] = 1 - victim
        return True

    def _flush_all(self):
        for ways in self.tlb:
            for way in range(self.tlb_ways):
                ways[way] = self._empty_entry()

    def _flush_scope(self):
        """ASID-scoped fence: local translations go, global ones survive."""
        for ways in self.tlb:
            for way in range(self.tlb_ways):
                if ways[way]["valid"] and not ways[way]["global"]:
                    ways[way] = self._empty_entry()

    # ------------------------------------------------------------------
    # Stimulus step
    # ------------------------------------------------------------------
    def _clear_pulses(self):
        for key in list(self._seq):
            self._seq[key] = False
        self.stalled = 0
        self.fence_class = 0
        self.tlb_result = IDLE
        self.walk_result = IDLE
        self.hazard = IDLE
        self.result = IDLE
        self.dirty_before = IDLE
        self.accessed_clear = IDLE
        self.entry_global = IDLE

    def step(self, request):
        self._clear_pulses()
        if not int(request.get("reset_n", 1)):
            self.reset()
            return

        if request.get("flush"):
            self._flush_all()
            self.fence_class = 2
            self._fence_seen = True
            self._fence_scope = 2
        elif request.get("satp"):
            self.satp_delta ^= 1
            self.hit_pages.clear()
            self._flush_all()
            self.fence_class = 2
            self._fence_seen = True
            self._fence_scope = 2
            self._satp_seen = True
        elif request.get("fence"):
            self._flush_scope()
            self.fence_class = 1
            self._fence_seen = True
            self._fence_scope = 1

        if request.get("stall"):
            self.stalled = 1
            self._stall_seen = True
            return
        if not request.get("valid"):
            return

        vpn = int(request.get("vpn", 0)) & 0xFFFFFFFF
        self.vpn = vpn
        self.op = int(request.get("op", 0)) & 3
        self.priv = int(request.get("priv", 0)) & 1
        self.asid = int(request.get("asid", 0)) & 3
        want_global = int(request.get("global", 0)) & 1

        set_idx = self._set_index(vpn)
        ways = self.tlb[set_idx]
        hit_way = None
        alias_way = None
        for way, entry in enumerate(ways):
            if not entry["valid"] or entry["vpn"] != vpn:
                continue
            if entry["global"] or (entry["asid"] & self.asid_mask) == \
                    (self.asid & self.asid_mask):
                hit_way = way
                break
            alias_way = way
        self.asid_alias = 1 if (hit_way is None and alias_way is not None) else 0

        if hit_way is not None:
            entry = ways[hit_way]
            pte = entry["pte"]
            self.entry_global = 1 if entry["global"] else 0
            self.tlb_result = 0
            self.walk_result = 0
            fill = False
        else:
            pte = self._walk(vpn)
            self.entry_global = want_global
            if not pte["valid"]:
                self.walk_result = 2
            elif pte["locked"]:
                self.walk_result = 3
            else:
                self.walk_result = 1
            fill = self.walk_result == 1

        self.dirty_before = int(pte["dirty"])
        self.accessed_clear = 0 if pte["accessed"] else 1

        protected = False
        if self.op == STORE and not pte["w"]:
            protected = True
        elif self.op == ATOMIC and not (pte["r"] and pte["w"]):
            protected = True
        elif self.op == FETCH and not pte["x"]:
            protected = True
        if self.priv == 0 and not pte["u"]:
            protected = True

        if not pte["valid"]:
            self.hazard = 1
            self.result = 2
        elif protected:
            self.hazard = 2
            self.result = 3
        elif self.asid_alias:
            self.hazard = 3
            self.result = 1
        else:
            self.hazard = 0
            self.result = 0 if hit_way is not None else 1

        if self.result in (0, 1):
            pte["accessed"] = 1
            if self.op in (STORE, ATOMIC):
                pte["dirty"] = 1
            self.hit_pages.add(vpn)

        if hit_way is None:
            if fill:
                self.tlb_result = 2 if self._install(
                    set_idx, vpn, pte, want_global) else 1
            else:
                self.tlb_result = 3

        # ---- temporal sequences -------------------------------------
        if self._prev_refill and self._refill_vpn == vpn and self.tlb_result == 0:
            self._seq["seq1"] = True
        self._prev_refill = (self.tlb_result == 1)
        self._refill_vpn = vpn

        if self.tlb_result == 2:
            self._seq["seq2"] = True
        if self.hazard == 3:
            self._seq["seq3"] = True

        if self._fence_seen:
            if self._fence_scope == 1:
                if self.tlb_result != 0:
                    self._seq["seq4"] = True
                elif self.entry_global == 1:
                    self._seq["seq5"] = True
            self._fence_seen = False
        if self._satp_seen:
            if self.tlb_result != 0:
                self._seq["seq8"] = True
            self._satp_seen = False

        if self._prev_fault and self.result in (0, 1):
            self._seq["seq6"] = True
        self._prev_fault = self.hazard in (1, 2)

        if self._stall_seen:
            self._seq["seq7"] = True
            self._stall_seen = False

    def read_signals(self):
        signals = {
            "op": self.op, "priv": self.priv,
            "tlb_result": self.tlb_result, "walk_result": self.walk_result,
            "hazard": self.hazard, "result": self.result,
            "dirty_before": self.dirty_before,
            "accessed_clear": self.accessed_clear,
            "fence_class": self.fence_class, "stalled": self.stalled,
            "entry_global": self.entry_global,
        }
        signals.update({f"seq{i}": int(self._seq.get(f"seq{i}", False))
                        for i in range(1, 9)})
        return signals


if __name__ == "__main__":
    dut = TlbMmuValidation()
    print("tlb_mmu_validation local_sim loaded", dut.read_signals())
