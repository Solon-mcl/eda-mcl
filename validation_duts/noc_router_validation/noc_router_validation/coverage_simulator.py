#!/usr/bin/env python3
import json
import numpy as np


class CoverageSimulator:
    def __init__(self, meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            self.meta = json.load(handle)
        self.coverpoints = self.meta["coverpoints"]
        self.total_bins = sum(len(cp.get("bins", ())) for cp in self.coverpoints)
        if int(self.meta.get("total_bins", self.total_bins)) != self.total_bins:
            raise ValueError("coverage metadata total_bins mismatch")
        self.reset()

    def reset(self):
        self.bin_counts = {cp["name"]: {b["name"]: 0 for b in cp.get("bins", ())}
                           for cp in self.coverpoints}
        self.hit_bins = 0
        return self._vector()

    def _hit(self, cp, name):
        if self.bin_counts[cp][name] == 0:
            self.hit_bins += 1
        self.bin_counts[cp][name] += 1

    def step(self, signals):
        for cp in self.coverpoints:
            guard = cp.get("when")
            if guard and not signals.get(guard, False):
                continue
            if cp["type"] == "seq_hit":
                if signals.get(cp["seq"]):
                    self._hit(cp["name"], cp["bins"][0]["name"])
                continue
            values = [signals.get(name.strip()) for name in cp["signal"].split(",")]
            if any(value is None for value in values):
                continue
            for item in cp.get("bins", ()):
                match = (values == item["cross"] if cp["type"] == "cross"
                         else int(values[0]) == int(item["value"]))
                if match:
                    self._hit(cp["name"], item["name"])
        return self._vector()

    @property
    def coverage(self):
        return self.hit_bins / self.total_bins if self.total_bins else 0.0

    def get_bin_vector(self):
        return list(self._vector())

    def _vector(self):
        values = []
        for cp in self.coverpoints:
            values.extend(float(self.bin_counts[cp["name"]][b["name"]] > 0)
                          for b in cp.get("bins", ()))
        return np.asarray(values, dtype=np.float32)
