"""Coverage-directed macro scheduling for unknown DUTs.

This module is intentionally protocol-agnostic. It schedules reusable
verification primitives and learns their observed bin yield without updating
model weights.
"""

from __future__ import annotations

import math


MACROS = ("configure", "control", "temporal", "recovery")


class CoverageMacroScheduler:
    """Small UCB scheduler using newly covered bins per consumed cycle."""

    def __init__(self):
        self.counts = {name: 0 for name in MACROS}
        self.rewards = {name: 0.0 for name in MACROS}
        self.costs = {name: 0 for name in MACROS}
        self.active = None
        self.started_step = 0
        self.started_covered = 0
        self.last_gain_step = 0
        self.history = []

    def _finish(self, covered: int, step: int):
        if self.active is None:
            self.started_covered = covered
            self.started_step = step
            return
        elapsed = max(1, int(step) - self.started_step)
        reward = max(0, int(covered) - self.started_covered)
        self.counts[self.active] += 1
        self.rewards[self.active] += reward
        self.costs[self.active] += elapsed
        if reward:
            self.last_gain_step = int(step)
        self.history.append({"macro": self.active, "cycles": elapsed,
                             "new_bins": reward})

    def select(self, covered: int, step: int, max_steps: int) -> str:
        self._finish(int(covered), int(step))
        untried = [name for name in MACROS if self.counts[name] == 0]
        if untried:
            chosen = untried[0]
        else:
            total = max(1, sum(self.counts.values()))
            patience = max(128, min(2048, int(max_steps) // 40))
            stagnant = int(step) - self.last_gain_step >= patience
            scores = {}
            for name in MACROS:
                rate = self.rewards[name] / max(1, self.costs[name])
                explore = 0.08 * math.sqrt(math.log(total + 1) /
                                           self.counts[name])
                # Once short macros stop paying, favor state accumulation and
                # recovery so irreversible/temporal regions get new episodes.
                phase_bonus = (0.04 if stagnant and name in
                               ("temporal", "recovery") else 0.0)
                scores[name] = rate + explore + phase_bonus
            chosen = max(MACROS, key=lambda name: (scores[name], -MACROS.index(name)))
        self.active = chosen
        self.started_step = int(step)
        self.started_covered = int(covered)
        return chosen

