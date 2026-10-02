"""Target-first scheduling of spec-supported executable coverage programs."""

from __future__ import annotations

import math


class TargetTaskScheduler:
    """Track target attempts and allocate a bounded share of macro slots.

    Only completed-program feedback is used. Coverpoint names serve as opaque
    group identifiers for fair exploration; no DUT-specific names are tested.
    """

    def __init__(self, targets, dependency_graph, max_failed_attempts=2,
                 focus_addresses=()):
        self.targets = {item.index: item for item in targets}
        self.graph = dependency_graph
        self.max_failed_attempts = max(1, int(max_failed_attempts))
        self.focus_addresses = frozenset(int(item) for item in focus_addresses)
        self.cursor = 0
        self.stats = {}
        self.group_attempts = {}
        self.decisions = 0

    def observe(self, history):
        for record in history[self.cursor:]:
            context = record.get("context", {})
            if not context.get("joint_candidate"):
                continue
            index = int(context.get("target_index", -1))
            target = self.targets.get(index)
            if target is None:
                continue
            stat = self.stats.setdefault(index, {"attempts": 0, "hits": 0,
                                                  "new_bins": 0, "cycles": 0})
            stat["attempts"] += 1
            stat["hits"] += int(index in record.get("new_bin_indices", ()))
            stat["new_bins"] += int(record.get("new_bins", 0))
            stat["cycles"] += int(record.get("cycles", 0))
            group = target.coverpoint
            self.group_attempts[group] = self.group_attempts.get(group, 0) + 1
        self.cursor = len(history)

    def select(self, missing, candidates, step, max_steps):
        missing_indices = {target.index for target in missing}
        eligible = []
        for candidate in candidates:
            if candidate.target_index not in missing_indices:
                continue
            if (self.focus_addresses and not any(
                    address in self.focus_addresses for address, _ in
                    candidate.register_writes)):
                continue
            target = self.targets.get(candidate.target_index)
            if target is None:
                continue
            stat = self.stats.get(candidate.target_index)
            if stat and stat["hits"] == 0 and stat["attempts"] >= self.max_failed_attempts:
                continue
            eligible.append((candidate, target))
        if not eligible:
            return None
        self.decisions += 1
        total = max(1, sum(x["attempts"] for x in self.stats.values()))

        def score(pair):
            candidate, target = pair
            stat = self.stats.get(target.index)
            group_trials = self.group_attempts.get(target.coverpoint, 0)
            confidence = self.graph.target_confidence(target.index)
            if stat is None:
                # One trial for a legal, unseen target before replays. A
                # coverpoint with many bins does not get all slots at once.
                exploitation = 0.7 + 0.15 * confidence
                uncertainty = 0.35
            else:
                attempts = stat["attempts"]
                exploitation = (stat["new_bins"] / max(1, stat["cycles"])) * 64
                uncertainty = 0.25 * math.sqrt(math.log(total + 1) / attempts)
            group_fairness = 0.35 / math.sqrt(1 + group_trials)
            cost = max(1, 32 + int(candidate.hold_cycles) +
                       3 * len(candidate.register_writes))
            # Remaining time is a hard legality constraint, not a reward.
            if cost > max_steps - step:
                return (-1.0, 0, 0)
            return (exploitation + uncertainty + group_fairness,
                    -cost, -target.index)

        chosen, _ = max(eligible, key=score)
        if score((chosen, self.targets[chosen.target_index]))[0] < 0:
            return None
        return chosen

    def snapshot(self):
        return {str(index): dict(stat) for index, stat in sorted(self.stats.items())}
