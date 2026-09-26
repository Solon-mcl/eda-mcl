"""Coverage-directed macro scheduling for unknown DUTs.

This module is intentionally protocol-agnostic. It schedules reusable
verification primitives and learns their observed bin yield without updating
model weights.
"""

from __future__ import annotations

import math
from collections import deque


MACROS = ("configure", "control", "temporal", "recovery")
CAMPAIGNS = (
    ("configure", "temporal", "control"),
    ("control", "temporal", "recovery"),
    ("configure", "control", "recovery"),
)


class JointCandidateRanker:
    """Online, weight-free ranking from completed joint-program outcomes.

    The ranker changes scheduling state only. It never modifies model weights,
    which keeps it valid for inference-time coverage feedback.
    """

    def __init__(self, exploration: float = 0.35,
                 max_failed_attempts: int = 2):
        self.exploration = float(exploration)
        self.max_failed_attempts = max(1, int(max_failed_attempts))
        self.history_cursor = 0
        self.stats = {}

    def observe(self, records):
        records = list(records)
        for record in records[self.history_cursor:]:
            context = record.get("context", {})
            if not context.get("joint_candidate"):
                continue
            name = str(context.get("target_name", ""))
            if not name:
                continue
            stat = self.stats.setdefault(name, {
                "attempts": 0, "target_hits": 0, "new_bins": 0,
                "cycles": 0,
            })
            new_indices = {int(item) for item in
                           record.get("new_bin_indices", ())}
            target_index = int(context.get("target_index", -1))
            stat["attempts"] += 1
            stat["target_hits"] += int(target_index in new_indices)
            stat["new_bins"] += int(record.get("new_bins", 0))
            stat["cycles"] += int(record.get("cycles", 0))
        self.history_cursor = len(records)

    def score(self, candidate):
        stat = self.stats.get(candidate.target_name)
        total = sum(item["attempts"] for item in self.stats.values())
        if not stat or not stat["attempts"]:
            # Every legal candidate receives one attempt before replaying a
            # previously unsuccessful candidate.
            return (2.0, 0.0, -int(candidate.target_index))
        attempts = stat["attempts"]
        hit_rate = stat["target_hits"] / attempts
        explore = self.exploration * math.sqrt(
            math.log(total + 1.0) / attempts)
        yield_rate = stat["new_bins"] / max(1, stat["cycles"])
        return (hit_rate + explore, yield_rate, -int(candidate.target_index))

    def select(self, candidates):
        candidates = [candidate for candidate in candidates
                      if (candidate.target_name not in self.stats or
                          self.stats[candidate.target_name]["target_hits"] > 0 or
                          self.stats[candidate.target_name]["attempts"] <
                          self.max_failed_attempts)]
        if not candidates:
            return None
        return max(candidates, key=self.score)

    def snapshot(self):
        return {name: dict(values) for name, values in sorted(self.stats.items())}


class CoverageMacroScheduler:
    """Small UCB scheduler using newly covered bins per consumed cycle."""

    def __init__(self):
        self.counts = {name: 0 for name in MACROS}
        self.direct_rewards = {name: 0.0 for name in MACROS}
        self.rewards = {name: 0.0 for name in MACROS}
        self.costs = {name: 0 for name in MACROS}
        self.active = None
        self.started_step = 0
        self.started_covered = 0
        self.started_bins = frozenset()
        self.last_gain_step = 0
        self.history = []
        self.recent = deque(maxlen=3)
        self.pending = deque()
        self.campaign_index = 0
        self.last_campaign_step = -10**9
        self.campaign_history = []
        self.active_plan = []
        self.active_context = {}

    def _finish(self, covered: int, step: int, covered_bins=None):
        current_bins = frozenset(covered_bins or ())
        if self.active is None:
            self.started_covered = covered
            self.started_step = step
            self.started_bins = current_bins
            return
        elapsed = max(1, int(step) - self.started_step)
        new_bin_indices = sorted(current_bins - self.started_bins)
        reward = (len(new_bin_indices) if covered_bins is not None else
                  max(0, int(covered) - self.started_covered))
        self.counts[self.active] += 1
        self.direct_rewards[self.active] += reward
        self.rewards[self.active] += reward
        self.costs[self.active] += elapsed
        if reward:
            self.last_gain_step = int(step)
            # Coverage is often observed one or two macros after the setup
            # that made it reachable. Give bounded, decaying credit to the
            # preceding macros instead of attributing everything to the last
            # low-level action sequence.
            for distance, previous in enumerate(reversed(self.recent), start=1):
                self.rewards[previous] += reward * (0.5 ** distance)
        self.history.append({"macro": self.active, "cycles": elapsed,
                             "new_bins": reward,
                             "new_bin_indices": new_bin_indices,
                             "action_sequence": list(self.active_plan),
                             "context": dict(self.active_context),
                             "credited_reward": self.rewards[self.active]})
        self.recent.append(self.active)
        self.active_plan = []
        self.active_context = {}

    def _start_campaign(self, step: int) -> str:
        campaign = CAMPAIGNS[self.campaign_index % len(CAMPAIGNS)]
        self.campaign_index += 1
        self.last_campaign_step = int(step)
        self.pending.extend(campaign[1:])
        self.campaign_history.append({"step": int(step),
                                      "macros": list(campaign)})
        return campaign[0]

    def force_next(self, macro: str):
        if macro in MACROS:
            self.pending.appendleft(macro)

    def attach_plan(self, actions):
        self.active_plan = list(actions)

    def attach_context(self, **context):
        self.active_context.update(context)

    def select(self, covered: int, step: int, max_steps: int,
               covered_bins=None, target_weights=None, model_scores=None) -> str:
        self._finish(int(covered), int(step), covered_bins)
        untried = [name for name in MACROS if self.counts[name] == 0]
        if untried:
            chosen = untried[0]
        elif self.pending:
            chosen = self.pending.popleft()
        else:
            total = max(1, sum(self.counts.values()))
            patience = max(128, min(2048, int(max_steps) // 40))
            stagnant = int(step) - self.last_gain_step >= patience
            if (stagnant and
                    int(step) - self.last_campaign_step >= patience):
                chosen = self._start_campaign(step)
            else:
                scores = {}
                for name in MACROS:
                    rate = self.rewards[name] / max(1, self.costs[name])
                    explore = 0.08 * math.sqrt(math.log(total + 1) /
                                               self.counts[name])
                    # Once short macros stop paying, favor state accumulation
                    # and recovery between complete campaign attempts.
                    phase_bonus = (0.04 if stagnant and name in
                                   ("temporal", "recovery") else 0.0)
                    target_bonus = 0.06 * float((target_weights or {}).get(name, 0.0))
                    learned_bonus = 0.0
                    if model_scores is not None:
                        values = list(model_scores)
                        if len(values) == len(MACROS):
                            span = max(values) - min(values)
                            learned_bonus = (0.08 * (values[MACROS.index(name)] -
                                                     min(values)) /
                                             max(1e-6, span))
                    scores[name] = (rate + explore + phase_bonus +
                                    target_bonus + learned_bonus)
                chosen = max(MACROS, key=lambda name: (
                    scores[name], -MACROS.index(name)))
        self.active = chosen
        self.started_step = int(step)
        self.started_covered = int(covered)
        self.started_bins = frozenset(covered_bins or ())
        self.active_context = {
            "step": int(step),
            "max_steps": int(max_steps),
            "covered_bins": int(covered),
            "target_weights": {name: float((target_weights or {}).get(name, 0.0))
                               for name in MACROS},
        }
        return chosen


class EpisodeManager:
    """Starts fresh parameterized episodes after progress or stagnation."""

    def __init__(self):
        self.episode = 0
        self.started_step = 0
        self.started_bins = frozenset()
        self.last_progress_step = 0
        self.history = []

    def observe(self, covered_bins, step: int, max_steps: int):
        current = frozenset(covered_bins)
        new_bins = sorted(current - self.started_bins)
        age = int(step) - self.started_step
        min_age = max(64, min(512, int(max_steps) // 200))
        patience = max(256, min(4096, int(max_steps) // 30))
        reason = None
        if new_bins:
            self.last_progress_step = int(step)
            if age >= min_age:
                reason = "coverage_gain"
        elif int(step) - self.last_progress_step >= patience:
            reason = "stagnation"
        if reason:
            self.history.append({
                "episode": self.episode,
                "start_step": self.started_step,
                "end_step": int(step),
                "reason": reason,
                "new_bin_indices": new_bins,
            })
            self.episode += 1
            self.started_step = int(step)
            self.started_bins = current
            self.last_progress_step = int(step)
        return reason
