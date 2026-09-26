"""Coverage-directed action-program search with no DUT-family routing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SequenceCandidate:
    candidate_id: str
    steps: tuple
    metadata: dict = field(default_factory=dict)
    attempts: int = 0
    new_bins: int = 0
    cycles: int = 0

    @property
    def rate(self):
        return self.new_bins / max(1, self.cycles)

    @property
    def estimated_cycles(self):
        total = 0
        for step in self.steps:
            try:
                total += max(1, int(step[1]))
            except (IndexError, TypeError, ValueError):
                total += 1
        return total


class GenericSequenceSearch:
    """Select and score concrete programs using coverage gain per cycle."""

    def __init__(self, candidates=(), max_candidates=1024):
        self.candidates = []
        self.by_id = {}
        self.max_candidates = max(1, int(max_candidates))
        self._history_cursor = 0
        for candidate in candidates:
            self.add(candidate)

    def add(self, candidate):
        if (len(self.candidates) >= self.max_candidates or
                candidate.candidate_id in self.by_id):
            return False
        self.candidates.append(candidate)
        self.by_id[candidate.candidate_id] = candidate
        return True

    def observe(self, history):
        gained = []
        for item in history[self._history_cursor:]:
            context = item.get("context", {})
            candidate_id = context.get("generic_sequence_candidate")
            candidate = self.by_id.get(candidate_id)
            if candidate is None:
                continue
            candidate.attempts += 1
            delta = int(item.get("new_bins", 0))
            candidate.new_bins += delta
            candidate.cycles += int(item.get("cycles", 0))
            if delta:
                gained.append((candidate, delta,
                               tuple(item.get("new_bin_indices", ()))))
        self._history_cursor = len(history)
        return gained

    def select(self, max_cycles=None, allow_learned=True):
        eligible = self.candidates
        if not allow_learned:
            eligible = [item for item in eligible
                        if not item.metadata.get("learned_trace")]
        if max_cycles is not None:
            eligible = [item for item in eligible
                        if item.estimated_cycles <= max(1, int(max_cycles))]
        if not eligible:
            return None
        untried = [item for item in eligible if item.attempts == 0]
        if untried:
            # Feedback-derived candidates can arrive after a long static
            # candidate list. Explicit priority lets them run promptly while
            # insertion order remains the tie breaker.
            return max(untried, key=lambda item: int(
                item.metadata.get("priority", 10)))
        retryable = [item for item in eligible if item.attempts < 2]
        if not retryable:
            return None
        total = sum(item.attempts for item in self.candidates)
        return max(retryable, key=lambda item: (
            item.rate + 0.05 * math.sqrt(
                math.log(total + 1) / max(1, item.attempts)),
            int(item.metadata.get("priority", 10)),
            item.new_bins, -item.cycles, item.candidate_id))

    def snapshot(self):
        return {
            item.candidate_id: {
                "attempts": item.attempts,
                "new_bins": item.new_bins,
                "cycles": item.cycles,
                "rate": item.rate,
                "metadata": dict(item.metadata),
            } for item in self.candidates
        }
