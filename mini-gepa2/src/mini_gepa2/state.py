from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .adapter import Candidate


def _mean(score_map: dict[int, float]) -> float:
    return sum(score_map.values()) / len(score_map)


@dataclass(slots=True)
class MiniGEPAState:
    """Persistent optimizer state for the teaching implementation.

    The official repository stores much more information, but this reduced state still
    captures the main algorithmic ideas:

    - all discovered candidates
    - their parent links
    - each candidate's validation scores
    - the per-example Pareto frontier
    - which component each candidate should mutate next
    """

    program_candidates: list[Candidate]
    parent_program_ids: list[list[int]]
    val_scores_by_candidate: list[dict[int, float]]
    pareto_front_scores: dict[int, float]
    pareto_front_candidates: dict[int, set[int]]
    next_component_index_by_candidate: list[int]
    iteration: int = 0
    total_metric_calls: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_seed_candidate(
        cls,
        seed_candidate: Candidate,
        seed_val_scores: list[float],
    ) -> "MiniGEPAState":
        """Initialize the state from one seed candidate and one full validation pass."""

        val_score_map = {index: score for index, score in enumerate(seed_val_scores)}
        return cls(
            program_candidates=[dict(seed_candidate)],
            parent_program_ids=[[]],
            val_scores_by_candidate=[val_score_map],
            pareto_front_scores=dict(val_score_map),
            pareto_front_candidates={index: {0} for index in val_score_map},
            next_component_index_by_candidate=[0],
        )

    @property
    def component_names(self) -> list[str]:
        return list(self.program_candidates[0].keys())

    def average_val_score(self, candidate_id: int) -> float:
        return _mean(self.val_scores_by_candidate[candidate_id])

    @property
    def best_candidate_id(self) -> int:
        return max(range(len(self.program_candidates)), key=self.average_val_score)

    @property
    def best_candidate(self) -> Candidate:
        return self.program_candidates[self.best_candidate_id]

    def frontier_candidate_ids(self) -> list[int]:
        frontier_ids: set[int] = set()
        for candidate_ids in self.pareto_front_candidates.values():
            frontier_ids.update(candidate_ids)
        return sorted(frontier_ids)

    def add_candidate(
        self,
        candidate: Candidate,
        parent_candidate_id: int,
        val_scores: list[float],
    ) -> int:
        """Add an accepted candidate and update the per-example Pareto frontier.

        The important teaching point is that frontier maintenance happens here, not in the
        proposal logic. First we admit a candidate into the global pool, then we ask how it
        reshapes the search state for future iterations.
        """

        candidate_id = len(self.program_candidates)
        self.program_candidates.append(dict(candidate))
        self.parent_program_ids.append([parent_candidate_id])

        val_score_map = {index: score for index, score in enumerate(val_scores)}
        self.val_scores_by_candidate.append(val_score_map)

        # A child should continue the parent's component-update schedule instead of
        # jumping back to the first component. This matches the spirit of the official
        # implementation, where mutation order is part of the optimizer state rather
        # than an accidental side effect of candidate creation.
        self.next_component_index_by_candidate.append(
            self.next_component_index_by_candidate[parent_candidate_id]
        )

        for example_id, score in val_score_map.items():
            best_score = self.pareto_front_scores[example_id]
            if score > best_score:
                self.pareto_front_scores[example_id] = score
                self.pareto_front_candidates[example_id] = {candidate_id}
            elif score == best_score:
                self.pareto_front_candidates[example_id].add(candidate_id)

        return candidate_id
