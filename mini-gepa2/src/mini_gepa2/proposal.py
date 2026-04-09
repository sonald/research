from __future__ import annotations

from dataclasses import dataclass

from .adapter import Candidate, ReflectiveRecord


@dataclass(slots=True)
class CandidateProposal:
    """A proposed child candidate together with the data that justified it.

    Keeping this as a first-class object makes the GEPA control flow easier to read:
    the proposer is responsible for building a mutation and collecting local evidence,
    while the engine is responsible for deciding whether the mutation is admitted into
    the global candidate pool.
    """

    parent_candidate_id: int
    updated_component: str
    candidate: Candidate
    minibatch_indices: list[int]
    scores_before: list[float]
    scores_after: list[float]
    reflective_records: list[ReflectiveRecord]
    rendered_prompt: str
    raw_lm_output: str

    @property
    def before_sum(self) -> float:
        return sum(self.scores_before)

    @property
    def after_sum(self) -> float:
        return sum(self.scores_after)
