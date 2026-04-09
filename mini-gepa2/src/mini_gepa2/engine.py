from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .adapter import Adapter, Candidate
from .proposal import CandidateProposal
from .reflection import ReflectionLM, propose_component_update
from .sampler import ShuffledBatchSampler
from .selectors import ParetoCandidateSelector, RoundRobinComponentSelector
from .state import MiniGEPAState


@dataclass(slots=True)
class MiniGEPAEngine:
    """A compact, teaching-first version of the GEPA optimization loop."""

    adapter: Adapter
    reflection_lm: ReflectionLM
    trainset: list[Any]
    valset: list[Any]
    seed_candidate: Candidate
    num_iterations: int
    batch_sampler: ShuffledBatchSampler
    candidate_selector: ParetoCandidateSelector
    component_selector: RoundRobinComponentSelector

    def _evaluate_on_valset(
        self,
        candidate: Candidate,
        state: MiniGEPAState,
    ) -> list[float]:
        """Run the expensive global evaluation used to admit a candidate into the pool."""

        evaluation = self.adapter.evaluate(
            batch=list(self.valset),
            candidate=candidate,
            capture_traces=False,
        )
        state.total_metric_calls += len(self.valset)
        return evaluation.scores

    def _accept_reflective_proposal(self, proposal: CandidateProposal) -> bool:
        """Use the official default acceptance rule: strict minibatch improvement."""

        return proposal.after_sum > proposal.before_sum

    def _propose_candidate(self, state: MiniGEPAState) -> CandidateProposal:
        """Run the reflective-mutation half of one GEPA step.

        The teaching version keeps the full idea but executes it sequentially:

        1. choose a parent from the Pareto frontier
        2. sample one train minibatch
        3. evaluate the parent with traces
        4. build reflection input
        5. ask the reflection LM for a replacement text
        6. evaluate the child on the same minibatch
        """

        parent_candidate_id = self.candidate_selector.select_candidate_id(state)
        parent_candidate = state.program_candidates[parent_candidate_id]

        minibatch_indices = self.batch_sampler.next_batch_indices(len(self.trainset))
        minibatch = [self.trainset[index] for index in minibatch_indices]

        evaluation_before = self.adapter.evaluate(
            batch=minibatch,
            candidate=parent_candidate,
            capture_traces=True,
        )
        state.total_metric_calls += len(minibatch)

        component_name = self.component_selector.select_component(
            state=state,
            candidate_id=parent_candidate_id,
        )

        reflective_dataset = self.adapter.make_reflective_dataset(
            candidate=parent_candidate,
            evaluation=evaluation_before,
            components_to_update=[component_name],
        )
        reflective_records = list(reflective_dataset[component_name])

        new_text, rendered_prompt, raw_lm_output = propose_component_update(
            candidate=parent_candidate,
            component_name=component_name,
            reflective_records=reflective_records,
            reflection_lm=self.reflection_lm,
        )

        child_candidate = dict(parent_candidate)
        child_candidate[component_name] = new_text

        evaluation_after = self.adapter.evaluate(
            batch=minibatch,
            candidate=child_candidate,
            capture_traces=False,
        )
        state.total_metric_calls += len(minibatch)

        return CandidateProposal(
            parent_candidate_id=parent_candidate_id,
            updated_component=component_name,
            candidate=child_candidate,
            minibatch_indices=minibatch_indices,
            scores_before=evaluation_before.scores,
            scores_after=evaluation_after.scores,
            reflective_records=reflective_records,
            rendered_prompt=rendered_prompt,
            raw_lm_output=raw_lm_output,
        )

    def run(self) -> MiniGEPAState:
        """Run the full optimization loop.

        The seed candidate is fully evaluated once on the validation set, which gives us
        the initial candidate pool and the initial Pareto frontier. Each later iteration
        first proves local usefulness on a minibatch, then earns admission via a full
        validation pass.
        """

        seed_scores = self.adapter.evaluate(
            batch=list(self.valset),
            candidate=self.seed_candidate,
            capture_traces=False,
        ).scores
        state = MiniGEPAState.from_seed_candidate(
            seed_candidate=self.seed_candidate,
            seed_val_scores=seed_scores,
        )
        state.total_metric_calls += len(self.valset)

        for _ in range(self.num_iterations):
            proposal = self._propose_candidate(state)

            # Local gate: a proposal must first prove that it improved the same minibatch
            # that produced the reflective evidence. This is the cheap filter.
            accepted = self._accept_reflective_proposal(proposal)

            history_entry = {
                "iteration": state.iteration + 1,
                "parent_candidate_id": proposal.parent_candidate_id,
                "updated_component": proposal.updated_component,
                "minibatch_indices": proposal.minibatch_indices,
                "before_sum": proposal.before_sum,
                "after_sum": proposal.after_sum,
                "accepted": accepted,
                "candidate_count_before": len(state.program_candidates),
            }

            if accepted:
                # Global gate: only accepted proposals are worth the expensive full validation
                # pass. If the proposal survives that pass, it is admitted into the candidate
                # pool and can itself become a parent in later iterations.
                val_scores = self._evaluate_on_valset(
                    candidate=proposal.candidate,
                    state=state,
                )
                new_candidate_id = state.add_candidate(
                    candidate=proposal.candidate,
                    parent_candidate_id=proposal.parent_candidate_id,
                    val_scores=val_scores,
                )
                history_entry["accepted_candidate_id"] = new_candidate_id
                history_entry["accepted_val_average"] = state.average_val_score(
                    new_candidate_id
                )

            history_entry["best_candidate_id_after_step"] = state.best_candidate_id
            history_entry["candidate_count_after"] = len(state.program_candidates)
            history_entry["total_metric_calls"] = state.total_metric_calls

            state.history.append(history_entry)
            state.iteration += 1

        return state
