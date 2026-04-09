from __future__ import annotations

import re

from mini_gepa2.adapter import EvaluationBatch
from mini_gepa2.api import MiniGEPAConfig, optimize
from mini_gepa2.demo import (
    ToyReflectionLM,
    ToyWordTaskAdapter,
    build_demo_examples,
    build_seed_candidate,
)
from mini_gepa2.selectors import RoundRobinComponentSelector
from mini_gepa2.state import MiniGEPAState


def test_optimizer_improves_toy_prompt() -> None:
    trainset, valset = build_demo_examples()
    seed_candidate = build_seed_candidate()

    state = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=ToyWordTaskAdapter(),
        reflection_lm=ToyReflectionLM(),
        config=MiniGEPAConfig(
            num_iterations=6,
            minibatch_size=2,
            seed=0,
        ),
    )

    assert state.average_val_score(0) < 1.0
    assert state.average_val_score(state.best_candidate_id) == 1.0
    assert len(state.program_candidates) > 1

    best_prompt = state.best_candidate["prompt"]
    assert "banana => fruit" in best_prompt
    assert "hammer => tool" in best_prompt
    assert "apple => fruit" in best_prompt
    assert "dog => animal" in best_prompt
    assert "wrench => tool" in best_prompt


def test_round_robin_selector_cycles_per_candidate() -> None:
    state = MiniGEPAState.from_seed_candidate(
        seed_candidate={"system": "a", "critic": "b"},
        seed_val_scores=[0.0],
    )
    selector = RoundRobinComponentSelector()

    assert selector.select_component(state, 0) == "system"
    assert selector.select_component(state, 0) == "critic"
    assert selector.select_component(state, 0) == "system"


def test_child_inherits_component_cursor_from_parent() -> None:
    state = MiniGEPAState.from_seed_candidate(
        seed_candidate={"system": "a", "critic": "b"},
        seed_val_scores=[0.0],
    )
    selector = RoundRobinComponentSelector()

    # Mutate the parent once so its cursor advances to the next component.
    assert selector.select_component(state, 0) == "system"

    child_id = state.add_candidate(
        candidate={"system": "a2", "critic": "b2"},
        parent_candidate_id=0,
        val_scores=[0.1],
    )

    assert state.next_component_index_by_candidate[child_id] == 1
    assert selector.select_component(state, child_id) == "critic"


class _TrackingAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str, bool]] = []

    def evaluate(self, batch, candidate, capture_traces: bool = False):
        prompt = candidate["prompt"]
        batch_key = tuple(str(example) for example in batch)
        self.calls.append((batch_key, prompt, capture_traces))

        if prompt == "improved":
            scores = [1.0 for _ in batch]
        else:
            scores = [0.0 for _ in batch]

        trajectories = None
        if capture_traces:
            trajectories = [
                {
                    "input": str(example),
                    "prediction": "wrong",
                    "expected_label": "right",
                    "feedback": "needs improvement",
                }
                for example in batch
            ]

        return EvaluationBatch(
            outputs=[prompt for _ in batch],
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(self, candidate, evaluation, components_to_update):
        records = []
        for output, score, trajectory in zip(
            evaluation.outputs,
            evaluation.scores,
            evaluation.trajectories or [],
            strict=False,
        ):
            records.append(
                {
                    "input": trajectory["input"],
                    "prediction": output,
                    "expected_label": trajectory["expected_label"],
                    "score": score,
                    "feedback": trajectory["feedback"],
                }
            )

        return {component_name: list(records) for component_name in components_to_update}


class _EchoReflectionLM:
    def __call__(self, prompt: str) -> str:
        match = re.search(
            r"Current text:\n```text\n(.*?)\n```",
            prompt,
            re.DOTALL,
        )
        assert match is not None
        return match.group(1).strip()


class _AlwaysImproveReflectionLM:
    def __call__(self, prompt: str) -> str:
        return "improved"


def test_optimizer_rejects_non_improving_proposals() -> None:
    trainset = ["train-1", "train-2"]
    valset = ["val-1", "val-2"]

    adapter = _TrackingAdapter()
    state = optimize(
        seed_candidate={"prompt": "seed"},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=_EchoReflectionLM(),
        config=MiniGEPAConfig(
            num_iterations=1,
            minibatch_size=2,
            seed=0,
        ),
    )

    assert len(state.program_candidates) == 1
    assert state.history[0]["accepted"] is False

    val_calls = [
        call
        for call in adapter.calls
        if call[0] == tuple(valset)
    ]
    assert val_calls == [(tuple(valset), "seed", False)]


def test_optimizer_accepts_strict_improvement_and_full_evaluates_it() -> None:
    trainset = ["train-1", "train-2"]
    valset = ["val-1", "val-2", "val-3"]

    adapter = _TrackingAdapter()
    state = optimize(
        seed_candidate={"prompt": "seed"},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=_AlwaysImproveReflectionLM(),
        config=MiniGEPAConfig(
            num_iterations=1,
            minibatch_size=2,
            seed=0,
        ),
    )

    assert len(state.program_candidates) == 2
    assert state.history[0]["accepted"] is True
    assert state.best_candidate["prompt"] == "improved"

    assert (
        tuple(valset),
        "improved",
        False,
    ) in adapter.calls
