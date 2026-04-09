from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .adapter import Adapter, Candidate, EvaluationBatch
from .api import MiniGEPAConfig, optimize


@dataclass(slots=True)
class ToyWordExample:
    """Simple example used by the included deterministic demo."""

    word: str
    label: str


def _parse_rules(prompt_text: str) -> dict[str, str]:
    """Extract `word => label` rules from the prompt text."""

    rules: dict[str, str] = {}
    for line in prompt_text.splitlines():
        if "=>" not in line:
            continue
        left, right = line.split("=>", maxsplit=1)
        rules[left.strip()] = right.strip()
    return rules


class ToyWordTaskAdapter(Adapter):
    """A tiny task adapter for demonstrating the GEPA loop.

    The candidate contains one component called `prompt`. Inside that prompt we keep
    a growing list of explicit rules such as `banana => fruit`. The task is trivial,
    but it makes every step of the GEPA loop visible:

    - the prompt is the text we mutate
    - evaluation turns text into predictions
    - trajectories explain which examples failed
    - reflection proposes a better prompt
    """

    def evaluate(
        self,
        batch: list[ToyWordExample],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        rules = _parse_rules(candidate["prompt"])

        outputs: list[str] = []
        scores: list[float] = []
        trajectories: list[dict[str, str]] | None = [] if capture_traces else None

        for example in batch:
            prediction = rules.get(example.word, "unknown")
            score = 1.0 if prediction == example.label else 0.0

            outputs.append(prediction)
            scores.append(score)

            if trajectories is not None:
                trajectories.append(
                    {
                        "input": example.word,
                        "prediction": prediction,
                        "expected_label": example.label,
                        "feedback": (
                            "Correct."
                            if score == 1.0
                            else f'The word "{example.word}" should be labeled "{example.label}".'
                        ),
                    }
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: Candidate,
        evaluation: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, object]]]:
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


class ToyReflectionLM:
    """A deterministic stand-in for the reflection model.

    It reads the current prompt and the reflective records from the rendered prompt,
    then appends rules for every mistake it sees. A real LLM would infer higher-level
    strategies; the teaching version only needs a visible mutation operator.
    """

    _current_text_re = re.compile(
        r"Current text:\n```text\n(.*?)\n```",
        re.DOTALL,
    )
    _records_re = re.compile(
        r"Reflective records gathered from recent rollouts:\n```json\n(.*?)\n```",
        re.DOTALL,
    )

    def __call__(self, prompt: str) -> str:
        current_text_match = self._current_text_re.search(prompt)
        records_match = self._records_re.search(prompt)

        current_text = current_text_match.group(1).strip()
        reflective_records = json.loads(records_match.group(1))

        rules = _parse_rules(current_text)
        additions: list[str] = []

        for record in reflective_records:
            word = record["input"]
            expected_label = record["expected_label"]
            score = record["score"]

            if score < 1.0 and rules.get(word) != expected_label:
                rules[word] = expected_label
                additions.append(f"{word} => {expected_label}")

        if not additions:
            return current_text

        return current_text.rstrip() + "\n" + "\n".join(additions)


def build_demo_examples() -> tuple[list[ToyWordExample], list[ToyWordExample]]:
    """Return train and validation sets for the deterministic toy task."""

    trainset = [
        ToyWordExample(word="cat", label="animal"),
        ToyWordExample(word="banana", label="fruit"),
        ToyWordExample(word="hammer", label="tool"),
        ToyWordExample(word="apple", label="fruit"),
        ToyWordExample(word="dog", label="animal"),
        ToyWordExample(word="wrench", label="tool"),
    ]

    # The demo keeps validation simple on purpose. Using the same examples is enough to
    # show how the GEPA loop accepts proposals and updates the Pareto frontier.
    valset = list(trainset)
    return trainset, valset


def build_seed_candidate() -> Candidate:
    """Return a deliberately incomplete starting prompt."""

    return {
        "prompt": (
            "Classify each word as animal, fruit, or tool.\n"
            "Known rules:\n"
            "cat => animal"
        )
    }


def main() -> None:
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

    print("Seed validation average:", state.average_val_score(0))
    print("Best validation average:", state.average_val_score(state.best_candidate_id))
    print("Best candidate id:", state.best_candidate_id)
    print("\nBest prompt:\n")
    print(state.best_candidate["prompt"])


if __name__ == "__main__":
    main()
