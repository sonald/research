from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

# In the official project a candidate can contain several named text components.
# We keep that shape because it is a useful abstraction even in a small teaching build.
Candidate = dict[str, str]

# A reflective record is the JSON-like unit that the adapter hands to the reflection LM.
# The optimizer does not care about the exact schema beyond "it must be serializable enough
# to show to the LM". That loose contract is one of GEPA's strengths.
ReflectiveRecord = dict[str, Any]


@dataclass(slots=True)
class EvaluationBatch:
    """Outputs from evaluating one candidate on one batch of examples.

    The teaching version keeps only the three fields that the core loop truly needs:

    - outputs: raw task outputs
    - scores: numeric rewards, where larger is better
    - trajectories: optional per-example traces used to build reflection input
    """

    outputs: list[Any]
    scores: list[float]
    trajectories: list[Any] | None = None


class Adapter(Protocol):
    """Task-specific boundary between the optimizer and the world being optimized."""

    def evaluate(
        self,
        batch: list[Any],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Run the candidate on a batch and return outputs, scores, and optional traces."""

    def make_reflective_dataset(
        self,
        candidate: Candidate,
        evaluation: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[ReflectiveRecord]]:
        """Turn traces into a small reflection dataset for each requested component."""
