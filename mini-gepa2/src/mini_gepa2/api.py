from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .adapter import Adapter, Candidate
from .engine import MiniGEPAEngine
from .reflection import ReflectionLM
from .sampler import ShuffledBatchSampler
from .selectors import ParetoCandidateSelector, RoundRobinComponentSelector
from .state import MiniGEPAState


@dataclass(slots=True)
class MiniGEPAConfig:
    """Small configuration surface for the teaching implementation."""

    num_iterations: int = 6
    minibatch_size: int = 3
    seed: int = 0


def optimize(
    seed_candidate: Candidate,
    trainset: Sequence[Any],
    valset: Sequence[Any],
    adapter: Adapter,
    reflection_lm: ReflectionLM,
    config: MiniGEPAConfig | None = None,
) -> MiniGEPAState:
    """Run the teaching version of GEPA.

    This wrapper intentionally fixes most policy choices so the user can focus on the
    main control flow instead of a large configuration graph.
    """

    if config is None:
        config = MiniGEPAConfig()

    rng = random.Random(config.seed)

    engine = MiniGEPAEngine(
        adapter=adapter,
        reflection_lm=reflection_lm,
        trainset=list(trainset),
        valset=list(valset),
        seed_candidate=dict(seed_candidate),
        num_iterations=config.num_iterations,
        batch_sampler=ShuffledBatchSampler(
            batch_size=config.minibatch_size,
            rng=rng,
        ),
        candidate_selector=ParetoCandidateSelector(rng=rng),
        component_selector=RoundRobinComponentSelector(),
    )
    return engine.run()
