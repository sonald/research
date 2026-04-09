from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from .state import MiniGEPAState


@dataclass(slots=True)
class ParetoCandidateSelector:
    """Pick a parent from the current Pareto-style candidate pool.

    The official implementation also removes dominated programs before sampling.
    The teaching version stops one step earlier: we simply count how often each
    candidate appears on the per-example frontier, then sample proportionally to
    that frequency. This preserves the key intuition without extra machinery.
    """

    rng: random.Random

    def select_candidate_id(self, state: MiniGEPAState) -> int:
        counts: Counter[int] = Counter()
        for candidate_ids in state.pareto_front_candidates.values():
            for candidate_id in candidate_ids:
                counts[candidate_id] += 1

        weighted_population = [
            candidate_id
            for candidate_id, frequency in counts.items()
            for _ in range(frequency)
        ]
        return self.rng.choice(weighted_population)


class RoundRobinComponentSelector:
    """Cycle through candidate components one at a time.

    This is the simplest way to keep multi-component optimization visible. Each
    candidate carries its own cursor, so descendants can continue mutating from
    where their own lineage left off.
    """

    def select_component(self, state: MiniGEPAState, candidate_id: int) -> str:
        component_index = state.next_component_index_by_candidate[candidate_id]
        component_names = state.component_names
        state.next_component_index_by_candidate[candidate_id] = (
            component_index + 1
        ) % len(component_names)
        return component_names[component_index]
