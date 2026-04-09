from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(slots=True)
class ShuffledBatchSampler:
    """Small sequential sampler with per-epoch reshuffling.

    The full repository has a richer sampler abstraction. For teaching we only need
    one idea: each GEPA step sees a small minibatch instead of the full train set.
    """

    batch_size: int
    rng: random.Random
    _order: list[int] = field(default_factory=list)
    _cursor: int = 0

    def _reshuffle(self, dataset_size: int) -> None:
        self._order = list(range(dataset_size))
        self.rng.shuffle(self._order)
        self._cursor = 0

    def next_batch_indices(self, dataset_size: int) -> list[int]:
        """Return the next minibatch of example indices.

        We keep the implementation intentionally direct:

        - the order is shuffled once per epoch
        - we read examples sequentially from that order
        - when we run out, we reshuffle and continue
        """

        if not self._order:
            self._reshuffle(dataset_size)

        batch: list[int] = []
        target_size = min(self.batch_size, dataset_size)

        while len(batch) < target_size:
            if self._cursor >= len(self._order):
                self._reshuffle(dataset_size)
            batch.append(self._order[self._cursor])
            self._cursor += 1

        return batch
