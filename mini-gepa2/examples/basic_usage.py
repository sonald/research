from __future__ import annotations

"""Small usage example for the teaching-oriented mini-gepa2 package.

Run it from the mini-gepa2 directory:

    PYTHONPATH=src python examples/basic_usage.py
"""

from mini_gepa2.api import MiniGEPAConfig, optimize
from mini_gepa2.demo import (
    ToyReflectionLM,
    ToyWordTaskAdapter,
    build_demo_examples,
    build_seed_candidate,
)


def main() -> None:
    # 1. Prepare a seed candidate.
    seed_candidate = build_seed_candidate()

    # 2. Prepare train and validation examples.
    trainset, valset = build_demo_examples()

    # 3. Plug in an adapter and a reflection LM.
    adapter = ToyWordTaskAdapter()
    reflection_lm = ToyReflectionLM()

    # 4. Configure the teaching loop.
    config = MiniGEPAConfig(
        num_iterations=6,
        minibatch_size=2,
        seed=0,
    )

    # 5. Run optimization.
    state = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        config=config,
    )

    # 6. Inspect the result.
    print("Seed validation average:", state.average_val_score(0))
    print("Best validation average:", state.average_val_score(state.best_candidate_id))
    print("Best candidate id:", state.best_candidate_id)
    print("\nBest prompt:\n")
    print(state.best_candidate["prompt"])


if __name__ == "__main__":
    main()
