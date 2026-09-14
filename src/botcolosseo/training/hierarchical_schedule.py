"""Balanced training command contexts, independent of environment seed parity."""

import random


def replay_index(episode_index: int, count: int) -> int:
    """Visit every replay episode once per independently shuffled cycle."""
    if episode_index < 0 or count <= 0:
        raise ValueError("Invalid replay index or size")
    cycle, offset = divmod(episode_index, count)
    order = list(range(count))
    random.Random(2718 + cycle).shuffle(order)
    return order[offset]


def training_context(episode_index: int) -> tuple[str, int]:
    """Each 24 episodes covers profile x search offset x endpoint x role.

    Adjacent host/opponent cases share a context. The six command seeds encode
    all combinations of seed%3 search offsets and seed%2 extraction endpoints.
    Stateless block shuffling makes resume exact without an extra RNG stream.
    Environment layout selection is deliberately not an input.
    """
    if episode_index < 0:
        raise ValueError("Episode index must be nonnegative")
    pair = episode_index // 2
    block, index = divmod(pair, 12)
    contexts = [(profile, seed) for profile in ("search", "combat") for seed in range(6)]
    random.Random(1701 + block).shuffle(contexts)
    return contexts[index]
