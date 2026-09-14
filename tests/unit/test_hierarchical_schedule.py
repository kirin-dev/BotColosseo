from collections import Counter

import pytest

from botcolosseo.training.hierarchical_schedule import replay_index, training_context


def test_replay_cycles_cover_all_items_and_resume_exactly():
    for cycle in range(3):
        order = [replay_index(i, 370) for i in range(cycle * 370, (cycle + 1) * 370)]
        assert sorted(order) == list(range(370))
    first = [replay_index(i, 370) for i in range(129)]
    assert any(i >= 338 for i in first)  # Late-added corrections are sampled early.
    assert [replay_index(i, 370) for i in range(60, 129)] == first[60:]
    with pytest.raises(ValueError):
        replay_index(0, 0)


def test_every_block_covers_profiles_offsets_endpoints_and_roles():
    for block in range(6):
        counts = Counter()
        for episode in range(24 * block, 24 * (block + 1)):
            profile, seed = training_context(episode)
            counts[profile, seed % 3, seed % 2, episode % 2] += 1
            assert training_context(episode // 2 * 2) == training_context(episode)
        assert len(counts) == 24
        assert set(counts.values()) == {1}


def test_context_is_resume_stable_and_rejects_invalid_index():
    expected = [training_context(i) for i in range(122)]
    assert [training_context(i) for i in range(61, 122)] == expected[61:]
    assert {seed % 2 for profile, seed in expected if profile == "search"} == {0, 1}
    with pytest.raises(ValueError):
        training_context(-1)
