import pytest

from botcolosseo.training.pfsp import pfsp_probabilities, stable_uniform


def test_pfsp_uses_frozen_quadratic_weights_and_floor() -> None:
    probabilities = pfsp_probabilities({"easy": 1.0, "hard": 0.0, "mid": 0.5})

    total = 1.0 + 0.25 + 0.05
    assert probabilities == {
        "easy": pytest.approx(0.05 / total),
        "hard": pytest.approx(1.0 / total),
        "mid": pytest.approx(0.25 / total),
    }
    assert list(probabilities) == ["easy", "hard", "mid"]


@pytest.mark.parametrize("win_rates", [{}, {"bad": -0.1}, {"bad": 1.1}])
def test_pfsp_rejects_missing_or_invalid_payoffs(win_rates: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        pfsp_probabilities(win_rates)


def test_stable_uniform_is_stream_specific_and_repeatable() -> None:
    arguments = {
        "master_seed": 20260721,
        "pair_slot": 17,
        "pool_hash": "a" * 64,
        "payoff_hash": "b" * 64,
    }
    first = stable_uniform(**arguments, stream="source")
    second = stable_uniform(**arguments, stream="source")

    assert first == second
    assert 0.0 <= first < 1.0
    assert first != stable_uniform(**arguments, stream="opponent")
