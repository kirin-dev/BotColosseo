import pytest

from botcolosseo.cli.evaluate_hierarchical_executor import evaluate_case


@pytest.mark.parametrize("difficulty", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_difficulty_is_rejected_before_environment_creation(difficulty):
    with pytest.raises(ValueError, match="Difficulty"):
        evaluate_case(
            None, 0, "host", "cpu", expected_scenario="unused", difficulty=difficulty
        )
