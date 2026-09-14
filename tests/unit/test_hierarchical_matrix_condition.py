import pytest

from botcolosseo.training.hierarchical_protocol import (
    ControlCondition,
    matrix_condition,
    require_neutral_matrix,
)


def test_legacy_games_are_neutral_hard():
    assert matrix_condition({}) == ControlCondition()
    require_neutral_matrix({})
    require_neutral_matrix({"condition": [0, 0, 0, 1]})


@pytest.mark.parametrize("values", [[1, 0, 0, 1], [0, 0, 0, 0.5]])
def test_neutral_consumers_reject_conditioned_games(values):
    assert matrix_condition({"condition": values}).as_tuple() == tuple(values)
    with pytest.raises(ValueError, match="Neutral/Hard"):
        require_neutral_matrix({"condition": values})


@pytest.mark.parametrize("values", [[1, 0, 0], [0, 0, 0, 2], [float("nan"), 0, 0, 1], None])
def test_invalid_matrix_conditions(values):
    with pytest.raises(ValueError):
        matrix_condition({"condition": values})
