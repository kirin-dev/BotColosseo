import copy

import numpy as np
import pytest

from botcolosseo.training.hierarchical_conditional_population import conditional_marginals
from botcolosseo.training.hierarchical_protocol import ControlCondition


def solutions():
    return [
        {
            "identity": {"executor": "low", "strategies": ["a", "b"],
                         "condition": list(c.as_tuple())},
            "maximum_regret": 0,
            "host": h, "opponent": o,
        }
        for c, h, o in [
            (ControlCondition(), [1, 0], [0, 1]),
            (ControlCondition(aggressive=1), [0, 1], [1, 0]),
        ]
    ]


@pytest.mark.parametrize("role,expected", [
    ("host", [[0, 1], [1, 0]]), ("opponent", [[1, 0], [0, 1]])
])
def test_condition_and_role_select_the_correct_target(role, expected):
    actual = conditional_marginals(
        solutions()[::-1], [ControlCondition(), ControlCondition(aggressive=1)],
        executor="low", population=["a", "b"], learner_role=role,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "executor", "order", "nan", "prob"])
def test_reject_incompatible_condition_targets(mutation):
    data = copy.deepcopy(solutions())
    if mutation == "missing":
        data.pop()
    elif mutation == "duplicate":
        data.append(copy.deepcopy(data[0]))
    elif mutation == "executor":
        data[0]["identity"]["executor"] = "another"
    elif mutation == "order":
        data[0]["identity"]["strategies"].reverse()
    elif mutation == "nan":
        data[0]["maximum_regret"] = float("nan")
    else:
        data[0]["opponent"] = [0, 0]
    with pytest.raises(ValueError):
        conditional_marginals(
            data, [ControlCondition(), ControlCondition(aggressive=1)],
            executor="low", population=["a", "b"], learner_role="host",
        )
