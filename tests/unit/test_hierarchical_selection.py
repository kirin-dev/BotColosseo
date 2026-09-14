import pytest

from botcolosseo.evaluation.hierarchical_selection import screen_value


def report():
    return {"complete": True, "identity": {"seeds": [108], "repeats": 1},
            "cases": [{"seed": 108, "repeat": 0, "first_side": "host", "first_payoff": 0.2},
                      {"seed": 108, "repeat": 0, "first_side": "opponent", "first_payoff": 0.9}]}


def test_role_specific_screening():
    assert screen_value({0: report()}, role="host", opponent_mixture=[1])["mean_payoff"] == 0.2
    assert screen_value({0: report()}, role="opponent", opponent_mixture=[1])["mean_payoff"] == 0.9


def test_screen_rejects_missing_role_or_support():
    missing = report()
    missing["cases"] = missing["cases"][1:]
    with pytest.raises(ValueError):
        screen_value({0: missing}, role="host", opponent_mixture=[1])
    with pytest.raises(ValueError):
        screen_value({0: report()}, role="host", opponent_mixture=[0.5, 0.5])
