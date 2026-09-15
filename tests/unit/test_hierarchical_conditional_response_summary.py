import pytest

from botcolosseo.evaluation.hierarchical_conditional_response import summarize


def report():
    conditions = [[0, 0, 0, 1], [1, 0, 0, 1]]
    return {
        "complete": True,
        "identity": {"conditions": conditions, "seeds": [108, 109],
                     "repeats": 1, "role": "host"},
        "cases": [
            {"condition_index": c, "condition": condition, "seed": seed,
             "layout": seed, "repeat": 0, "method": method,
             "opponent_index": 1, "baseline_index": 0,
             "payoff": payoff, "payoffs": [payoff, 0.2]}
            for c, condition in enumerate(conditions) for seed in (108, 109)
            for method, payoff in (("baseline", 0.2), ("candidate", 0.4))
        ],
    }


def test_paired_gain_clusters_layouts_not_conditions():
    result = summarize(report())
    assert result["layout_clusters"] == 2
    assert result["overall"]["gain"] == pytest.approx(0.2)
    assert result["overall"]["gain_ci95"] == pytest.approx([0.2, 0.2])
    assert result["overall"]["improvement_supported"]


@pytest.mark.parametrize("error", ["missing", "opponent", "condition", "payoff"])
def test_reject_invalid_comparisons(error):
    data = report()
    if error == "missing":
        data["cases"].pop()
    elif error == "opponent":
        data["cases"][0]["opponent_index"] = 0
    elif error == "condition":
        data["cases"][0]["condition"] = [0, 1, 0, 1]
    else:
        data["cases"][0]["payoff"] = float("nan")
    with pytest.raises(ValueError):
        summarize(data)
