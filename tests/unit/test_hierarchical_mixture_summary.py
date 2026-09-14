import pytest

from botcolosseo.evaluation.hierarchical_mixture_comparison import summarize_comparison


def test_paired_role_payoffs_and_layout_bootstrap():
    report = {"complete": True, "identity": {"seeds": [1, 2], "repeats": 2}, "cases": []}
    for s in (1, 2):
        for repeat in (0, 1):
            for role in ("host", "opponent"):
                for method, value in (("fixed", 0.2), ("uniform", 0.3), ("meta", 0.4)):
                    report["cases"].append(
                        {
                            "seed": s,
                            "layout": s,
                            "repeat": repeat,
                            "role": role,
                            "method": method,
                            "opponent_index": repeat,
                            "payoff": value,
                            "payoffs": [value, value],
                        }
                    )
    result = summarize_comparison(report, draws=20)
    assert result["layout_clusters"] == 2
    assert result["methods"]["meta"]["games"] == 8
    assert result["paired_differences"]["meta_minus_fixed"][
        "layout_bootstrap_95_interval"
    ] == pytest.approx([0.2, 0.2])
    report["cases"][0]["opponent_index"] = 9
    with pytest.raises(ValueError, match="opponent"):
        summarize_comparison(report)


def test_partial_comparison_is_not_reported_as_final():
    with pytest.raises(ValueError, match="completed"):
        summarize_comparison({"complete": False})
