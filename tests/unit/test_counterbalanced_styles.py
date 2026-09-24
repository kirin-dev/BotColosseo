from itertools import permutations

import pytest

from botcolosseo.demo.hierarchical_controls import control_schedule
from botcolosseo.evaluation.counterbalanced_styles import STYLES, summarize


def reports():
    result = []
    for order in permutations(STYLES):
        schedule = [(t, c.as_tuple()) for t, c in control_schedule(style_order=order)]
        trace = []
        for i in range(321):
            requested = next(c for t, c in reversed(schedule) if t <= i)
            if i % 8 == 0:
                applied = requested
            command = 3 if applied[0] else 5 if applied[1] else 0
            trace.append(
                dict(
                    decision=i,
                    replanned=i % 8 == 0,
                    requested_style=requested[:3],
                    requested_difficulty=1,
                    style=applied[:3],
                    high_difficulty=1,
                    difficulty=1,
                    command=command,
                    action=9 if applied[0] else 1,
                )
            )
        result.append(
            dict(
                complete=True,
                switch_mode="style",
                difficulty=1,
                executor="low",
                strategy="high",
                opponent="opponent",
                style_order=order,
                control_schedule=schedule,
                cases=[
                    dict(seed=1, role="host", payoff=0.5, decisions=len(trace), control_trace=trace)
                ],
            )
        )
    return result


def test_balanced_summary_does_not_count_orders_as_independent_cases():
    result = summarize(reports())
    assert result["episodes"] == 6 and result["unique_cases"] == 1
    assert result["pre_switch_prefix_identical"]
    assert result["phases"]["phase_1_aggressive"]["attack_fraction"] == 1
    assert result["phases"]["phase_2_defensive"]["extract_fraction"] == 1
    assert result["phases"]["phase_3_explorer"]["search_fraction"] == 1


def test_prefix_difference_is_reported_not_hidden():
    data = reports()
    data[1]["cases"][0]["control_trace"][0]["action"] = 9
    result = summarize(data)
    assert not result["pre_switch_prefix_identical"]
    assert sum(r["action_differences"] for r in result["prefix_comparisons"]) == 1


def test_missing_order_is_rejected():
    with pytest.raises(ValueError, match="All six"):
        summarize(reports()[:-1])
