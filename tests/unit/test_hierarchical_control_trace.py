import pytest

from botcolosseo.evaluation.hierarchical_controls import summarize_control_trace


def test_independent_low_and_high_latency_and_unreached_events():
    schedule = [(0, (0, 0, 0, 1)), (3, (1, 0, 0, 0.5)), (20, (0, 1, 0, 1))]
    trace = [
        {
            "decision": i,
            "requested_style": [int(i >= 3), 0, 0],
            "requested_difficulty": 1 if i < 3 else 0.5,
            "style": [int(i >= 8), 0, 0],
            "difficulty": 1 if i < 3 else 0.5,
            "high_difficulty": 1 if i < 8 else 0.5,
            "replanned": i in (0, 8),
        }
        for i in range(10)
    ]
    result = summarize_control_trace(trace, schedule)
    event = result["events"][0]
    assert event["style"]["delay_decisions"] == 5
    assert event["low_difficulty"]["delay_decisions"] == 0
    assert event["high_difficulty"]["delay_decisions"] == 5
    assert result["events"][1]["style"]["status"] == "not_reached"
    trace[3]["requested_difficulty"] = 1
    with pytest.raises(ValueError, match="requests"):
        summarize_control_trace(trace, schedule)
