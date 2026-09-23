import pytest

from botcolosseo.cli.evaluate_hierarchical_styles import finalize_report
from botcolosseo.demo.control_trace_audit import audit_control_report
from botcolosseo.demo.hierarchical_controls import control_schedule


def report():
    schedule = [[t, list(c.as_tuple())] for t, c in control_schedule("joint")]
    trace = []
    high = None
    for i in range(250):
        condition = next(c for t, c in reversed(schedule) if t <= i)
        if i % 8 == 0:
            high = condition
        trace.append({"decision": i, "replanned": i % 8 == 0,
                      "requested_style": condition[:3], "requested_difficulty": condition[3],
                      "difficulty": condition[3], "style": high[:3],
                      "high_difficulty": high[3]})
    return {"complete": True, "switch_mode": "joint", "difficulty": 1,
            "control_schedule": schedule, "cases": [{"seed": 1, "role": "host",
            "decisions": len(trace), "control_trace": trace}]}


def test_valid_boundaries():
    assert audit_control_report(report()) == {
        "cases": 1, "reached": 3, "applied": 3,
        "pending_at_episode_end": 0, "max_delay": 7}


def test_finalization_persists_audit_only_after_success():
    data = report()
    data["complete"] = False
    finalize_report(data)
    assert data["complete"]
    assert data["control_audit"]["applied"] == 3


def test_invalid_finalization_preserves_incomplete_status():
    data = report()
    data["complete"] = False
    data["cases"][0]["control_trace"][81]["difficulty"] = 1
    with pytest.raises(ValueError):
        finalize_report(data)
    assert data["complete"] is False
    assert "control_audit" not in data


def test_static_finalization_does_not_require_control_trace():
    data = {"complete": False, "cases": [{"payoff": 0.5}]}
    finalize_report(data)
    assert data == {"complete": True, "cases": [{"payoff": 0.5}]}


@pytest.mark.parametrize("fault", ["request", "low", "high", "gap", "duplicate", "incomplete"])
def test_reject_corruption(fault):
    data = report()
    trace = data["cases"][0]["control_trace"]
    if fault == "request":
        trace[81]["requested_difficulty"] = trace[81]["difficulty"] = 1
    elif fault == "low":
        trace[81]["difficulty"] = 1
    elif fault == "high":
        trace[81]["style"] = [1, 0, 0]
    elif fault == "gap":
        trace[81]["decision"] = 82
    elif fault == "duplicate":
        data["cases"].append(data["cases"][0])
    else:
        data["complete"] = False
    with pytest.raises(ValueError):
        audit_control_report(data)
