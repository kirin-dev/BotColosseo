import pytest

from botcolosseo.demo.control_trace_audit import audit_control_report
from botcolosseo.demo.hierarchical_controls import control_schedule


@pytest.mark.parametrize("name", ["aggressive", "defensive", "explorer"])
def test_single_style_transition_and_audit(name):
    schedule = [(t, list(c.as_tuple())) for t, c in control_schedule(single_style=name)]
    assert [t for t, _ in schedule] == [0, 81]
    trace = []
    high = schedule[0][1]
    for decision in range(100):
        requested = schedule[decision >= 81][1]
        replan = decision % 8 == 0
        if replan:
            high = requested
        trace.append(dict(decision=decision, replanned=replan,
                          requested_style=requested[:3], requested_difficulty=1,
                          style=high[:3], difficulty=1, high_difficulty=1))
    report = dict(complete=True, switch_mode="style", difficulty=1,
                  single_switch_style=name, control_schedule=schedule,
                  cases=[dict(seed=104, role="host", decisions=100, control_trace=trace)])
    assert audit_control_report(report)["max_delay"] == 7
    report["single_switch_style"] = "defensive" if name != "defensive" else "explorer"
    with pytest.raises(ValueError, match="Schedule"):
        audit_control_report(report)
