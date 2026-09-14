import pytest

from botcolosseo.evaluation.hierarchical_executor import compare_executors, summarize_executor


def test_applicability_denominator_and_missing_commands():
    records = [
        {"command": "ENGAGE", "start": 0, "completed_at": 10, "applicable": True},
        {"command": "ENGAGE", "start": 20, "completed_at": None, "applicable": True},
        {"command": "ENGAGE", "start": 40, "completed_at": None, "applicable": False},
    ]
    report = {
        "complete": True,
        "cases": [{"seed": 96, "side": "host", "banked": 45, "commands": records}],
    }
    result = summarize_executor(report)
    assert result["commands"]["ENGAGE"]["success_rate"] == 0.5
    assert result["commands"]["ENGAGE"]["mean_success_delay_decisions"] == 10
    assert result["commands"]["DISENGAGE"]["success_rate"] is None
    assert result["positive_banked_rate"] == 1
    report["cases"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_executor(report)


def test_partial_report_is_not_final_evidence():
    with pytest.raises(ValueError, match="completed"):
        summarize_executor({"complete": False, "cases": []})


def test_regression_keeps_missing_opportunities_inconclusive():
    import copy

    old = {
        "complete": True,
        "profile": "search",
        "protocol": "test",
        "command_schema": "test",
        "scoring_version": "test",
        "extraction_endpoint": None,
        "cases": [
            {
                "seed": 1,
                "side": "host",
                "scenario_hash": "test",
                "banked": 50,
                "commands": [
                    {
                        "command": "EXTRACT_NORTH",
                        "start": 240,
                        "completed_at": 300,
                        "applicable": True,
                    }
                ],
            }
        ],
    }
    new = copy.deepcopy(old)
    new["cases"][0]["banked"] = 0
    new["cases"][0]["commands"][0]["completed_at"] = None
    result = compare_executors(old, new)
    assert result["extraction_rate_delta"] == -1
    assert result["commands"]["EXTRACT_NORTH"]["common_success_delta"] == -1
    assert result["commands"]["EXTRACT_NORTH"]["descriptive_retention"] is False
    assert result["commands"]["ENGAGE"]["descriptive_retention"] is None
    new["difficulty"] = 0.5
    with pytest.raises(ValueError, match="difficulty"):
        compare_executors(old, new)
    new["difficulty"] = 1.0
    assert compare_executors(old, new)["extraction_rate_delta"] == -1
    new["cases"][0]["seed"] = 2
    with pytest.raises(ValueError, match="matching"):
        compare_executors(old, new)
