import pytest

from botcolosseo.cli.solve_hierarchical_matrix import solve_report


def report():
    return {"complete": True, "row_role": "host", "column_role": "opponent",
            "identity": {"strategies": ["a", "b"], "seeds": [96], "repeats": 1},
            "A": [[1, 0], [0, 1]], "B": [[0, 1], [1, 0]], "counts": [[2, 2], [2, 2]]}


def test_bound_matrix_solution():
    result = solve_report(report())
    assert result["host"] == pytest.approx([0.5, 0.5])
    assert result["opponent"] == pytest.approx([0.5, 0.5])
    assert result["maximum_regret"] <= 1e-6


def test_distinct_role_populations():
    data = report()
    data["identity"].update(host_strategies=["a", "c"], opponent_strategies=["b"])
    data.update(A=[[0.2], [0.5]], B=[[0.4], [0.1]], counts=[[2], [2]])
    result = solve_report(data)
    assert result["host"] == pytest.approx([0, 1])
    assert result["opponent"] == pytest.approx([1])


def test_reject_incomplete_or_invalid_payoff():
    for change in ({"complete": False}, {"counts": [[1, 2], [2, 2]]},
                   {"A": [[2, 0], [0, 1]]}, {"row_role": "first"}):
        with pytest.raises(ValueError):
            solve_report(report() | change)
