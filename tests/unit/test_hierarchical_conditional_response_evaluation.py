import pytest

from botcolosseo.cli.evaluate_hierarchical_conditional_response import draw_pair, main


@pytest.mark.parametrize("role", ["host", "opponent"])
def test_fixed_marginals_and_repeatable_pair_draw(role):
    assert draw_pair(108, 0, role, 0, [0, 1], [1, 0]) == (1, 0)
    first = draw_pair(108, 1, role, 2, [0.3, 0.7], [0.6, 0.4])
    assert first == draw_pair(108, 1, role, 2, [0.3, 0.7], [0.6, 0.4])


def test_reject_layout_aliases_before_loading_artifacts(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "evaluate", "--executor", "missing", "--candidate", "missing",
        "--output", "missing", "--condition-solutions", "missing",
        "--population", "missing", "--role", "host", "--seeds", "108", "236",
    ])
    with pytest.raises(ValueError, match="distinct"):
        main()
