from botcolosseo.cli.select_defensive import _compare


def _candidate(alpha: float, shift: float, retention: float) -> dict[str, object]:
    return {
        "alpha": alpha,
        "protective_presence_delta": shift,
        "skill_retention": retention,
    }


def test_defensive_ranking_prefers_clear_shift_then_retention_then_lower_alpha() -> None:
    assert _compare(_candidate(0.75, 0.30, 0.86), _candidate(0.25, 0.20, 1.0)) < 0
    assert _compare(_candidate(0.25, 0.30, 0.90), _candidate(0.50, 0.27, 0.95)) > 0
    assert _compare(_candidate(0.25, 0.30, 0.95), _candidate(0.50, 0.27, 0.95)) < 0
