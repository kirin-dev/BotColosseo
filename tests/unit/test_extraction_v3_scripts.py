from pathlib import Path


def test_downstream_style_runners_require_objective_aligned_aggressive_gate() -> None:
    for script in (
        Path("scripts/run_extraction_v3_style.sh"),
        Path("scripts/run_extraction_v3_select_style.sh"),
    ):
        source = script.read_text(encoding="utf-8")

        assert 'selection.get("policy") == "aggressive"' in source
        assert 'selection.get("gate_schema_version") == 2' in source
        assert 'selection.get("eligible") is True' in source


def test_style_runner_exposes_resumable_stage_budget() -> None:
    source = Path("scripts/run_extraction_v3_style.sh").read_text(encoding="utf-8")

    assert 'BOTCOLOSSEO_STOP_AFTER_STEPS:-600000' in source
    assert '--stop-after-steps "$STOP_AFTER_STEPS"' in source
    assert "STOP_AFTER_STEPS > 600000" in source
