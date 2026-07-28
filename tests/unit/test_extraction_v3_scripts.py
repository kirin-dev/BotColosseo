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


def test_style_selector_rebuilds_ranking_after_new_stage_candidates() -> None:
    source = Path(
        "scripts/run_extraction_v3_select_style.sh"
    ).read_text(encoding="utf-8")

    assert 'ranking_tmp="$EVAL_ROOT/.ranking.json.tmp"' in source
    assert '--output "$ranking_tmp"' in source
    assert 'mv "$ranking_tmp" "$ranking"' in source


def test_aggressive_calibration_runner_is_isolated_and_staged() -> None:
    source = Path(
        "scripts/run_extraction_v3_aggressive_calibration.sh"
    ).read_text(encoding="utf-8")

    assert 'STOP_AFTER_STEPS:-100000' in source
    assert "STOP_AFTER_STEPS > 200000" in source
    assert 'OUTPUT="runs/extraction/styles/aggressive-calibration-v2"' in source
    assert 'PARENT="runs/extraction/styles/aggressive/candidate-0600000.pt"' in source
    assert '--initialize-from "$PARENT"' in source
    assert '--resume "$OUTPUT/latest.pt"' in source


def test_style_selector_accepts_an_isolated_output_root() -> None:
    source = Path(
        "scripts/run_extraction_v3_select_style.sh"
    ).read_text(encoding="utf-8")

    assert "BOTCOLOSSEO_STYLE_OUTPUT:-runs/extraction/styles/$STYLE" in source
