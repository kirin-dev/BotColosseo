from pathlib import Path


def test_downstream_style_runners_use_fail_closed_aggressive_prerequisite() -> None:
    for script in (
        Path("scripts/run_extraction_v3_style.sh"),
        Path("scripts/run_extraction_v3_select_style.sh"),
    ):
        source = script.read_text(encoding="utf-8")

        assert (
            "botcolosseo.cli.check_extraction_aggressive_prerequisite" in source
        )


def test_style_runner_defaults_to_resumable_200k_stage() -> None:
    source = Path("scripts/run_extraction_v3_style.sh").read_text(encoding="utf-8")

    assert 'BOTCOLOSSEO_STOP_AFTER_STEPS:-200000' in source
    assert '--stop-after-steps "$STOP_AFTER_STEPS"' in source
    assert "STOP_AFTER_STEPS > 600000" in source
    assert 'int(sys.argv[3]) <= summary.get("environment_steps", 0)' in source
    assert '[[ ! -f "$OUTPUT/latest.pt" ]]' in source


def test_strong_runner_defaults_to_resumable_600k_stage() -> None:
    source = Path("scripts/run_extraction_v3_strong.sh").read_text(encoding="utf-8")

    assert 'BOTCOLOSSEO_STOP_AFTER_STEPS:-600000' in source
    assert '--stop-after-steps "$STOP_AFTER_STEPS"' in source
    assert "STOP_AFTER_STEPS > 2000000" in source
    assert 'int(sys.argv[3]) <= summary.get("environment_steps", 0)' in source


def test_style_runner_resolves_research_or_product_strong_base() -> None:
    source = Path("scripts/run_extraction_v3_style.sh").read_text(encoding="utf-8")

    assert "botcolosseo.cli.resolve_extraction_strong_artifact" in source
    assert "--field checkpoint" in source
    assert 'BASE="runs/extraction/strong-ppo/selected.pt"' not in source


def test_style_selector_rebuilds_ranking_after_new_stage_candidates() -> None:
    source = Path(
        "scripts/run_extraction_v3_select_style.sh"
    ).read_text(encoding="utf-8")

    assert 'ranking_tmp="$EVAL_ROOT/.ranking.json.tmp"' in source
    assert '--output "$ranking_tmp"' in source
    assert 'mv "$ranking_tmp" "$ranking"' in source


def test_style_selector_resolves_strong_product_or_research_evidence() -> None:
    source = Path(
        "scripts/run_extraction_v3_select_style.sh"
    ).read_text(encoding="utf-8")

    assert source.count("botcolosseo.cli.resolve_extraction_strong_artifact") == 3
    assert "--field checkpoint" in source
    assert "--field validation_report" in source
    assert "--field heldout_report" in source
    assert 'STRONG_SELECTION="runs/extraction/strong-ppo/selection.json"' not in source


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


def test_defensive_calibration_runner_is_fresh_isolated_and_staged() -> None:
    source = Path(
        "scripts/run_extraction_v3_defensive_calibration.sh"
    ).read_text(encoding="utf-8")

    assert 'STOP_AFTER_STEPS:-200000' in source
    assert 'OUTPUT="runs/extraction/styles/defensive-calibration-v2"' in source
    assert "styles-defensive-calibration.yaml" in source
    assert '--resume "$OUTPUT/latest.pt"' in source
    assert "--initialize-from" not in source


def test_directional_style_admission_runner_freezes_source_and_destination() -> None:
    source = Path(
        "scripts/run_extraction_v3_admit_style_showcase.sh"
    ).read_text(encoding="utf-8")

    assert "BOTCOLOSSEO_STYLE_SOURCE" in source
    assert 'DESTINATION="runs/extraction/styles/$STYLE"' in source
    assert "botcolosseo.cli.admit_extraction_showcase" in source
    assert '--policy "$STYLE"' in source
    assert source.count("botcolosseo.cli.resolve_extraction_strong_artifact") == 3
    assert 'BASE="runs/extraction/strong-ppo/selected.pt"' not in source


def test_validation_demonstration_runner_freezes_source_and_disclosure_tier() -> None:
    source = Path(
        "scripts/run_extraction_v3_create_style_demonstration.sh"
    ).read_text(encoding="utf-8")

    assert "BOTCOLOSSEO_STYLE_SOURCE" in source
    assert 'DESTINATION="runs/extraction/styles/$STYLE"' in source
    assert "botcolosseo.cli.create_extraction_showcase_demonstration" in source
    assert "showcase-demonstration.json" in source
    assert source.count("botcolosseo.cli.resolve_extraction_strong_artifact") == 3
    assert 'STRONG_SELECTION="runs/extraction/strong-ppo/selection.json"' not in source


def test_showcase_runner_binds_manifests_retries_and_product_audits() -> None:
    source = Path(
        "scripts/render_extraction_v3_showcase.sh"
    ).read_text(encoding="utf-8")

    assert "--aggressive-manifest" in source
    assert "--defensive-manifest" in source
    assert "--explorer-manifest" in source
    assert "--max-attempts 5" in source
    assert "botcolosseo.cli.audit_extraction_showcase" in source
    assert source.count("botcolosseo.cli.resolve_extraction_strong_artifact") == 3
    assert "--strong-manifest \"$STRONG_MANIFEST\"" in source
    assert 'BASE="runs/extraction/strong-ppo/selected.pt"' not in source


def test_aggressive_admission_resolves_product_or_research_strong_evidence() -> None:
    source = Path(
        "scripts/run_extraction_v3_admit_aggressive_showcase.sh"
    ).read_text(encoding="utf-8")

    assert source.count("botcolosseo.cli.resolve_extraction_strong_artifact") == 3
    assert "--field checkpoint" in source
    assert "--field validation_report" in source
    assert "--field heldout_report" in source
    assert 'STRONG_SELECTION="runs/extraction/strong-ppo/selection.json"' not in source


def test_style_selector_accepts_an_isolated_output_root() -> None:
    source = Path(
        "scripts/run_extraction_v3_select_style.sh"
    ).read_text(encoding="utf-8")

    assert "BOTCOLOSSEO_STYLE_OUTPUT:-runs/extraction/styles/$STYLE" in source
