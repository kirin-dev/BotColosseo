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
