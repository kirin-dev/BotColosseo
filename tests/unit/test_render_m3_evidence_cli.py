import json
from pathlib import Path

import pytest

import botcolosseo.cli.render_m3_evidence as render_cli


def test_renderer_accepts_integrity_audited_capability_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit_calls: list[tuple[Path, dict[str, object]]] = []

    def audit(path: Path, **kwargs: object) -> dict[str, object]:
        audit_calls.append((path, kwargs))
        return {
            "integrity_passed": True,
            "capability_passed": False,
            "passed": False,
        }

    monkeypatch.setattr(render_cli, "audit_m3_evidence", audit)
    monkeypatch.setattr(
        render_cli,
        "render_evidence_bundle",
        lambda **kwargs: {"executed_rows": 360},
    )
    argv = [
        "--official-report-dir",
        str(tmp_path / "official"),
        "--crossplay-csv",
        str(tmp_path / "crossplay.csv"),
        "--pool-history",
        str(tmp_path / "pool-history.json"),
        "--heatmap-output",
        str(tmp_path / "heatmap.png"),
        "--pool-output",
        str(tmp_path / "pool.png"),
        "--matrix-output",
        str(tmp_path / "matrix.json"),
    ]

    assert render_cli.main(argv) == 0

    assert audit_calls[0][1]["require_capability_pass"] is False
    output = json.loads(capsys.readouterr().out)
    assert output["audit"]["integrity_passed"] is True
    assert output["audit"]["capability_passed"] is False
    assert output["matrix"]["executed_rows"] == 360
