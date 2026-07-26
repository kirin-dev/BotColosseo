from __future__ import annotations

import json
import subprocess
from pathlib import Path

from botcolosseo.scenarios.build import (
    BuildSettings,
    build_crystal_run_extraction,
)
from botcolosseo.scenarios.wad import inspect_pwad


class FakeRunner:
    def __call__(self, command, **kwargs):
        command = [str(item) for item in command]
        if len(command) == 1:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="This is version 1.60 (test)\n",
                stderr="",
            )
        Path(command[-1]).write_bytes(b"extraction-behavior")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_build_extraction_scenario_has_isolated_protocol_and_decorate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "map.udmf").write_text('namespace = "zdoom";\n', encoding="utf-8")
    (source / "crystal_run_extraction.acs").write_text(
        "int extraction = 1;\n", encoding="utf-8"
    )
    (source / "decorate.txt").write_text(
        "actor BotExtractionMarker {}\n", encoding="utf-8"
    )
    (source / "regions.yaml").write_text("regions: []\n", encoding="utf-8")
    include = tmp_path / "include"
    include.mkdir()
    (include / "zcommon.acs").write_text("// header\n", encoding="utf-8")
    acc = tmp_path / "acc"
    acc.touch()
    settings = BuildSettings(
        source_dir=source,
        output_wad=tmp_path / "extraction.wad",
        manifest_path=tmp_path / "manifest.json",
        acc_path=acc,
        acc_include=include,
    )

    manifest = build_crystal_run_extraction(settings, runner=FakeRunner())

    entries = inspect_pwad(settings.output_wad.read_bytes())
    assert [entry.name for entry in entries] == [
        "DECORATE",
        "MAP01",
        "TEXTMAP",
        "BEHAVIOR",
        "SCRIPTS",
        "ENDMAP",
    ]
    assert manifest.protocol_version == 3
    assert manifest.maps == ("MAP01",)
    payload = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    assert set(payload["source_sha256"]) == {
        "crystal_run_extraction.acs",
        "decorate.txt",
        "map.udmf",
        "regions.yaml",
    }
