import json
import re
import subprocess
from pathlib import Path

import pytest

from botcolosseo.scenarios.build import (
    BuildSettings,
    build_crystal_run,
    resolve_acc,
)
from botcolosseo.scenarios.wad import inspect_pwad


def make_source_tree(root: Path) -> tuple[Path, Path]:
    source_dir = root / "src"
    source_dir.mkdir()
    (source_dir / "map.udmf").write_text('namespace = "zdoom";\n', encoding="utf-8")
    (source_dir / "crystal_run.acs").write_text("int marker = BOTC_TASK_ID;\n", encoding="utf-8")
    (source_dir / "regions.yaml").write_text("regions: []\n", encoding="utf-8")
    (source_dir / "task_variants.yaml").write_text("protocol_version: 1\n", encoding="utf-8")
    include_dir = root / "include"
    include_dir.mkdir()
    (include_dir / "zcommon.acs").write_text("// test header\n", encoding="utf-8")
    return source_dir, include_dir


class FakeRunner:
    def __init__(self, *, fail_task: int | None = None) -> None:
        self.commands: list[list[str]] = []
        self.wrapper_headers: list[str] = []
        self.fail_task = fail_task

    def __call__(self, command, **kwargs):
        command = [str(item) for item in command]
        self.commands.append(command)
        if len(command) == 1:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="This is version 1.60 (test)\n",
                stderr="",
            )
        wrapper = Path(command[-2])
        self.wrapper_headers.append(wrapper.read_text(encoding="utf-8").splitlines()[0])
        output = Path(command[-1])
        task_id = int(wrapper.stem.rsplit("-", 1)[-1])
        if task_id == self.fail_task:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="compile failed")
        output.write_bytes(f"behavior-{task_id}".encode())
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_resolve_acc_prefers_explicit_then_environment(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit-acc"
    environment = tmp_path / "environment-acc"
    explicit.touch()
    environment.touch()

    assert resolve_acc(explicit, environ={"ACC_PATH": str(environment)}) == explicit.resolve()
    assert resolve_acc(None, environ={"ACC_PATH": str(environment)}) == environment.resolve()


def test_build_crystal_run_has_seven_deterministic_map_groups(tmp_path: Path) -> None:
    source_dir, include_dir = make_source_tree(tmp_path)
    acc_path = tmp_path / "acc"
    acc_path.touch()
    runner = FakeRunner()
    settings = BuildSettings(
        source_dir=source_dir,
        output_wad=tmp_path / "crystal_run.wad",
        manifest_path=tmp_path / "manifest.json",
        acc_path=acc_path,
        acc_include=include_dir,
    )

    manifest = build_crystal_run(settings, runner=runner)

    entries = inspect_pwad(settings.output_wad.read_bytes())
    expected = [
        name
        for map_name in (f"MAP{index:02d}" for index in range(1, 8))
        for name in (map_name, "TEXTMAP", "BEHAVIOR", "SCRIPTS", "ENDMAP")
    ]
    assert [entry.name for entry in entries] == expected
    assert runner.wrapper_headers == [f"#define BOTC_TASK_ID {index}" for index in range(1, 8)]
    payload = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    assert payload["wad_sha256"] == manifest.wad_sha256
    assert list(payload["source_sha256"]) == sorted(payload["source_sha256"])
    assert payload["maps"] == [f"MAP{index:02d}" for index in range(1, 8)]
    assert payload["protocol_version"] == 2


def test_compiler_failure_does_not_replace_outputs(tmp_path: Path) -> None:
    source_dir, include_dir = make_source_tree(tmp_path)
    acc_path = tmp_path / "acc"
    acc_path.touch()
    output_wad = tmp_path / "crystal_run.wad"
    output_wad.write_bytes(b"previous")
    settings = BuildSettings(
        source_dir=source_dir,
        output_wad=output_wad,
        manifest_path=tmp_path / "manifest.json",
        acc_path=acc_path,
        acc_include=include_dir,
    )

    with pytest.raises(RuntimeError, match="compile failed"):
        build_crystal_run(settings, runner=FakeRunner(fail_task=3))

    assert output_wad.read_bytes() == b"previous"
    assert not settings.manifest_path.exists()


def test_tracked_udmf_has_the_reviewed_minimal_geometry() -> None:
    text = Path("assets/scenarios/crystal_run/src/map.udmf").read_text(encoding="utf-8")

    assert text.count("vertex {") == 4
    assert text.count("linedef {") == 4
    assert text.count("sidedef {") == 4
    assert text.count("type = 1;") == 1
    assert text.count("type = 11;") == 2
    assert text.count("type = 30;") >= 12
    texture_names = set(re.findall(r'texture(?:middle|floor|ceiling) = "([A-Z0-9_]+)"', text))
    assert texture_names == {"BRICK9", "FLAT4", "FLOOR0_1"}
