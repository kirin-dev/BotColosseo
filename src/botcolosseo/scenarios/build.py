from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from botcolosseo.scenarios.wad import WadLump, inspect_pwad, write_pwad

MAPS = tuple(f"MAP{index:02d}" for index in range(1, 7))
Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class BuildSettings:
    source_dir: Path
    output_wad: Path
    manifest_path: Path
    acc_path: Path | None = None
    acc_include: Path | None = None


@dataclass(frozen=True)
class ScenarioManifest:
    schema_version: int
    protocol_version: int
    built_on: str
    acc_version: str
    source_sha256: dict[str, str]
    wad_sha256: str
    lump_names: tuple[str, ...]
    maps: tuple[str, ...]

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def resolve_acc(
    explicit: Path | None,
    *,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> Path:
    environment = os.environ if environ is None else environ
    candidates = [explicit]
    if environment.get("ACC_PATH"):
        candidates.append(Path(environment["ACC_PATH"]))
    discovered = which("acc")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError("ACC not found; pass --acc, set ACC_PATH, or add acc to PATH")


def resolve_acc_include(
    explicit: Path | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    environment = os.environ if environ is None else environ
    candidates = [explicit]
    if environment.get("ACC_INCLUDE"):
        candidates.append(Path(environment["ACC_INCLUDE"]))
    for candidate in candidates:
        if candidate is not None:
            resolved = candidate.expanduser().resolve()
            if (resolved / "zcommon.acs").is_file():
                return resolved
    raise FileNotFoundError(
        "ACC include directory not found; pass --acc-include or set ACC_INCLUDE"
    )


def build_crystal_run(
    settings: BuildSettings,
    *,
    runner: Runner = subprocess.run,
) -> ScenarioManifest:
    source_dir = settings.source_dir.expanduser().resolve()
    required = (
        source_dir / "map.udmf",
        source_dir / "crystal_run.acs",
        source_dir / "regions.yaml",
        source_dir / "task_variants.yaml",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Crystal Run sources: {missing}")
    acc_path = resolve_acc(settings.acc_path)
    acc_include = resolve_acc_include(settings.acc_include)
    acc_version = _read_acc_version(acc_path, runner)

    map_data = required[0].read_bytes()
    script_data = required[1].read_bytes()
    lumps: list[WadLump] = []
    with tempfile.TemporaryDirectory(prefix="botcolosseo-acs-") as temporary:
        temporary_dir = Path(temporary)
        for task_id, map_name in enumerate(MAPS, start=1):
            wrapper = temporary_dir / f"task-{task_id}.acs"
            behavior = temporary_dir / f"task-{task_id}.o"
            wrapper.write_text(
                f'#define BOTC_TASK_ID {task_id}\n#include "crystal_run.acs"\n',
                encoding="utf-8",
            )
            result = runner(
                [
                    str(acc_path),
                    "-i",
                    str(acc_include),
                    "-i",
                    str(source_dir),
                    str(wrapper),
                    str(behavior),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                message = result.stderr.strip() or result.stdout.strip() or "unknown ACC error"
                raise RuntimeError(f"ACC failed for {map_name}: {message}")
            if not behavior.is_file() or behavior.stat().st_size == 0:
                raise RuntimeError(f"ACC produced no behavior object for {map_name}")
            lumps.extend(
                (
                    WadLump(map_name, b""),
                    WadLump("TEXTMAP", map_data),
                    WadLump("BEHAVIOR", behavior.read_bytes()),
                    WadLump("SCRIPTS", script_data),
                    WadLump("ENDMAP", b""),
                )
            )

    output_wad = write_pwad(lumps, settings.output_wad)
    wad_data = output_wad.read_bytes()
    entries = inspect_pwad(wad_data)
    source_hashes = {
        path.name: _sha256(path.read_bytes())
        for path in sorted(required, key=lambda item: item.name)
    }
    manifest = ScenarioManifest(
        schema_version=1,
        protocol_version=1,
        built_on=date.today().isoformat(),
        acc_version=acc_version,
        source_sha256=source_hashes,
        wad_sha256=_sha256(wad_data),
        lump_names=tuple(entry.name for entry in entries),
        maps=MAPS,
    )
    _write_manifest(manifest, settings.manifest_path)
    return manifest


def _read_acc_version(acc_path: Path, runner: Runner) -> str:
    result = runner(
        [str(acc_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    lines = output.splitlines()
    for line in lines:
        if line.strip().lower().startswith("this is version"):
            return line.strip()
    for line in lines:
        if "version" in line.lower() and any(character.isdigit() for character in line):
            return line.strip()
    raise RuntimeError("Could not determine ACC version")


def _write_manifest(manifest: ScenarioManifest, output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary.write_text(manifest.to_json(), encoding="utf-8")
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
