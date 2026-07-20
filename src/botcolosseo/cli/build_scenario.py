from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from botcolosseo.scenarios.build import BuildSettings, build_crystal_run


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Crystal Run scenario")
    parser.add_argument("--acc", type=Path)
    parser.add_argument("--acc-include", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repository_root()
    scenario_dir = root / "assets/scenarios/crystal_run"
    tracked_wad = scenario_dir / "crystal_run.wad"
    tracked_manifest = scenario_dir / "manifest.json"
    if args.check:
        with tempfile.TemporaryDirectory(prefix="botcolosseo-scenario-check-") as temporary:
            temporary_dir = Path(temporary)
            manifest = build_crystal_run(
                BuildSettings(
                    source_dir=scenario_dir / "src",
                    output_wad=temporary_dir / "crystal_run.wad",
                    manifest_path=temporary_dir / "manifest.json",
                    acc_path=args.acc,
                    acc_include=args.acc_include,
                )
            )
            rebuilt_wad = temporary_dir.joinpath("crystal_run.wad")
            if not tracked_wad.is_file() or rebuilt_wad.read_bytes() != tracked_wad.read_bytes():
                print("Tracked Crystal Run WAD differs from a clean rebuild")
                return 1
            tracked = json.loads(tracked_manifest.read_text(encoding="utf-8"))
            if tracked["wad_sha256"] != manifest.wad_sha256:
                print("Tracked Crystal Run manifest has the wrong WAD hash")
                return 1
    else:
        manifest = build_crystal_run(
            BuildSettings(
                source_dir=scenario_dir / "src",
                output_wad=tracked_wad,
                manifest_path=tracked_manifest,
                acc_path=args.acc,
                acc_include=args.acc_include,
            )
        )
    print(manifest.to_json(), end="")
    return 0
