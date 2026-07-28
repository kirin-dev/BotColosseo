from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote a passing Aggressive calibration selection"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("runs/extraction/styles/aggressive-calibration-v2"),
    )
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=Path("runs/extraction/styles/aggressive"),
    )
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _copy_if_absent_or_identical(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise FileExistsError(f"Refusing to overwrite promoted artifact: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def promote(source_dir: Path, canonical_dir: Path, *, root: Path) -> None:
    source_checkpoint = source_dir / "selected.pt"
    source_selection = source_dir / "selection.json"
    if not source_checkpoint.is_file() or not source_selection.is_file():
        raise FileNotFoundError("Aggressive calibration selection is incomplete")
    selection = json.loads(source_selection.read_text(encoding="utf-8"))
    if (
        selection.get("policy") != "aggressive"
        or selection.get("gate_schema_version") != 2
        or selection.get("eligible") is not True
        or selection.get("test_cases_accessed") is not False
        or selection.get("selected_checkpoint_sha256")
        != sha256_file(source_checkpoint)
    ):
        raise ValueError("Aggressive calibration selection is not promotable")
    selected_path = root / selection["selected_checkpoint"]
    if selected_path.resolve() != source_checkpoint.resolve():
        raise ValueError("Aggressive calibration selected checkpoint path drifted")
    _copy_if_absent_or_identical(source_checkpoint, canonical_dir / "selected.pt")
    _copy_if_absent_or_identical(source_selection, canonical_dir / "selection.json")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    promote(
        _resolve(root, args.source_dir),
        _resolve(root, args.canonical_dir),
        root=root,
    )
    print("Crystal Run: Extraction Aggressive calibration promoted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
