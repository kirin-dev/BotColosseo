import json
from pathlib import Path

import pytest

from botcolosseo.cli.promote_extraction_calibration import promote
from botcolosseo.data.demonstrations import sha256_file


def _selection(root: Path) -> tuple[Path, Path]:
    source = root / "runs/extraction/styles/aggressive-calibration-v2"
    source.mkdir(parents=True)
    checkpoint = source / "selected.pt"
    checkpoint.write_bytes(b"calibrated-policy")
    selection = source / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "eligible": True,
                "gate_schema_version": 2,
                "policy": "aggressive",
                "selected_checkpoint": str(checkpoint.relative_to(root)),
                "selected_checkpoint_sha256": sha256_file(checkpoint),
                "test_cases_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    return source, selection


def test_promotion_preserves_checkpoint_and_manifest_bytes(tmp_path: Path) -> None:
    source, source_selection = _selection(tmp_path)
    canonical = tmp_path / "runs/extraction/styles/aggressive"

    promote(source, canonical, root=tmp_path)
    promote(source, canonical, root=tmp_path)

    assert (canonical / "selected.pt").read_bytes() == b"calibrated-policy"
    assert (canonical / "selection.json").read_bytes() == source_selection.read_bytes()


def test_promotion_refuses_to_overwrite_canonical_drift(tmp_path: Path) -> None:
    source, _ = _selection(tmp_path)
    canonical = tmp_path / "runs/extraction/styles/aggressive"
    canonical.mkdir(parents=True)
    (canonical / "selected.pt").write_bytes(b"different-policy")

    with pytest.raises(FileExistsError, match="Refusing"):
        promote(source, canonical, root=tmp_path)
