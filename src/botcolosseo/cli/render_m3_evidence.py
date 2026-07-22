from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from botcolosseo.evaluation.m3_evidence_audit import audit_m3_evidence
from botcolosseo.evaluation.m3_figures import render_evidence_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render audited M3 product evidence")
    parser.add_argument(
        "--official-report-dir", type=Path, default=Path("reports/m3/official")
    )
    parser.add_argument(
        "--crossplay-csv", type=Path, default=Path("reports/m3/crossplay.csv")
    )
    parser.add_argument("--pool-history", type=Path, required=True)
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=Path("docs/assets/m3-crossplay-heatmap.png"),
    )
    parser.add_argument(
        "--pool-output",
        type=Path,
        default=Path("docs/assets/m3-pfsp-pool-history.png"),
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=Path("reports/m3/crossplay-matrix.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    audit = audit_m3_evidence(
        resolve(args.official_report_dir),
        artifact_root=root,
        require_capability_pass=False,
    )
    payload = render_evidence_bundle(
        crossplay_csv=resolve(args.crossplay_csv),
        pool_history_path=resolve(args.pool_history),
        heatmap_output=resolve(args.heatmap_output),
        pool_output=resolve(args.pool_output),
        matrix_output=resolve(args.matrix_output),
    )
    print(json.dumps({"audit": audit, "matrix": payload}, indent=2, sort_keys=True))
    return 0
