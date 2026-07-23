from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.project_closeout import build_project_closeout
from botcolosseo.evaluation.showcase import canonical_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the audited current-scenario project closeout"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/m6/project-closeout.json"),
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output = (root / args.output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite project closeout")
    payload = build_project_closeout(
        root=root,
        synthetic_summary_path=root
        / "reports/m6/user-study/summary.synthetic.json",
        synthetic_provenance_path=root
        / "reports/m6/user-study/synthetic-provenance.json",
        synthetic_responses_path=root
        / "reports/m6/user-study/responses.synthetic.csv",
        curation_manifest_path=root
        / "artifacts/m6-curated-clips-v2/manifest.json",
        study_manifest_path=root / "artifacts/m6-user-study-v2/manifest.json",
        release_record_path=root / "reports/m6/hybrid-release.json",
        policy_archive_path=root
        / "artifacts/botcolosseo-hybrid-product-release.tar.gz",
        study_archive_path=root
        / "artifacts/botcolosseo-m6-user-study-v2.tar.gz",
        curated_archive_path=root
        / "artifacts/botcolosseo-m6-curated-clips-v2.tar.gz",
        future_plan_path=root
        / "docs/plans/2026-07-23-crystal-run-extraction-v2.md",
        readme_path=root / "README.md",
        readme_cn_path=root / "README_CN.md",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
