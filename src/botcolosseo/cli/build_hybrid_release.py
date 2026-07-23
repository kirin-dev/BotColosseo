from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.evaluation.hybrid_release import build_hybrid_release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a portable learned/hybrid GitHub release package"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/showcase/hybrid-product.yaml"),
    )
    parser.add_argument(
        "--showcase-manifest",
        type=Path,
        default=Path("reports/showcase/hybrid-product/manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    manifest = build_hybrid_release(
        root=root,
        config_path=(root / args.config).resolve(),
        showcase_manifest_path=(root / args.showcase_manifest).resolve(),
        output_dir=(root / args.output_dir).resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
