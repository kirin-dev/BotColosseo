from __future__ import annotations

import argparse
import json
import secrets
from dataclasses import asdict
from pathlib import Path

from botcolosseo.data.extraction_demonstrations import ExtractionCase
from botcolosseo.evaluation.extraction_official_test import (
    OFFICIAL_TEST_EPISODES,
    load_sealed_extraction_official_test,
)
from botcolosseo.evaluation.extraction_protocol import (
    SCRIPT_STYLES,
    load_extraction_evaluation_protocol,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create unseen official-test cases after policy selection and before "
            "freezing the Extraction release"
        )
    )
    parser.add_argument(
        "--validation-protocol",
        type=Path,
        default=Path("configs/extraction/evaluation.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/extraction/release/official-test-manifest.json"),
    )
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output = _resolve(root, args.output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to regenerate sealed official-test cases: {output}"
        )
    protocol_path = _resolve(root, args.validation_protocol)
    protocol = load_extraction_evaluation_protocol(protocol_path)
    scenario_hash = json.loads(
        (
            root / "assets/scenarios/crystal_run_extraction/manifest.json"
        ).read_text(encoding="utf-8")
    )["wad_sha256"]
    generator = secrets.SystemRandom()
    used_seeds: set[int] = set()
    cases: list[ExtractionCase] = []
    for style in SCRIPT_STYLES:
        for _ in range(50):
            seed = generator.randrange(100_000_000, 2_000_000_000)
            while seed in used_seeds:
                seed = generator.randrange(100_000_000, 2_000_000_000)
            used_seeds.add(seed)
            cases.extend(
                (
                    ExtractionCase("test", seed, "host", style, "heldout-a"),
                    ExtractionCase("test", seed, "opponent", style, "heldout-a"),
                )
            )
    payload = {
        "schema_version": 1,
        "split": "test",
        "episode_count": len(cases),
        "scenario_hash": scenario_hash,
        "validation_protocol_sha256": protocol.sha256,
        "generation": "system-random-after-policy-selection",
        "cases": [asdict(case) for case in cases],
        "test_cases_executed": False,
    }
    if len(cases) != OFFICIAL_TEST_EPISODES:
        raise RuntimeError("Official-test generator produced the wrong budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    load_sealed_extraction_official_test(output)
    try:
        output_label = str(output.relative_to(root))
    except ValueError:
        output_label = str(output)
    print(
        json.dumps(
            {
                "episode_count": len(cases),
                "output": output_label,
                "scenario_hash": scenario_hash,
                "test_cases_executed": False,
                "validation_protocol_sha256": protocol.sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
