from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.cli.render_showcase import load_showcase_policy
from botcolosseo.demo.showcase import record_showcase_episode
from botcolosseo.evaluation.showcase import (
    build_m4_showcase_metric_payload,
    canonical_json,
    case_id,
    load_showcase_cases,
    load_showcase_config,
)
from botcolosseo.scenarios.regions import RegionGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export hash-bound M4 metrics for the public showcase"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config = load_showcase_config(args.config, root=root)
    if config.stage != "m4" or not config.publication or config.metrics_path is None:
        raise ValueError("Showcase metric export requires an M4 publication config")
    evaluation_path = (
        args.evaluation.resolve()
        if args.evaluation.is_absolute()
        else (root / args.evaluation).resolve()
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not isinstance(evaluation, dict):
        raise ValueError("M4 evaluation summary must be an object")

    cases = load_showcase_cases(config.cases_path, root=root, expected_count=8)
    scenario_hash = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text(
            encoding="utf-8"
        )
    )["wad_sha256"]
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
    policies = {
        policy.policy_id: load_showcase_policy(
            policy,
            publication=True,
            checkpoint_root=args.checkpoint_root,
            scenario_hash=scenario_hash,
            device=args.device,
        )
        for policy in config.policies
    }
    records: list[dict[str, object]] = []
    total = len(cases) * len(config.policies)
    for case in cases:
        for policy in config.policies:
            episode = record_showcase_episode(
                case,
                policy_id=policy.policy_id,
                policy_label=policy.label,
                policy=policies[policy.policy_id],
                graph=graph,
                config_path=root / "assets/scenarios/crystal_run/crystal_run.cfg",
                max_decisions=config.render.max_decisions,
            )
            if episode.scenario_hash != scenario_hash:
                raise ValueError("Showcase metric episode scenario hash does not match")
            records.append(episode.to_record())
            print(f"M4 showcase metric progress: {len(records)}/{total}", flush=True)

    hashes = {
        policy.policy_id: policy.expected_sha256 for policy in config.policies
    }
    payload = build_m4_showcase_metric_payload(
        evaluation,
        records,
        case_ids=tuple(case_id(case) for case in cases),
        expected_hashes=hashes,
    )
    output = config.metrics_path
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(canonical_json(payload))
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
