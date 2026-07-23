from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from botcolosseo.agents.hybrid_policy import HybridEvaluationPolicy
from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.agents.style_governor import GovernorTelemetry
from botcolosseo.cli.render_hybrid_showcase import (
    load_hybrid_showcase_policies,
)
from botcolosseo.demo.showcase import (
    ObserverStudyStep,
    record_showcase_episode,
)
from botcolosseo.envs.video import write_mp4
from botcolosseo.evaluation.hybrid_showcase import (
    load_hybrid_showcase_config,
)
from botcolosseo.evaluation.showcase import canonical_json
from botcolosseo.evaluation.user_study_video import (
    rank_user_study_candidates,
)
from botcolosseo.scenarios.duel_splits import DuelCase
from botcolosseo.scenarios.regions import RegionGraph

_LEDGERS = {
    "aggressive": Path("reports/m4/evaluation/aggressive-alpha-025/episodes.jsonl"),
    "defensive": Path("reports/m5/hybrid/defensive/formal-a/episodes.jsonl"),
    "explorer": Path("reports/m5/hybrid/explorer/formal-c/episodes.jsonl"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render high-signal validation candidates for blind review"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/showcase/hybrid-product.yaml"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("configs/m2/validation.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument(
        "--styles",
        nargs="+",
        choices=tuple(_LEDGERS),
        default=tuple(_LEDGERS),
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output_dir = (root / args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite user-study candidates")
    if not 1 <= args.fps <= 30:
        raise ValueError("User-study candidate FPS must be in [1, 30]")
    config = load_hybrid_showcase_config(args.config, root=root)
    records = {
        style: _read_jsonl(root / path) for style, path in _LEDGERS.items()
    }
    rankings = rank_user_study_candidates(
        aggressive_records=records["aggressive"],
        defensive_records=records["defensive"],
        explorer_records=records["explorer"],
        limit=args.limit,
    )
    case_path = (root / args.cases).resolve()
    cases = _case_lookup(_read_json(case_path))
    scenario_hash = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text(
            encoding="utf-8"
        )
    )["wad_sha256"]
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
    arena_config = root / "assets/scenarios/crystal_run/crystal_run.cfg"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir.mkdir(parents=True)
    rendered = []
    for style in args.styles:
        candidates = rankings[style]
        style_dir = output_dir / style
        style_dir.mkdir()
        for rank, candidate in enumerate(candidates, start=1):
            case_id = str(candidate["case_id"])
            case = cases.get(case_id)
            if case is None:
                raise ValueError(f"Candidate case is absent from validation: {case_id}")
            policies = load_hybrid_showcase_policies(
                config,
                root=root,
                scenario_hash=scenario_hash,
                device=device,
                policy_ids={style},
            )
            policy = policies[style]
            episode = record_showcase_episode(
                case,
                policy_id=style,
                policy_label="blind",
                policy=policy,
                graph=graph,
                config_path=arena_config,
                max_decisions=config.max_decisions,
                observer_study_hud=True,
                pre_episode_resets=int(candidate["formal_episode_ordinal"]),
            )
            if episode.protocol_inconsistent:
                raise ValueError(f"Candidate replay is protocol-inconsistent: {case_id}")
            target = style_dir / f"candidate-{rank:02d}.mp4"
            write_mp4(episode.frames, target, fps=args.fps)
            telemetry = (
                policy.drain_telemetry()
                if isinstance(policy, HybridEvaluationPolicy)
                else ()
            )
            rendered.append(
                {
                    **candidate,
                    "rank": rank,
                    "path": target.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(target),
                    "bytes": target.stat().st_size,
                    "fps": args.fps,
                    "frame_count": len(episode.frames),
                    "duration_seconds": len(episode.frames) / args.fps,
                    "replay": episode.to_record(),
                    "visible_summary": _visible_summary(episode.observer_steps),
                    "governor_summary": _governor_summary(telemetry),
                }
            )
            print(
                f"M6 candidate rendered: {style} {rank}/{len(candidates)} "
                f"{case_id} {len(episode.frames) / args.fps:.1f}s",
                flush=True,
            )
    manifest = {
        "schema_version": 1,
        "stage": "m6-user-study-candidates",
        "selection": "formal-ledger shortlist for curated visual review",
        "config_sha256": config.config_sha256,
        "cases_sha256": sha256_file(case_path),
        "ledger_sha256": {
            style: sha256_file(root / path) for style, path in _LEDGERS.items()
        },
        "policy_artifact_sha256": {
            row.policy_id: row.expected_sha256
            for row in config.policies
            if row.policy_id in _LEDGERS
        },
        "scenario_hash": scenario_hash,
        "candidates": rendered,
        "observer_only_fields": [
            "opponent_health",
            "core_owner",
            "neutral_event_labels",
        ],
        "observer_fields_not_available_to_policy": True,
        "test_cases_accessed": False,
    }
    (output_dir / "manifest.json").write_bytes(canonical_json(manifest))
    print(
        json.dumps(
            {
                "manifest": (output_dir / "manifest.json").as_posix(),
                "candidates": len(rendered),
                "test_cases_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _visible_summary(steps: tuple[ObserverStudyStep, ...]) -> dict[str, object]:
    events = Counter(
        event
        for step in steps
        for event in step.events
    )
    actions = Counter(step.action for step in steps)
    return {
        "events": dict(sorted(events.items())),
        "actions": dict(sorted(actions.items())),
        "minimum_self_health": min(step.self_health for step in steps),
        "minimum_opponent_health": min(
            step.opponent_health for step in steps
        ),
        "core_owner_counts": dict(
            sorted(Counter(step.core_owner for step in steps).items())
        ),
    }


def _governor_summary(
    telemetry: tuple[GovernorTelemetry, ...],
) -> dict[str, object] | None:
    if not telemetry:
        return None
    return {
        "states": dict(
            sorted(Counter(row.state for row in telemetry).items())
        ),
        "route_modes": dict(
            sorted(
                Counter(
                    row.route_mode
                    for row in telemetry
                    if row.route_mode is not None
                ).items()
            )
        ),
        "interventions": sum(row.intervened for row in telemetry),
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_lookup(payload: object) -> dict[str, DuelCase]:
    if not isinstance(payload, list):
        raise ValueError("Validation cases must be a JSON list")
    result = {}
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("Validation case row must be an object")
        case = DuelCase(**row)
        key = f"{case.opponent}:{case.pair_index}:{case.learner_side}"
        if key in result:
            raise ValueError("Validation cases contain duplicate identities")
        result[key] = case
    return result
