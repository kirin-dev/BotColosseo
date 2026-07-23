from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import imageio.v2 as imageio
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from botcolosseo.agents.hybrid_config import load_hybrid_policy_config
from botcolosseo.agents.hybrid_policy import (
    HybridEvaluationPolicy,
    build_hybrid_evaluation_policy,
)
from botcolosseo.agents.league_opponents import (
    CheckpointOpponentPolicy,
    OpponentSpec,
    sha256_file,
)
from botcolosseo.demo.showcase import (
    CheckpointEvaluationPolicy,
    compose_showcase_comparison,
    record_showcase_episode,
)
from botcolosseo.envs.video import read_video_frames, write_gif, write_mp4
from botcolosseo.evaluation.hybrid_showcase import (
    HybridShowcaseConfig,
    HybridShowcasePolicy,
    load_hybrid_showcase_config,
)
from botcolosseo.evaluation.showcase import (
    canonical_json,
    load_showcase_cases,
    publish_staged_files,
    select_highlight_window,
    write_jsonl,
)
from botcolosseo.scenarios.regions import RegionGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the honest learned/hybrid four-policy showcase"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _checkpoint_policy(
    row: HybridShowcasePolicy,
    *,
    scenario_hash: str,
    device: torch.device,
) -> CheckpointEvaluationPolicy:
    spec = OpponentSpec(
        opponent_id=row.policy_id,
        kind="checkpoint",
        checkpoint=str(row.artifact),
        checkpoint_sha256=row.expected_sha256,
        scenario_hash=scenario_hash,
        selection_evidence=f"hybrid-showcase:{row.policy_id}",
    )
    return CheckpointEvaluationPolicy(
        row.policy_id,
        CheckpointOpponentPolicy.load(spec, device=device),
    )


def _load_policies(
    config: HybridShowcaseConfig,
    *,
    root: Path,
    scenario_hash: str,
    device: torch.device,
) -> dict[str, object]:
    policies: dict[str, object] = {}
    for row in config.policies:
        if row.kind == "checkpoint":
            policies[row.policy_id] = _checkpoint_policy(
                row,
                scenario_hash=scenario_hash,
                device=device,
            )
            continue
        hybrid_config = load_hybrid_policy_config(row.artifact, root=root)
        if hybrid_config.style != row.policy_id:
            raise ValueError("Hybrid showcase governor style does not match policy ID")
        if hybrid_config.scenario_hash != scenario_hash:
            raise ValueError("Hybrid showcase governor scenario hash does not match")
        policies[row.policy_id] = build_hybrid_evaluation_policy(
            hybrid_config,
            device=device,
        )
    return policies


def _validate_evidence(
    config: HybridShowcaseConfig,
) -> dict[str, dict[str, object]]:
    result = {}
    for row in config.evidence:
        payload = json.loads(row.summary.read_text(encoding="utf-8"))
        if row.style == "aggressive":
            valid = (
                payload.get("stage") == "m4"
                and payload.get("split") == "validation"
                and payload.get("passed") is True
                and payload.get("test_cases_accessed") is False
            )
        else:
            product = payload.get("product")
            valid = (
                payload.get("stage") == "m5-hybrid"
                and payload.get("style") == row.style
                and payload.get("split") == "validation"
                and payload.get("test_cases_accessed") is False
                and isinstance(product, dict)
                and product.get("passed") is True
                and product.get("episodes") == 200
            )
        if not valid:
            raise ValueError(f"Hybrid showcase {row.style} evidence has not passed")
        result[row.style] = payload
    return result


def _metric_value(payload: object, *keys: str) -> float:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            raise ValueError("Hybrid showcase metric source is invalid")
        current = current.get(key)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ValueError("Hybrid showcase metric value is invalid")
    return float(current)


def _render_metric_card(
    evidence: dict[str, dict[str, object]],
    output: Path,
) -> Path:
    explorer_signature = _metric_value(
        evidence["explorer"],
        "product",
        "route_action_signature_distance",
    )
    cards = (
        (
            "Strong Base win rate",
            f"{_metric_value(evidence['aggressive'], 'policies', 'strong_base', 'win_rate'):.1%}",
        ),
        (
            "Aggressive shift",
            f"{_metric_value(evidence['aggressive'], 'engagement_initiation_delta'):+.3f}",
        ),
        (
            "Defensive retention",
            f"{_metric_value(evidence['defensive'], 'product', 'skill_retention'):.1%}",
        ),
        (
            "Defensive intervention",
            f"{_metric_value(evidence['defensive'], 'product', 'intervention_rate'):.1%}",
        ),
        (
            "Explorer retention",
            f"{_metric_value(evidence['explorer'], 'product', 'skill_retention'):.1%}",
        ),
        (
            "Explorer signature",
            f"{explorer_signature:.3f}",
        ),
    )
    figure = Figure(figsize=(12, 4.8), dpi=150, facecolor="#111827")
    FigureCanvasAgg(figure)
    try:
        for index, (label, value) in enumerate(cards):
            axis = figure.add_subplot(2, 3, index + 1)
            axis.set_facecolor("#111827")
            axis.axis("off")
            axis.text(
                0.5,
                0.60,
                value,
                ha="center",
                va="center",
                color="white",
                fontsize=22,
                fontweight="bold",
            )
            axis.text(
                0.5,
                0.30,
                label,
                ha="center",
                va="center",
                color="#cbd5e1",
                fontsize=10,
            )
        figure.savefig(output, facecolor=figure.get_facecolor())
    finally:
        figure.clear()
    return output


def _git_provenance(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(tracked_status.strip())


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(payload))
    return path


def _event_scores(episodes: Sequence[object], *, decisions: int) -> list[float]:
    weights = {"PICKUP": 2.0, "VALID_HIT": 1.0, "DROP": 1.0, "SCORE": 4.0}
    scores = [0.0] * decisions
    for episode in episodes:
        for event in episode.events:
            if 0 <= event.decision_index < decisions:
                scores[event.decision_index] += weights.get(event.label, 0.0)
    return scores


def render_hybrid_showcase(
    *,
    root: Path,
    config_path: Path,
    device_name: str,
) -> dict[str, object]:
    root = root.resolve()
    config = load_hybrid_showcase_config(config_path, root=root)
    evidence_payloads = _validate_evidence(config)
    commit, dirty = _git_provenance(root)
    if dirty:
        raise ValueError("Hybrid publication showcase requires no tracked changes")
    scenario_hash = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text(encoding="utf-8")
    )["wad_sha256"]
    cases = load_showcase_cases(config.cases_manifest, root=root, expected_count=1)
    selected = [
        case
        for case in cases
        if f"{case.opponent}:{case.pair_index}:{case.learner_side}"
        == config.selected_case_id
    ]
    if len(selected) != 1:
        raise ValueError("Hybrid showcase selected case is not uniquely frozen")
    case = selected[0]
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    policies = _load_policies(
        config,
        root=root,
        scenario_hash=scenario_hash,
        device=device,
    )
    graph = RegionGraph.from_yaml(root / "assets/scenarios/crystal_run/src/regions.yaml")
    arena_config = root / "assets/scenarios/crystal_run/crystal_run.cfg"
    with tempfile.TemporaryDirectory(prefix="botcolosseo-hybrid-showcase-") as directory:
        staging = Path(directory)
        episodes = []
        staged_media: dict[str, Path] = {}
        full_frames: dict[str, tuple] = {}
        telemetry_rows: list[dict[str, object]] = []
        for row in config.policies:
            policy = policies[row.policy_id]
            episode = record_showcase_episode(
                case,
                policy_id=row.policy_id,
                policy_label=row.label,
                policy=policy,
                graph=graph,
                config_path=arena_config,
                max_decisions=config.max_decisions,
            )
            if episode.scenario_hash != scenario_hash or episode.protocol_inconsistent:
                raise ValueError("Hybrid showcase episode failed protocol validation")
            episodes.append(episode)
            target = staging / f"hybrid-{row.policy_id.replace('_', '-')}.mp4"
            write_mp4(episode.frames, target, fps=config.fps)
            staged_media[row.policy_id] = target
            full_frames[row.policy_id] = episode.frames
            if isinstance(policy, HybridEvaluationPolicy):
                telemetry_rows.extend(
                    {
                        "policy_id": row.policy_id,
                        **telemetry.__dict__,
                    }
                    for telemetry in policy.drain_telemetry()
                )
        decision_count = max(episode.decisions for episode in episodes)
        scores = _event_scores(episodes, decisions=decision_count)
        highlight = select_highlight_window(
            scores,
            window_frames=config.gif_seconds * config.fps,
        )
        comparison_frames = compose_showcase_comparison(
            tuple(
                (
                    row.label,
                    full_frames[row.policy_id][highlight[0] : highlight[1]],
                )
                for row in config.policies
            ),
            subtitle=(
                f"VALIDATION | seed={case.seed} | "
                f"vs {case.opponent} | {case.learner_side}"
            ),
        )
        gif = staging / "hybrid-four-policy.gif"
        write_gif(
            comparison_frames,
            gif,
            fps=config.fps,
            max_bytes=config.gif_max_bytes,
        )
        staged_media["comparison"] = gif
        metrics = _render_metric_card(
            evidence_payloads,
            staging / "hybrid-metrics.png",
        )
        staged_media["metrics"] = metrics
        published_frames = read_video_frames(gif)
        episode_rows = [episode.to_record() for episode in episodes]
        staged_episodes = write_jsonl(staging / "episodes.jsonl", episode_rows)
        staged_telemetry = write_jsonl(staging / "telemetry.jsonl", telemetry_rows)
        media = []
        targets: dict[str, Path] = {}
        for row in config.policies:
            source = staged_media[row.policy_id]
            target = config.output_dir / source.name
            targets[row.policy_id] = target
            frames = full_frames[row.policy_id]
            media.append(
                {
                    "policy_id": row.policy_id,
                    "label": row.label,
                    "path": target.relative_to(root).as_posix(),
                    "sha256": sha256_file(source),
                    "bytes": source.stat().st_size,
                    "frame_count": len(frames),
                    "dimensions": list(frames[0].shape[:2]),
                    "fps": config.fps,
                }
            )
        comparison_target = config.output_dir / gif.name
        targets["comparison"] = comparison_target
        media.append(
            {
                "policy_id": "comparison",
                "label": "Four-policy comparison",
                "path": comparison_target.relative_to(root).as_posix(),
                "sha256": sha256_file(gif),
                "bytes": gif.stat().st_size,
                "frame_count": len(published_frames),
                "dimensions": list(published_frames[0].shape[:2]),
                "fps": config.fps,
            }
        )
        metrics_target = config.output_dir / metrics.name
        targets["metrics"] = metrics_target
        metrics_image = imageio.imread(metrics)
        media.append(
            {
                "policy_id": "metrics",
                "label": "Hybrid product metrics",
                "path": metrics_target.relative_to(root).as_posix(),
                "sha256": sha256_file(metrics),
                "bytes": metrics.stat().st_size,
                "frame_count": 1,
                "dimensions": list(metrics_image.shape[:2]),
                "fps": 1,
            }
        )
        identity = {
            "config_sha256": config.config_sha256,
            "git_commit": commit,
            "scenario_hash": scenario_hash,
            "case_manifest_sha256": config.cases_sha256,
            "selected_case_id": config.selected_case_id,
            "policy_artifact_sha256": {
                row.policy_id: row.expected_sha256 for row in config.policies
            },
            "evidence_sha256": {
                row.style: row.expected_sha256 for row in config.evidence
            },
            "episodes_sha256": sha256_file(staged_episodes),
            "telemetry_sha256": sha256_file(staged_telemetry),
            "highlight": list(highlight),
            "test_cases_accessed": False,
        }
        run_identity = hashlib.sha256(canonical_json(identity)).hexdigest()
        selection = _write_json(
            staging / "selection.json",
            {
                "selected_case_id": config.selected_case_id,
                "selection_source": "formal validation mechanism contrast",
                "highlight": list(highlight),
                "event_score_total": sum(scores),
            },
        )
        manifest = {
            "schema_version": 1,
            "stage": "hybrid_product_showcase",
            "publication": True,
            "git_commit": commit,
            "git_dirty": False,
            "run_identity": run_identity,
            **identity,
            "episodes": len(episodes),
            "media": media,
            "test_cases_accessed": False,
        }
        staged_manifest = _write_json(staging / "manifest.json", manifest)
        transfers = [
            (staged_media[row.policy_id], targets[row.policy_id])
            for row in config.policies
        ] + [
            (gif, comparison_target),
            (metrics, metrics_target),
            (staged_episodes, config.evidence_dir / "episodes.jsonl"),
            (staged_telemetry, config.evidence_dir / "telemetry.jsonl"),
            (selection, config.evidence_dir / "selection.json"),
        ]
        publish_staged_files(
            transfers,
            staged_manifest=staged_manifest,
            target_manifest=config.evidence_dir / "manifest.json",
            run_identity=run_identity,
        )
        return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    manifest = render_hybrid_showcase(
        root=root,
        config_path=args.config,
        device_name=args.device,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
