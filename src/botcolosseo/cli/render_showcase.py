from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import torch

from botcolosseo.agents.league_opponents import (
    CheckpointOpponentPolicy,
    OpponentSpec,
    sha256_file,
)
from botcolosseo.demo.showcase import (
    CheckpointEvaluationPolicy,
    compose_showcase_comparison,
    record_showcase_episode,
    render_metrics_card,
)
from botcolosseo.envs.video import read_video_frames, write_gif, write_mp4
from botcolosseo.evaluation.m2 import load_actor_policy
from botcolosseo.evaluation.showcase import (
    ShowcaseConfig,
    ShowcaseMetricEvidence,
    ShowcasePolicySpec,
    build_showcase_manifest,
    canonical_json,
    load_metric_evidence,
    load_showcase_cases,
    load_showcase_config,
    publish_staged_files,
    select_highlight_window,
    select_showcase_case,
    write_jsonl,
)
from botcolosseo.scenarios.regions import RegionGraph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render auditable Bot showcase media")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cpu")
    return parser


def load_showcase_policy(
    policy: ShowcasePolicySpec,
    *,
    publication: bool,
    checkpoint_root: Path,
    scenario_hash: str,
    device: str,
) -> object:
    if not publication and policy.policy_id not in ("ppo", "bc"):
        raise ValueError("Unsupported development policy")
    if publication and policy.policy_id not in (
        "strong_base",
        "aggressive",
        "defensive",
        "explorer",
    ):
        raise ValueError("Unsupported publication policy")
    checkpoint = _checkpoint_under_root(policy.checkpoint, checkpoint_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256_file(checkpoint) != policy.expected_sha256:
        raise ValueError("Showcase checkpoint hash does not match")
    torch_device = torch.device(device)
    if not publication:
        return load_actor_policy(
            policy.policy_id,
            checkpoint,
            device=torch_device,
            expected_scenario_hash=scenario_hash,
        )
    spec = OpponentSpec(
        opponent_id=policy.policy_id,
        kind="checkpoint",
        checkpoint=str(checkpoint),
        checkpoint_sha256=policy.expected_sha256,
        scenario_hash=scenario_hash,
        selection_evidence=f"showcase:{policy.policy_id}",
    )
    loaded = CheckpointOpponentPolicy.load(spec, device=torch_device)
    return CheckpointEvaluationPolicy(policy.policy_id, loaded)


def render_showcase(
    *,
    root: Path,
    config_path: Path,
    checkpoint_root: Path,
    device: str,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    config = load_showcase_config(config_path, root=root)
    cases = load_showcase_cases(
        config.cases_path,
        root=root,
        expected_count=1 if config.stage == "development" else 8,
    )
    scenario_manifest_path = root / "assets/scenarios/crystal_run/manifest.json"
    scenario_manifest = _load_json(scenario_manifest_path)
    scenario_hash = scenario_manifest.get("wad_sha256")
    if not isinstance(scenario_hash, str):
        raise ValueError("Scenario manifest is missing the WAD hash")
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
    expected_hashes = {
        policy.policy_id: policy.expected_sha256 for policy in config.policies
    }
    metric_evidence = _load_public_metrics(config, expected_hashes)
    git_commit, git_dirty = _git_provenance(root)
    if config.publication and git_dirty:
        raise ValueError("Production showcase requires a clean worktree")
    policies = {
        policy.policy_id: load_showcase_policy(
            policy,
            publication=config.publication,
            checkpoint_root=checkpoint_root,
            scenario_hash=scenario_hash,
            device=device,
        )
        for policy in config.policies
    }

    with tempfile.TemporaryDirectory(prefix="botcolosseo-showcase-") as directory:
        staging = Path(directory)
        raw_video: dict[tuple[str, str], Path] = {}
        episode_rows: list[dict[str, object]] = []
        for case in cases:
            for policy_spec in config.policies:
                episode = record_showcase_episode(
                    case,
                    policy_id=policy_spec.policy_id,
                    policy_label=policy_spec.label,
                    policy=policies[policy_spec.policy_id],
                    graph=graph,
                    config_path=root
                    / "assets/scenarios/crystal_run/crystal_run.cfg",
                    max_decisions=config.render.max_decisions,
                )
                if episode.scenario_hash != scenario_hash:
                    raise ValueError("Showcase episode scenario hash does not match")
                row = episode.to_record()
                episode_rows.append(row)
                path = staging / "raw" / _safe_name(row["case_id"]) / (
                    f"{policy_spec.policy_id}.mp4"
                )
                write_mp4(episode.frames, path, fps=config.render.fps)
                raw_video[(str(row["case_id"]), policy_spec.policy_id)] = path

        contrast_scores = (
            metric_evidence.case_contrast_scores
            if metric_evidence is not None
            else {str(row["case_id"]): 0.0 for row in episode_rows}
        )
        selection = select_showcase_case(
            episode_rows,
            tuple(policy.policy_id for policy in config.policies),
            contrast_scores,
            require_normal_termination=config.publication,
        )
        decision_scores = (
            metric_evidence.decision_contrast_scores[selection.selected_case_id]
            if metric_evidence is not None
            else tuple(
                0.0
                for _ in range(
                    max(int(row["decisions"]) for row in selection.selected_records)
                )
            )
        )
        highlight = select_highlight_window(
            decision_scores,
            window_frames=config.render.gif_seconds * config.render.fps,
        )
        target_paths = _target_paths(config)
        staged_paths: dict[str, Path] = {}
        selected_frames: dict[str, tuple] = {}
        for policy in config.policies:
            source = raw_video[(selection.selected_case_id, policy.policy_id)]
            target_name = target_paths[f"{policy.policy_id}_mp4"].name
            staged = staging / "publish" / target_name
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, staged)
            staged_paths[f"{policy.policy_id}_mp4"] = staged
            selected_frames[policy.policy_id] = read_video_frames(source)

        selected_case = next(
            case
            for case in cases
            if f"{case.opponent}:{case.pair_index}:{case.learner_side}"
            == selection.selected_case_id
        )
        comparison_frames = compose_showcase_comparison(
            tuple(
                (
                    policy.label,
                    selected_frames[policy.policy_id][highlight[0] : highlight[1]],
                )
                for policy in config.policies
            ),
            subtitle=(
                f"VALIDATION | seed={selected_case.seed} | "
                f"vs {selected_case.opponent} | {selected_case.learner_side}"
            ),
        )
        comparison = staging / "publish" / target_paths["comparison_gif"].name
        write_gif(
            comparison_frames,
            comparison,
            fps=config.render.fps,
            max_bytes=config.render.gif_max_bytes,
        )
        staged_paths["comparison_gif"] = comparison

        if metric_evidence is not None:
            metrics = staging / "publish" / target_paths["metrics_png"].name
            render_metrics_card(metric_evidence, metrics)
            staged_paths["metrics_png"] = metrics

        staged_episodes = write_jsonl(staging / "evidence/episodes.jsonl", episode_rows)
        selection_payload = {
            **asdict(selection),
            "highlight": list(highlight),
            "publication_eligibility": config.publication,
        }
        staged_selection = _write_json(
            staging / "evidence/case-selection.json", selection_payload
        )
        media = _media_metadata(
            root=root,
            config=config,
            target_paths=target_paths,
            staged_paths=staged_paths,
            selected_frames=selected_frames,
            comparison_frames=comparison_frames,
        )
        manifest = build_showcase_manifest(
            git_commit=git_commit,
            git_dirty=git_dirty,
            config=config,
            scenario_hash=scenario_hash,
            case_manifest_sha256=sha256_file(config.cases_path),
            checkpoint_sha256=expected_hashes,
            metric_sha256=(
                metric_evidence.source_sha256 if metric_evidence is not None else None
            ),
            episodes_path=staged_episodes,
            selected_case=selection.selected_case_id,
            highlight=highlight,
            media=media,
            gate_passed=metric_evidence is not None,
        )
        staged_manifest = _write_json(staging / "evidence/manifest.json", manifest)
        transfers = [
            (staged_paths[key], target_paths[key]) for key in staged_paths
        ] + [
            (staged_episodes, target_paths["episodes"]),
            (staged_selection, target_paths["selection"]),
        ]
        publish_staged_files(
            transfers,
            staged_manifest=staged_manifest,
            target_manifest=target_paths["manifest"],
            run_identity=str(manifest["run_identity"]),
        )
        return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    manifest = render_showcase(
        root=root,
        config_path=args.config,
        checkpoint_root=args.checkpoint_root,
        device=args.device,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _checkpoint_under_root(checkpoint: Path, checkpoint_root: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    checkpoint_root = checkpoint_root.expanduser().resolve()
    try:
        checkpoint.relative_to(checkpoint_root)
        return checkpoint
    except ValueError:
        pass
    try:
        runs_index = checkpoint.parts.index("runs")
    except ValueError as error:
        raise ValueError("Showcase checkpoint path must be under runs/") from error
    return checkpoint_root.joinpath(*checkpoint.parts[runs_index:])


def _load_public_metrics(
    config: ShowcaseConfig, expected_hashes: Mapping[str, str]
) -> ShowcaseMetricEvidence | None:
    if not config.publication:
        return None
    if config.metrics_path is None:
        raise ValueError("Production showcase requires metric evidence")
    return load_metric_evidence(
        config.metrics_path,
        expected_stage=config.stage,
        expected_hashes=expected_hashes,
    )


def _git_provenance(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def _target_paths(config: ShowcaseConfig) -> dict[str, Path]:
    if config.stage == "development":
        comparison_name = "development-comparison.gif"
        policy_prefix = "development"
    elif config.stage == "m4":
        comparison_name = "m4-base-vs-aggressive.gif"
        policy_prefix = "m4"
    else:
        comparison_name = "m5-style-comparison.gif"
        policy_prefix = "m5"
    paths = {
        "comparison_gif": config.output_dir / comparison_name,
        "episodes": config.evidence_dir / "episodes.jsonl",
        "selection": config.evidence_dir / "case-selection.json",
        "manifest": config.evidence_dir / "manifest.json",
    }
    for policy in config.policies:
        public_name = policy.policy_id.replace("_", "-")
        paths[f"{policy.policy_id}_mp4"] = (
            config.output_dir / f"{policy_prefix}-{public_name}.mp4"
        )
    if config.publication:
        paths["metrics_png"] = config.output_dir / f"{config.stage}-metrics.png"
    return paths


def _media_metadata(
    *,
    root: Path,
    config: ShowcaseConfig,
    target_paths: Mapping[str, Path],
    staged_paths: Mapping[str, Path],
    selected_frames: Mapping[str, tuple],
    comparison_frames: tuple,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy in config.policies:
        key = f"{policy.policy_id}_mp4"
        frames = selected_frames[policy.policy_id]
        rows.append(
            _media_row(
                root,
                config,
                target_paths[key],
                staged_paths[key],
                frames,
                config.render.fps,
            )
        )
    rows.append(
        _media_row(
            root,
            config,
            target_paths["comparison_gif"],
            staged_paths["comparison_gif"],
            comparison_frames,
            config.render.fps,
        )
    )
    if "metrics_png" in staged_paths:
        image = imageio.imread(staged_paths["metrics_png"])
        rows.append(
            {
                "path": _manifest_path(root, config, target_paths["metrics_png"]),
                "sha256": sha256_file(staged_paths["metrics_png"]),
                "bytes": staged_paths["metrics_png"].stat().st_size,
                "frame_count": 1,
                "dimensions": [int(image.shape[0]), int(image.shape[1])],
                "fps": 1,
            }
        )
    return rows


def _media_row(
    root: Path,
    config: ShowcaseConfig,
    target: Path,
    staged: Path,
    frames: Sequence,
    fps: int,
) -> dict[str, object]:
    first = frames[0]
    return {
        "path": _manifest_path(root, config, target),
        "sha256": sha256_file(staged),
        "bytes": staged.stat().st_size,
        "frame_count": len(frames),
        "dimensions": [int(first.shape[0]), int(first.shape[1])],
        "fps": fps,
    }


def _manifest_path(root: Path, config: ShowcaseConfig, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        if config.publication:
            raise ValueError(
                "Public showcase media must stay inside the repository"
            ) from None
        return path.name


def _safe_name(value: object) -> str:
    return str(value).replace(":", "-").replace("/", "-")


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json(payload))
    temporary.replace(path)
    return path
