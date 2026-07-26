from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction import (
    ExtractionEpisodeMetrics,
    evaluate_extraction_episode,
    summarize_extraction_episodes,
)
from botcolosseo.evaluation.extraction_protocol import (
    load_extraction_evaluation_protocol,
)
from botcolosseo.training.bc import append_jsonl
from botcolosseo.training.extraction_checkpoint import (
    load_extraction_strong_actor,
    load_extraction_style_actor,
)

POLICIES = ("strong", "aggressive", "defensive", "explorer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume the one frozen Extraction official test"
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=Path("runs/extraction/release/manifest.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/extraction/evaluation.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/extraction/release/official-test"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_policy(
    *,
    root: Path,
    policy: str,
    spec: dict[str, object],
    scenario_hash: str,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = root / str(spec["checkpoint"])
    if sha256_file(checkpoint) != spec["checkpoint_sha256"]:
        raise ValueError(f"{policy} release checkpoint hash drifted")
    if policy == "strong":
        model, _ = load_extraction_strong_actor(
            checkpoint,
            expected_scenario_hash=scenario_hash,
            expected_sha256=str(spec["checkpoint_sha256"]),
            device=device,
        )
        return model
    base_checkpoint = root / str(spec["base_checkpoint"])
    model, _ = load_extraction_style_actor(
        checkpoint,
        base_checkpoint=base_checkpoint,
        expected_scenario_hash=scenario_hash,
        expected_base_sha256=str(spec["base_checkpoint_sha256"]),
        bottleneck=32,
        max_delta=2.0,
        expected_sha256=str(spec["checkpoint_sha256"]),
        device=device,
    )
    return model


def _load_partial(
    path: Path,
    cases,
) -> list[ExtractionEpisodeMetrics]:
    if not path.exists():
        return []
    episodes = [
        ExtractionEpisodeMetrics(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(episodes) > len(cases):
        raise ValueError("Official-test partial result is longer than protocol")
    for episode, case in zip(episodes, cases, strict=False):
        if (
            episode.seed,
            episode.learner_side,
            episode.opponent_style,
        ) != (case.seed, case.learner_side, case.opponent_style):
            raise ValueError("Official-test partial result identity drifted")
    return episodes


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    manifest_path = _resolve(root, args.release_manifest)
    protocol_path = _resolve(root, args.protocol)
    output_dir = _resolve(root, args.output_dir)
    receipt_path = output_dir / "receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Official Extraction test already has a receipt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = load_extraction_evaluation_protocol(protocol_path)
    if (
        manifest.get("test_cases_accessed") is not False
        or manifest.get("protocol_sha256") != protocol.sha256
        or set(manifest.get("policies", {})) != set(POLICIES)
    ):
        raise ValueError("Official-test release identity does not match")
    lock_path = output_dir / "lock.json"
    lock = {
        "release_sha256": manifest["release_sha256"],
        "release_manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": protocol.sha256,
        "test_cases_accessed": True,
    }
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise ValueError("Official-test lock belongs to another release")
    else:
        _atomic_json(lock, lock_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cases = protocol.cases("test")
    policy_summaries: dict[str, object] = {}
    for policy in POLICIES:
        model = _load_policy(
            root=root,
            policy=policy,
            spec=manifest["policies"][policy],
            scenario_hash=manifest["scenario_hash"],
            device=device,
        )
        episodes_path = output_dir / f"{policy}-episodes.jsonl"
        episodes = _load_partial(episodes_path, cases)
        for case in cases[len(episodes) :]:
            episode = evaluate_extraction_episode(
                root=root,
                checkpoint=root / manifest["policies"][policy]["checkpoint"],
                style=policy,
                case=case,
                device=device,
                policy_model=model,
            )
            append_jsonl(episodes_path, asdict(episode))
            episodes.append(episode)
        summary = summarize_extraction_episodes(tuple(episodes))
        policy_summaries[policy] = summary
        _atomic_json(
            {
                "schema_version": 1,
                "policy": policy,
                "episodes_evaluated": len(episodes),
                "checkpoint_sha256": manifest["policies"][policy][
                    "checkpoint_sha256"
                ],
                "metrics": summary,
                "split": "test",
                "test_cases_accessed": True,
            },
            output_dir / f"{policy}-summary.json",
        )
    receipt = {
        "schema_version": 1,
        "release_sha256": manifest["release_sha256"],
        "protocol_sha256": protocol.sha256,
        "episodes_per_policy": len(cases),
        "total_episodes": len(cases) * len(POLICIES),
        "policy_metrics": policy_summaries,
        "test_cases_accessed": True,
        "complete": True,
    }
    _atomic_json(receipt, receipt_path)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
