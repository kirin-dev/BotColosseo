from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from botcolosseo.cli.evaluate_m2 import main as evaluate_m2_main
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS


@dataclass(frozen=True)
class CandidateValidation:
    checkpoint: str
    checkpoint_sha256: str
    environment_steps: int
    episodes: int
    objective_rate: float
    win_rate: float
    mean_score_difference: float
    protocol_inconsistencies: int


def select_candidate(
    candidates: Sequence[CandidateValidation], *, expected_episodes: int
) -> CandidateValidation:
    if not candidates or expected_episodes <= 0:
        raise ValueError("Candidate selection requires validation evidence")
    if any(candidate.episodes != expected_episodes for candidate in candidates):
        raise ValueError("Candidate validation episode count is incomplete")
    if any(candidate.protocol_inconsistencies for candidate in candidates):
        raise ValueError("Candidate validation contains protocol inconsistencies")
    return max(
        candidates,
        key=lambda item: (
            item.objective_rate,
            item.win_rate,
            item.mean_score_difference,
            -item.environment_steps,
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_validation(
    evidence_dir: Path,
    *,
    checkpoint: Path,
    environment_steps: int,
    expected_episodes: int,
) -> CandidateValidation:
    summary = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((evidence_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_sha = _sha256(checkpoint)
    if summary["official"] or summary["passed"] or summary["complete"]:
        raise ValueError("Candidate selection evidence must remain development-only")
    if manifest["split"] != "validation" or manifest["official"]:
        raise ValueError("Candidate selection did not use validation-only evidence")
    if manifest["checkpoint_sha256"].get("ppo") != checkpoint_sha:
        raise ValueError("Candidate validation checkpoint hash does not match")
    policy = summary["policies"]["ppo"]
    return CandidateValidation(
        checkpoint=checkpoint.name,
        checkpoint_sha256=checkpoint_sha,
        environment_steps=environment_steps,
        episodes=int(policy["episodes"]),
        objective_rate=float(policy["objectives"]["rate"]),
        win_rate=float(policy["wins"]["rate"]),
        mean_score_difference=float(policy["mean_score_difference"]),
        protocol_inconsistencies=int(summary["protocol_inconsistencies"])
        + int(summary["artifact_inconsistencies"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select an M2 PPO checkpoint using validation cases only"
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/m2/ppo-full"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs-per-opponent", type=int, default=3)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args(argv)
    if args.pairs_per_opponent <= 0:
        parser.error("--pairs-per-opponent must be positive")
    root = Path(__file__).resolve().parents[3]
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    training_summary_path = run_dir / "summary.json"
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    if not training_summary.get("completed"):
        raise ValueError("PPO training must complete before checkpoint selection")
    candidate_specs = training_summary.get("candidate_checkpoints", [])
    if not candidate_specs:
        raise ValueError("PPO training produced no candidate checkpoints")
    evidence_root = args.evidence_dir or run_dir / "validation-selection"
    if not evidence_root.is_absolute():
        evidence_root = root / evidence_root
    evidence_root.mkdir(parents=True, exist_ok=True)
    candidates: list[CandidateValidation] = []
    expected_episodes = len(DUEL_OPPONENTS) * args.pairs_per_opponent * 2
    for item in candidate_specs:
        checkpoint = run_dir / item["checkpoint"]
        if _sha256(checkpoint) != item["sha256"]:
            raise ValueError(f"Candidate checkpoint hash drift: {checkpoint.name}")
        evidence_dir = evidence_root / checkpoint.stem
        if not (evidence_dir / "summary.json").exists():
            result = evaluate_m2_main(
                [
                    "--split",
                    "validation",
                    "--development",
                    "--max-pairs",
                    str(args.pairs_per_opponent),
                    "--policies",
                    "ppo",
                    "--opponents",
                    *DUEL_OPPONENTS,
                    "--ppo-checkpoint",
                    str(checkpoint),
                    "--device",
                    args.device,
                    "--output",
                    str(evidence_dir),
                ]
            )
            if result != 0:
                raise RuntimeError(f"Validation failed for {checkpoint.name}")
        candidates.append(
            _read_validation(
                evidence_dir,
                checkpoint=checkpoint,
                environment_steps=int(item["environment_steps"]),
                expected_episodes=expected_episodes,
            )
        )
    selected = select_candidate(candidates, expected_episodes=expected_episodes)
    selected_path = run_dir / "selected.pt"
    _atomic_copy(run_dir / selected.checkpoint, selected_path)
    selection = {
        "schema_version": 1,
        "split": "validation",
        "test_cases_accessed": False,
        "pairs_per_opponent": args.pairs_per_opponent,
        "episodes_per_candidate": expected_episodes,
        "selection_rule": [
            "objective_rate_desc",
            "win_rate_desc",
            "mean_score_difference_desc",
            "environment_steps_asc",
        ],
        "training_summary_sha256": _sha256(training_summary_path),
        "candidates": [asdict(candidate) for candidate in candidates],
        "selected": asdict(selected),
        "selected_checkpoint": selected_path.name,
        "selected_checkpoint_sha256": _sha256(selected_path),
    }
    _atomic_json(selection, run_dir / "selection.json")
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)
    return 0
