from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from botcolosseo.agents.duel_teachers import create_duel_teacher
from botcolosseo.agents.league_opponents import (
    CheckpointOpponentPolicy,
    OpponentSpec,
    sha256_file,
)
from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.evaluation.crossplay import (
    CrossplayRow,
    evaluate_crossplay,
    run_crossplay_episode,
    summarize_payoff_matrix,
    write_crossplay_csv_atomic,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS
from botcolosseo.scenarios.league_splits import load_league_cases
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.duel_rollout import (
    DuelOpponentController,
    ScriptDuelOpponentController,
)
from botcolosseo.training.historical_pool import HistoricalPoolManifest, load_pool
from botcolosseo.training.league_rollout import CheckpointDuelOpponentController


class CrossplayControllerFactory:
    def __init__(
        self,
        *,
        graph: RegionGraph,
        device: torch.device,
        checkpoint_loader: Callable[..., Any] = CheckpointOpponentPolicy.load,
    ) -> None:
        self._graph = graph
        self._device = device
        self._checkpoint_loader = checkpoint_loader
        self._templates: dict[str, Any] = {}

    def create(self, spec: OpponentSpec, *, side: str) -> DuelOpponentController:
        if spec.kind == "script":
            return ScriptDuelOpponentController(
                create_duel_teacher(spec.opponent_id, self._graph, side=side)
            )
        template = self._templates.get(spec.opponent_id)
        if template is None:
            template = self._checkpoint_loader(spec, device=self._device)
            self._templates[spec.opponent_id] = template
        return CheckpointDuelOpponentController(template.fork())


def run_case_with_retries(
    runner: Callable[[], CrossplayRow], *, max_attempts: int
) -> CrossplayRow:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    for attempt in range(1, max_attempts + 1):
        try:
            return replace(runner(), environment_attempts=attempt)
        except RuntimeError as error:
            retriable = str(error) == (
                "Duel respawn did not complete within the warm-up limit"
            )
            if not retriable or attempt == max_attempts:
                raise
    raise AssertionError("unreachable")


def ensure_evidence_targets_absent(output_dir: Path) -> None:
    names = (
        "crossplay.csv",
        "matrix.json",
        "manifest.json",
    )
    conflicts = [name for name in names if (output_dir / name).exists()]
    if conflicts:
        raise FileExistsError(
            f"Cross-play evidence already exists: {', '.join(conflicts)}"
        )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else root / path


def _pool_specs(
    pool: HistoricalPoolManifest, *, root: Path
) -> tuple[OpponentSpec, ...]:
    return tuple(
        OpponentSpec(
            opponent_id=entry.policy_id,
            kind="checkpoint",
            checkpoint=str(root / entry.checkpoint),
            checkpoint_sha256=entry.checkpoint_sha256,
            scenario_hash=entry.scenario_hash,
            selection_evidence=entry.validation_report,
        )
        for entry in pool.entries
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the M3 validation cross-play matrix")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument(
        "--cases", type=Path, default=Path("configs/m3/validation.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-decisions", type=int, default=525)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--include-scripts", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_decisions <= 0 or args.max_attempts <= 0:
        raise ValueError("Cross-play limits must be positive")
    if (args.candidate_checkpoint is None) != (args.candidate_id is None):
        raise ValueError("Candidate checkpoint and ID must be provided together")
    root = _project_root()
    pool_path = _resolve(root, args.pool)
    cases_path = _resolve(root, args.cases)
    output_dir = _resolve(root, args.output_dir)
    pool = load_pool(pool_path, artifact_root=root)
    cases = load_league_cases(
        cases_path, expected_split="validation", expected_pairs=50
    )
    scenario_hash = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text(
            encoding="utf-8"
        )
    )["wad_sha256"]
    if any(entry.scenario_hash != scenario_hash for entry in pool.entries):
        raise ValueError("Cross-play pool differs from the current scenario")
    specs = list(_pool_specs(pool, root=root))
    candidate_hash: str | None = None
    if args.candidate_checkpoint is not None:
        candidate_path = _resolve(root, args.candidate_checkpoint)
        candidate_hash = sha256_file(candidate_path)
        specs.append(
            OpponentSpec(
                opponent_id=args.candidate_id,
                kind="checkpoint",
                checkpoint=str(candidate_path),
                checkpoint_sha256=candidate_hash,
                scenario_hash=scenario_hash,
                selection_evidence="validation:candidate",
            )
        )
    if args.include_scripts:
        specs.extend(
            OpponentSpec(
                opponent_id=name,
                kind="script",
                checkpoint=None,
                checkpoint_sha256=None,
                scenario_hash=scenario_hash,
                selection_evidence=f"builtin:{name}",
            )
            for name in DUEL_OPPONENTS
        )
    if len({spec.opponent_id for spec in specs}) != len(specs):
        raise ValueError("Cross-play roster contains duplicate policy IDs")
    ordered_specs = tuple(sorted(specs, key=lambda spec: spec.opponent_id))
    preflight = {
        "candidate_checkpoint_sha256": candidate_hash,
        "cases_sha256": sha256_file(cases_path),
        "expected_executed_rows": 5 * len(ordered_specs) * (len(ordered_specs) + 1),
        "policy_count": len(ordered_specs),
        "pool_manifest_sha256": pool.manifest_sha256,
        "preflight_passed": True,
        "scenario_hash": scenario_hash,
        "split": "validation",
        "test_cases_accessed": False,
        "roster": [
            {
                "checkpoint_sha256": spec.checkpoint_sha256,
                "kind": spec.kind,
                "policy_id": spec.opponent_id,
            }
            for spec in ordered_specs
        ],
    }
    ensure_evidence_targets_absent(output_dir)
    if args.preflight:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
    factory = CrossplayControllerFactory(graph=graph, device=device)
    config_path = root / "assets/scenarios/crystal_run/crystal_run.cfg"
    completed = 0
    total = int(preflight["expected_executed_rows"])

    def runner(left: OpponentSpec, right: OpponentSpec, case) -> CrossplayRow:
        nonlocal completed
        left_side = case.learner_side
        right_side = "opponent" if left_side == "host" else "host"
        row = run_case_with_retries(
            lambda: run_crossplay_episode(
                case,
                left_spec=left,
                right_spec=right,
                left_controller=factory.create(left, side=left_side),
                right_controller=factory.create(right, side=right_side),
                graph=graph,
                config_path=config_path,
                max_decisions=args.max_decisions,
            ),
            max_attempts=args.max_attempts,
        )
        completed += 1
        if completed % 10 == 0 or completed == total:
            print(f"M3 cross-play progress: {completed}/{total}", flush=True)
        return row

    rows = evaluate_crossplay(ordered_specs, cases, episode_runner=runner)
    matrix = summarize_payoff_matrix(
        rows, policy_ids=tuple(spec.opponent_id for spec in ordered_specs)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "crossplay.csv"
    matrix_path = output_dir / "matrix.json"
    write_crossplay_csv_atomic(rows, csv_path)
    _atomic_json(matrix, matrix_path)
    manifest = {
        **preflight,
        "crossplay_csv_sha256": sha256_file(csv_path),
        "executed_rows": len(rows),
        "matrix_sha256": sha256_file(matrix_path),
        "protocol_inconsistencies": 0,
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
