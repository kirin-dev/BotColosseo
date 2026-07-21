from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import torch
import yaml

from botcolosseo.agents.duel_teachers import create_duel_teacher
from botcolosseo.agents.league_opponents import (
    CheckpointOpponentPolicy,
    OpponentSpec,
    sha256_file,
)
from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.evaluation.m3 import (
    FROZEN_M3_THRESHOLDS,
    M3EpisodeRecord,
    NoOpponentDuelController,
    evaluate_m3_records,
    run_m3_episode,
)
from botcolosseo.evaluation.paired_bootstrap import (
    FROZEN_BOOTSTRAP_CONFIDENCE,
    FROZEN_BOOTSTRAP_SAMPLES,
    FROZEN_BOOTSTRAP_SEED,
)
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS
from botcolosseo.scenarios.league_splits import LeagueCase, load_league_cases
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.duel_rollout import ScriptDuelOpponentController
from botcolosseo.training.historical_pool import HistoricalPoolManifest, load_pool
from botcolosseo.training.league_rollout import CheckpointDuelOpponentController


@dataclass(frozen=True)
class EvaluationJob:
    index: int
    policy: str
    category: str
    opponent: str
    case: LeagueCase

    @property
    def identity(self) -> tuple[str, str, str, int, str]:
        return (
            self.policy,
            self.category,
            self.opponent,
            self.case.pair_index,
            self.case.learner_side,
        )


JobRunner = Callable[[EvaluationJob], M3EpisodeRecord]


class EvaluationSummary(Protocol):
    episodes: int
    expected_episodes: int

    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class PreparedEvaluation:
    root: Path
    output_dir: Path
    jobs: tuple[EvaluationJob, ...]
    run_identity: dict[str, object]
    pool: HistoricalPoolManifest
    selected_spec: OpponentSpec
    baseline_spec: OpponentSpec
    max_decisions: int
    max_attempts: int
    graph: RegionGraph
    scenario_config: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen M3 evaluator")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m3/evaluation.yaml")
    )
    parser.add_argument(
        "--selection-report", type=Path, default=Path("reports/m3/selection.json")
    )
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--m2-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def build_official_jobs(
    test_cases: Sequence[LeagueCase],
    heldout_cases: Sequence[LeagueCase],
    *,
    historical_policy_ids: Sequence[str],
) -> tuple[EvaluationJob, ...]:
    historical = tuple(sorted(historical_policy_ids))
    if len(test_cases) != 100 or len(heldout_cases) != 100:
        raise ValueError("Official M3 schedules require 50 side-swapped pairs")
    if len(set(historical)) != len(historical) or not 8 <= len(historical) <= 12:
        raise ValueError("Official M3 requires 8 to 12 unique historical policies")
    if any(case.split != "test" for case in test_cases) or any(
        case.split != "heldout" for case in heldout_cases
    ):
        raise ValueError("Official M3 jobs may use only test and heldout cases")
    jobs: list[EvaluationJob] = []

    def add(policy: str, category: str, opponent: str, case: LeagueCase) -> None:
        jobs.append(EvaluationJob(len(jobs), policy, category, opponent, case))

    for opponent in DUEL_OPPONENTS:
        for case in test_cases:
            add("strong_base", "script", opponent, case)
    for case in test_cases:
        add("strong_base", "no_opponent", "no_opponent", case)
    for offset, case in enumerate(heldout_cases):
        add(
            "strong_base",
            "heldout",
            DUEL_OPPONENTS[(offset // 2) % len(DUEL_OPPONENTS)],
            case,
        )
    historical_cases = tuple(test_cases[:40])
    for opponent in historical:
        for policy in ("strong_base", "m2_baseline"):
            for case in historical_cases:
                add(policy, "historical", opponent, case)
    return tuple(jobs)


def _record_identity(record: M3EpisodeRecord) -> tuple[str, str, str, int, str]:
    return (
        record.policy,
        record.category,
        record.opponent,
        record.pair_index,
        record.learner_side,
    )


def _canonical_line(record: M3EpisodeRecord) -> bytes:
    return (
        json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def append_episode_row(path: Path, record: M3EpisodeRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        payload = _canonical_line(record)
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("Episode ledger append was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_episode_rows(path: Path) -> tuple[M3EpisodeRecord, ...]:
    if not path.exists():
        return ()
    rows: list[M3EpisodeRecord] = []
    by_identity: dict[tuple[str, str, str, int, str], M3EpisodeRecord] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"Empty episode ledger row at line {line_number}")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid episode ledger row at line {line_number}")
        score_difference = payload.pop("score_difference", None)
        try:
            record = M3EpisodeRecord(**payload)
        except TypeError as error:
            raise ValueError(
                f"Episode ledger schema mismatch at line {line_number}"
            ) from error
        if score_difference != record.score_difference:
            raise ValueError(f"Episode score mismatch at line {line_number}")
        identity = _record_identity(record)
        previous = by_identity.get(identity)
        if previous is not None:
            if previous != record:
                raise ValueError("Episode ledger contains a conflicting duplicate")
            continue
        by_identity[identity] = record
        rows.append(record)
    return tuple(rows)


def run_case_with_retries(
    job: EvaluationJob, *, runner: JobRunner, max_attempts: int
) -> M3EpisodeRecord:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    for attempt in range(1, max_attempts + 1):
        try:
            row = replace(runner(job), environment_attempts=attempt)
            if _record_identity(row) != job.identity:
                raise ValueError("Evaluation runner changed the job identity")
            return row
        except RuntimeError as error:
            retriable = str(error) == (
                "Duel respawn did not complete within the warm-up limit"
            )
            if not retriable or attempt == max_attempts:
                raise
    raise AssertionError("unreachable")


def run_resumable_rows(
    jobs: Sequence[EvaluationJob],
    *,
    runner: JobRunner,
    ledger_path: Path,
    run_identity: dict[str, object],
    resume: bool,
    max_attempts: int,
    stop_after: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[tuple[M3EpisodeRecord, ...], bool]:
    ordered = tuple(jobs)
    if [job.index for job in ordered] != list(range(len(ordered))):
        raise ValueError("Evaluation jobs must use contiguous canonical indexes")
    if len({job.identity for job in ordered}) != len(ordered):
        raise ValueError("Evaluation jobs contain duplicate identities")
    if stop_after is not None and stop_after <= 0:
        raise ValueError("stop_after must be positive")
    ledger_path = ledger_path.expanduser().resolve()
    identity_path = ledger_path.parent / "run-identity.json"
    if resume:
        if not identity_path.is_file():
            raise FileNotFoundError("Resume requires run-identity.json")
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing_identity != run_identity:
            raise ValueError("Resume run identity does not match")
    else:
        if ledger_path.exists() or identity_path.exists():
            raise FileExistsError("Evaluation evidence exists; use --resume")
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_identity, identity_path)
    rows = load_episode_rows(ledger_path)
    job_by_identity = {job.identity: job for job in ordered}
    for row in rows:
        job = job_by_identity.get(_record_identity(row))
        if job is None or (
            row.seed,
            row.split,
        ) != (
            job.case.seed,
            job.case.split,
        ):
            raise ValueError("Episode ledger row does not belong to this run identity")
    completed = {_record_identity(row) for row in rows}
    newly_completed = 0
    for job in ordered:
        if job.identity in completed:
            continue
        row = run_case_with_retries(job, runner=runner, max_attempts=max_attempts)
        append_episode_row(ledger_path, row)
        completed.add(job.identity)
        newly_completed += 1
        if progress is not None:
            progress(len(completed), len(ordered))
        if stop_after is not None and newly_completed >= stop_after:
            break
    final_rows = load_episode_rows(ledger_path)
    return final_rows, len(final_rows) == len(ordered)


def write_m3_evidence(
    output_dir: Path,
    *,
    summary: EvaluationSummary,
    run_identity: dict[str, object],
) -> dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    ledger_path = output_dir / "episodes.jsonl"
    identity_path = output_dir / "run-identity.json"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    if not ledger_path.is_file() or not identity_path.is_file():
        raise FileNotFoundError("M3 raw rows and run identity must exist first")
    if summary_path.exists() or manifest_path.exists():
        raise FileExistsError("Completed M3 evidence already exists")
    stored_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if stored_identity != run_identity:
        raise ValueError("M3 evidence run identity changed before finalization")
    rows = load_episode_rows(ledger_path)
    if len(rows) != summary.episodes or summary.episodes != summary.expected_episodes:
        raise ValueError("M3 evidence cannot finalize incomplete raw rows")
    summary_payload = summary.to_dict()
    _atomic_json(summary_payload, summary_path)
    manifest = {
        "schema_version": 1,
        "official": True,
        "episodes": len(rows),
        "run_identity_sha256": sha256_file(identity_path),
        "episodes_sha256": sha256_file(ledger_path),
        "summary_sha256": sha256_file(summary_path),
        "selected_checkpoint_sha256": run_identity["selected_checkpoint_sha256"],
        "pool_manifest_sha256": run_identity["pool_manifest_sha256"],
        "m2_baseline_sha256": run_identity["m2_baseline_sha256"],
    }
    _atomic_json(manifest, manifest_path)
    return manifest


def _resolve(root: Path, path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _git_provenance(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    return commit, bool(status.strip())


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def prepare_official_evaluation(args: argparse.Namespace) -> PreparedEvaluation:
    root = Path(__file__).resolve().parents[3]
    config_path = _resolve(root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported M3 evaluation config schema")
    if config.get("thresholds") != FROZEN_M3_THRESHOLDS:
        raise ValueError("M3 evaluation thresholds differ from the frozen implementation")
    expected_bootstrap = {
        "seed": FROZEN_BOOTSTRAP_SEED,
        "samples": FROZEN_BOOTSTRAP_SAMPLES,
        "confidence": FROZEN_BOOTSTRAP_CONFIDENCE,
    }
    if config.get("bootstrap") != expected_bootstrap:
        raise ValueError("M3 paired bootstrap differs from the frozen implementation")
    if config.get("action_selection") != "greedy":
        raise ValueError("Official M3 action selection must be greedy")
    split_paths: dict[str, Path] = {}
    split_hashes: dict[str, str] = {}
    cases: dict[str, tuple[LeagueCase, ...]] = {}
    for split in ("test", "heldout"):
        spec = config["splits"][split]
        path = _resolve(root, spec["path"])
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise ValueError(f"Frozen M3 {split} manifest hash does not match")
        split_paths[split] = path
        split_hashes[split] = digest
        cases[split] = load_league_cases(
            path, expected_split=split, expected_pairs=50
        )
    scenario_manifest = root / "assets/scenarios/crystal_run/manifest.json"
    scenario_payload = _json_object(scenario_manifest)
    scenario_hash = str(scenario_payload["wad_sha256"])
    selection_path = _resolve(root, args.selection_report)
    selection = _json_object(selection_path)
    selected = selection.get("selected")
    if (
        selection.get("split") != "validation"
        or selection.get("test_cases_accessed") is not False
        or not isinstance(selected, dict)
    ):
        raise ValueError("Strong Base selection is not validation-only")
    selected_checkpoint = _resolve(root, args.selected_checkpoint)
    selected_hash = sha256_file(selected_checkpoint)
    if selected.get("checkpoint_sha256") != selected_hash:
        raise ValueError("Selected checkpoint does not match the committed selection")
    pool_path = _resolve(root, args.pool)
    pool = load_pool(pool_path, artifact_root=root)
    if not 8 <= len(pool.entries) <= 12:
        raise ValueError("Official M3 requires 8 to 12 historical policies")
    if any(entry.scenario_hash != scenario_hash for entry in pool.entries):
        raise ValueError("Historical pool scenario hash does not match")
    baseline_path = _resolve(root, args.m2_baseline)
    baseline_hash = sha256_file(baseline_path)
    m2_config = yaml.safe_load(
        (root / "configs/m2/evaluation.yaml").read_text(encoding="utf-8")
    )
    baseline_selection_path = _resolve(
        root, m2_config["policies"]["ppo"]["selection_summary"]
    )
    baseline_selection = _json_object(baseline_selection_path)
    recorded_baseline = baseline_selection.get(
        "selected_checkpoint_sha256",
        baseline_selection.get("checkpoint_sha256"),
    )
    if recorded_baseline != baseline_hash:
        raise ValueError("M2 baseline does not match its validation selection")
    historical_ids = tuple(entry.policy_id for entry in pool.entries)
    jobs = build_official_jobs(
        cases["test"], cases["heldout"], historical_policy_ids=historical_ids
    )
    commit, dirty = _git_provenance(root)
    run_identity = {
        "schema_version": 1,
        "official": True,
        "test_cases_accessed": True,
        "training_test_cases_accessed": False,
        "git_commit": commit,
        "git_dirty": dirty,
        "config": _relative(root, config_path),
        "config_sha256": sha256_file(config_path),
        "test_manifest": _relative(root, split_paths["test"]),
        "test_manifest_sha256": split_hashes["test"],
        "heldout_manifest": _relative(root, split_paths["heldout"]),
        "heldout_manifest_sha256": split_hashes["heldout"],
        "scenario_manifest_sha256": sha256_file(scenario_manifest),
        "scenario_hash": scenario_hash,
        "selection_report": _relative(root, selection_path),
        "selection_report_sha256": sha256_file(selection_path),
        "selected_checkpoint": _relative(root, selected_checkpoint),
        "selected_checkpoint_sha256": selected_hash,
        "pool": _relative(root, pool_path),
        "pool_file_sha256": sha256_file(pool_path),
        "pool_manifest_sha256": pool.manifest_sha256,
        "historical_policy_ids": list(historical_ids),
        "m2_baseline": _relative(root, baseline_path),
        "m2_baseline_sha256": baseline_hash,
        "m2_selection_report": _relative(root, baseline_selection_path),
        "m2_selection_report_sha256": sha256_file(baseline_selection_path),
        "expected_episodes": len(jobs),
        "max_episode_decisions": int(config["max_episode_decisions"]),
        "max_environment_attempts": int(config["max_environment_attempts"]),
        "action_selection": config["action_selection"],
        "bootstrap": config["bootstrap"],
        "thresholds": config["thresholds"],
    }
    selected_spec = OpponentSpec(
        opponent_id="strong_base",
        kind="checkpoint",
        checkpoint=str(selected_checkpoint),
        checkpoint_sha256=selected_hash,
        scenario_hash=scenario_hash,
        selection_evidence=_relative(root, selection_path),
    )
    baseline_spec = OpponentSpec(
        opponent_id="m2_baseline",
        kind="checkpoint",
        checkpoint=str(baseline_path),
        checkpoint_sha256=baseline_hash,
        scenario_hash=scenario_hash,
        selection_evidence=_relative(root, baseline_selection_path),
    )
    return PreparedEvaluation(
        root=root,
        output_dir=_resolve(root, args.output_dir),
        jobs=jobs,
        run_identity=run_identity,
        pool=pool,
        selected_spec=selected_spec,
        baseline_spec=baseline_spec,
        max_decisions=int(config["max_episode_decisions"]),
        max_attempts=int(config["max_environment_attempts"]),
        graph=RegionGraph.from_yaml(
            root / "assets/scenarios/crystal_run/src/regions.yaml"
        ),
        scenario_config=root / "assets/scenarios/crystal_run/crystal_run.cfg",
    )


class M3Runtime:
    def __init__(self, prepared: PreparedEvaluation, *, device: torch.device) -> None:
        self._prepared = prepared
        self._device = device
        self._specs = {
            prepared.selected_spec.opponent_id: prepared.selected_spec,
            prepared.baseline_spec.opponent_id: prepared.baseline_spec,
            **{
                entry.policy_id: OpponentSpec(
                    opponent_id=entry.policy_id,
                    kind="checkpoint",
                    checkpoint=str(prepared.root / entry.checkpoint),
                    checkpoint_sha256=entry.checkpoint_sha256,
                    scenario_hash=entry.scenario_hash,
                    selection_evidence=entry.validation_report,
                )
                for entry in prepared.pool.entries
            },
        }
        self._templates: dict[str, CheckpointOpponentPolicy] = {}

    def _checkpoint_controller(self, policy_id: str):
        template = self._templates.get(policy_id)
        if template is None:
            template = CheckpointOpponentPolicy.load(
                self._specs[policy_id], device=self._device
            )
            self._templates[policy_id] = template
        return CheckpointDuelOpponentController(template.fork())

    def __call__(self, job: EvaluationJob) -> M3EpisodeRecord:
        learner = self._checkpoint_controller(job.policy)
        opponent_side = "opponent" if job.case.learner_side == "host" else "host"
        if job.category in ("script", "heldout"):
            opponent = ScriptDuelOpponentController(
                create_duel_teacher(
                    job.opponent, self._prepared.graph, side=opponent_side
                )
            )
        elif job.category == "no_opponent":
            opponent = NoOpponentDuelController()
        else:
            opponent = self._checkpoint_controller(job.opponent)
        return run_m3_episode(
            job.case,
            policy_name=job.policy,
            category=job.category,
            opponent_name=job.opponent,
            learner=learner,
            opponent=opponent,
            graph=self._prepared.graph,
            config_path=self._prepared.scenario_config,
            max_decisions=self._prepared.max_decisions,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.preflight and args.resume:
        parser.error("--preflight and --resume are mutually exclusive")
    prepared = prepare_official_evaluation(args)
    preflight = {
        **prepared.run_identity,
        "device": args.device,
        "preflight_passed": prepared.run_identity["git_dirty"] is False,
    }
    if args.preflight:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0 if preflight["preflight_passed"] else 1
    if prepared.run_identity["git_dirty"] is not False:
        raise RuntimeError("Official M3 evaluation requires a clean tracked worktree")
    if (prepared.output_dir / "summary.json").exists():
        raise FileExistsError("Official M3 evaluation is already complete")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    runtime = M3Runtime(prepared, device=device)
    rows, complete = run_resumable_rows(
        prepared.jobs,
        runner=runtime,
        ledger_path=prepared.output_dir / "episodes.jsonl",
        run_identity=prepared.run_identity,
        resume=args.resume,
        max_attempts=prepared.max_attempts,
        progress=lambda completed, total: print(
            f"M3 official evaluation progress: {completed}/{total}", flush=True
        )
        if completed % 10 == 0 or completed == total
        else None,
    )
    if not complete:
        raise RuntimeError("Official M3 evaluation stopped before all rows completed")
    summary = evaluate_m3_records(
        rows,
        historical_policy_ids=tuple(entry.policy_id for entry in prepared.pool.entries),
        expected_scenario_hash=prepared.selected_spec.scenario_hash,
    )
    write_m3_evidence(
        prepared.output_dir,
        summary=summary,
        run_identity=prepared.run_identity,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0 if summary.passed else 1
