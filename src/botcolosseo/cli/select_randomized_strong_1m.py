from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from botcolosseo.data.demonstrations import sha256_file

OPPONENT_STYLES = ("strong", "aggressive", "defensive", "explorer")
SCENARIO_DIRECTORY = "crystal_run_extraction_randomized"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the approved Randomized Strong 1M validation funnel"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--python", type=Path)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/extraction-randomized/strong-ppo-1m"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/extraction/randomized-strong-1m-selection"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/extraction/randomized-strong-1m-selection.json"),
    )
    return parser


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _case_manifest(*, episodes: int, layout_id: str) -> dict[str, object]:
    if episodes <= 0 or episodes % 8:
        raise ValueError("Selection episodes must be positive and divisible by eight")
    return {
        "schema_version": 1,
        "split": "validation",
        "paired_side_swaps": True,
        "cases": [
            {
                "split": "validation",
                "seed": seed,
                "learner_side": side,
                "opponent_style": OPPONENT_STYLES[(seed - 62_000) % 4],
                "layout_id": layout_id,
            }
            for seed in range(62_000, 62_000 + episodes // 2)
            for side in ("host", "opponent")
        ],
    }


def _rank_key(report: dict[str, object]) -> tuple[float, float, float, float]:
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    return (
        float(metrics["extraction_rate"]),
        float(metrics["win_rate"]),
        float(metrics["mean_extracted_value_advantage"]),
        -float(metrics["death_rate"]),
    )


def _brief(report: dict[str, object]) -> dict[str, object]:
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    return {
        "checkpoint": report["checkpoint"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "episodes": len(metrics["episodes"]),
        "extraction_rate": metrics["extraction_rate"],
        "win_rate": metrics["win_rate"],
        "mean_extracted_value_advantage": metrics[
            "mean_extracted_value_advantage"
        ],
        "death_rate": metrics["death_rate"],
        "protocol_inconsistencies": metrics["protocol_inconsistencies"],
        "paired_bootstrap_95": metrics["paired_bootstrap_95"],
        "actor_privilege_violations": report["actor_privilege_violations"],
        "fair_actor_observation_only": report["fair_actor_observation_only"],
        "test_cases_accessed": report["test_cases_accessed"],
    }


def _load_report(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("test_cases_accessed") is not False
        or report.get("actor_privilege_violations") != 0
        or report.get("fair_actor_observation_only") is not True
        or report["metrics"].get("protocol_inconsistencies") != 0
    ):
        raise ValueError(f"Invalid selection evaluation: {path}")
    return report


def _evaluate(
    *,
    root: Path,
    python: Path,
    checkpoint: Path,
    cases: Path,
    output: Path,
    device: str,
    checkpoint_scenario_directory: str = SCENARIO_DIRECTORY,
) -> None:
    if output.is_file():
        report = _load_report(output)
        if (
            report["checkpoint_sha256"] != sha256_file(checkpoint)
            or report["case_manifest_sha256"] != sha256_file(cases)
        ):
            raise ValueError(f"Existing evaluation identity drifted: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    command = (
        str(python),
        "-m",
        "botcolosseo.cli.evaluate_extraction_policy",
        "--checkpoint",
        str(checkpoint.relative_to(root)),
        "--style",
        "strong",
        "--strong-ppo",
        "--cases",
        str(cases.relative_to(root)),
        "--split",
        "validation",
        "--scenario-directory",
        SCENARIO_DIRECTORY,
        "--checkpoint-scenario-directory",
        checkpoint_scenario_directory,
        "--device",
        device,
        "--output",
        str(output.relative_to(root)),
    )
    environment = os.environ.copy()
    for name in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        environment.pop(name, None)
    with log.open("w", encoding="utf-8") as target:
        subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=target,
            check=True,
        )
    _load_report(output)
    log.unlink(missing_ok=True)


def _run_tasks(tasks: list[dict[str, object]], *, workers: int) -> None:
    if workers not in (1, 2):
        raise ValueError("Selection supports one or two GPU workers")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _evaluate,
                **task,
                device=f"cuda:{index % workers}",
            )
            for index, task in enumerate(tasks)
        ]
        for future in futures:
            future.result()


def _promotion_gates(
    candidate: dict[str, object],
    *,
    baseline_random: dict[str, object],
    candidate_base: dict[str, object],
    baseline_base: dict[str, object],
    candidate_heldout: dict[str, object],
    baseline_heldout: dict[str, object],
) -> dict[str, bool]:
    candidate_metrics = candidate["metrics"]
    baseline_metrics = baseline_random["metrics"]
    return {
        "randomized_extraction_improves_by_5pp": (
            candidate_metrics["extraction_rate"]
            + 1e-12
            >= baseline_metrics["extraction_rate"] + 0.05
        ),
        "randomized_win_regression_at_most_2_5pp": (
            candidate_metrics["win_rate"] + 1e-12
            >= baseline_metrics["win_rate"] - 0.025
        ),
        "randomized_death_regression_at_most_5pp": (
            candidate_metrics["death_rate"]
            <= baseline_metrics["death_rate"] + 0.05 + 1e-12
        ),
        "base_extraction_regression_at_most_5pp": (
            candidate_base["metrics"]["extraction_rate"]
            + 1e-12
            >= baseline_base["metrics"]["extraction_rate"] - 0.05
        ),
        "heldout_extraction_regression_at_most_5pp": (
            candidate_heldout["metrics"]["extraction_rate"]
            + 1e-12
            >= baseline_heldout["metrics"]["extraction_rate"] - 0.05
        ),
        "protocol_inconsistencies_zero": (
            candidate_metrics["protocol_inconsistencies"] == 0
            and candidate_base["metrics"]["protocol_inconsistencies"] == 0
            and candidate_heldout["metrics"]["protocol_inconsistencies"] == 0
        ),
        "actor_privilege_violations_zero": (
            candidate["actor_privilege_violations"] == 0
            and candidate_base["actor_privilege_violations"] == 0
            and candidate_heldout["actor_privilege_violations"] == 0
        ),
        "test_cases_not_accessed": (
            candidate["test_cases_accessed"] is False
            and candidate_base["test_cases_accessed"] is False
            and candidate_heldout["test_cases_accessed"] is False
        ),
    }


def _regular_candidate_items(
    items: list[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    expected_steps = tuple(range(50_000, 1_000_001, 50_000))
    by_step: dict[int, dict[str, object]] = {}
    for item in items:
        step = int(item["environment_steps"])
        if step in by_step:
            raise ValueError(f"Duplicate candidate step: {step}")
        by_step[step] = item
    extras = set(by_step) - set(expected_steps)
    if extras - {10_000} or any(step not in by_step for step in expected_steps):
        raise ValueError("The 1M candidate schedule is incomplete")
    return tuple(by_step[step] for step in expected_steps)


def _audit_candidates(run_dir: Path) -> list[Path]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("completed") is not True
        or summary.get("environment_steps") != 1_000_000
        or summary.get("test_cases_accessed") is not False
    ):
        raise ValueError("The 1M training artifact is incomplete")
    all_items = summary.get("candidate_checkpoints")
    if not isinstance(all_items, list):
        raise ValueError("The 1M candidate manifest is missing")
    regular_items = _regular_candidate_items(all_items)
    checkpoints = []
    for item in regular_items:
        path = run_dir / item["checkpoint"]
        environment_steps = int(item["environment_steps"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            sha256_file(path) != item["sha256"]
            or payload["metadata"]["counters"]["environment_steps"]
            != environment_steps
            or payload["metadata"]["scenario_hash"] != summary["scenario_hash"]
        ):
            raise ValueError(f"Candidate artifact audit failed: {path}")
        checkpoints.append(path)
    return checkpoints


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    python = args.python or Path(sys.executable)
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else root / args.output_root
    )
    manifests = output_root / "manifests"
    manifest_specs = {
        "randomized-120.json": (120, "randomized"),
        "randomized-240.json": (240, "randomized"),
        "base-120.json": (120, "base"),
        "heldout-120.json": (120, "heldout-a"),
    }
    for name, (episodes, layout_id) in manifest_specs.items():
        path = manifests / name
        expected = _case_manifest(episodes=episodes, layout_id=layout_id)
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != expected:
                raise ValueError(f"Selection manifest drifted: {path}")
        else:
            _atomic_json(expected, path)

    screening_manifest = root / (
        "configs/extraction/randomized/evaluation-unseen-random.json"
    )
    checkpoints = _audit_candidates(run_dir)

    def tasks_for(
        selected: list[Path], cases: Path, directory: str
    ) -> list[dict[str, object]]:
        return [
            {
                "root": root,
                "python": python,
                "checkpoint": checkpoint,
                "cases": cases,
                "output": output_root / directory / f"{checkpoint.stem}.json",
            }
            for checkpoint in selected
        ]

    screening_tasks = tasks_for(checkpoints, screening_manifest, "screening-32")
    _run_tasks(screening_tasks, workers=args.workers)
    screening_reports = [_load_report(task["output"]) for task in screening_tasks]
    screening_ranked = sorted(screening_reports, key=_rank_key, reverse=True)
    top_four_paths = [root / report["checkpoint"] for report in screening_ranked[:4]]

    expanded_manifest = manifests / "randomized-120.json"
    expanded_tasks = tasks_for(top_four_paths, expanded_manifest, "expanded-120")
    _run_tasks(expanded_tasks, workers=args.workers)
    expanded_reports = [_load_report(task["output"]) for task in expanded_tasks]
    expanded_ranked = sorted(expanded_reports, key=_rank_key, reverse=True)
    finalist_paths = [root / report["checkpoint"] for report in expanded_ranked[:2]]

    baseline_200k = root / (
        "runs/extraction-randomized/strong-ppo/candidate-0200000.pt"
    )
    baseline_fixed = root / "runs/extraction/strong-ppo/candidate-0400384.pt"
    random_final_paths = finalist_paths + [baseline_200k, baseline_fixed]
    random_final_tasks = tasks_for(
        random_final_paths, manifests / "randomized-240.json", "final-randomized-240"
    )
    random_final_tasks[-1][
        "checkpoint_scenario_directory"
    ] = "crystal_run_extraction"
    _run_tasks(random_final_tasks, workers=args.workers)
    random_final = {
        Path(task["checkpoint"]).resolve(): _load_report(task["output"])
        for task in random_final_tasks
    }

    generalization_paths = finalist_paths + [baseline_200k]
    base_tasks = tasks_for(
        generalization_paths, manifests / "base-120.json", "final-base-120"
    )
    heldout_tasks = tasks_for(
        generalization_paths,
        manifests / "heldout-120.json",
        "final-heldout-120",
    )
    _run_tasks(base_tasks + heldout_tasks, workers=args.workers)
    base_reports = {
        Path(task["checkpoint"]).resolve(): _load_report(task["output"])
        for task in base_tasks
    }
    heldout_reports = {
        Path(task["checkpoint"]).resolve(): _load_report(task["output"])
        for task in heldout_tasks
    }

    baseline_random = random_final[baseline_200k.resolve()]
    gates = {}
    for finalist in finalist_paths:
        key = str(finalist.relative_to(root))
        result = _promotion_gates(
            random_final[finalist.resolve()],
            baseline_random=baseline_random,
            candidate_base=base_reports[finalist.resolve()],
            baseline_base=base_reports[baseline_200k.resolve()],
            candidate_heldout=heldout_reports[finalist.resolve()],
            baseline_heldout=heldout_reports[baseline_200k.resolve()],
        )
        gates[key] = {
            "passed": all(result.values()),
            "gates": result,
            "failed_gates": [name for name, passed in result.items() if not passed],
        }
    passing = [
        path
        for path in finalist_paths
        if gates[str(path.relative_to(root))]["passed"]
    ]
    selected = (
        max(passing, key=lambda path: _rank_key(random_final[path.resolve()]))
        if passing
        else baseline_200k
    )
    report = {
        "schema_version": 1,
        "selection_protocol": "randomized-strong-1m-32-120-240",
        "ranking": [
            "extraction_rate_desc",
            "win_rate_desc",
            "mean_extracted_value_advantage_desc",
            "death_rate_asc",
        ],
        "test_cases_accessed": False,
        "manifests": {
            path.name: {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
            }
            for path in [screening_manifest, *sorted(manifests.glob("*.json"))]
        },
        "artifact_audit": {
            "candidate_count": 20,
            "run_dir": str(run_dir.relative_to(root)),
            "checkpoint_interval_steps": 50_000,
            "hashes_and_counters_verified": True,
            "training_completed_steps": 1_000_000,
            "training_test_cases_accessed": False,
        },
        "screening_32": [_brief(report) for report in screening_ranked],
        "expanded_120": [_brief(report) for report in expanded_ranked],
        "final_randomized_240": {
            str(path.relative_to(root)): _brief(random_final[path.resolve()])
            for path in random_final_paths
        },
        "final_base_120": {
            str(path.relative_to(root)): _brief(base_reports[path.resolve()])
            for path in generalization_paths
        },
        "final_heldout_120": {
            str(path.relative_to(root)): _brief(heldout_reports[path.resolve()])
            for path in generalization_paths
        },
        "promotion": gates,
        "selected_checkpoint": str(selected.relative_to(root)),
        "selected_checkpoint_sha256": sha256_file(selected),
        "promoted_1m_candidate": selected in finalist_paths,
        "fallback_if_no_gate_passes": str(baseline_200k.relative_to(root)),
    }
    report_path = args.report if args.report.is_absolute() else root / args.report
    _atomic_json(report, report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
