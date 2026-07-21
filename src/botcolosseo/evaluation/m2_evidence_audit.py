from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from botcolosseo.evaluation.m2 import M2_POLICIES, sha256_file
from botcolosseo.scenarios.duel_splits import DUEL_OPPONENTS

REQUIRED_GATES = {
    "official",
    "complete",
    "protocol_clean",
    "artifact_clean",
    "ppo_win_rate_minus_bc",
    "ppo_win_rate_minus_random",
    "ppo_objective_rate_minus_bc",
    "paired_score_lcb_positive",
    "per_opponent_floor",
}
INTEGRITY_GATES = {"official", "complete", "protocol_clean", "artifact_clean"}


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def audit_official_evidence(
    report_dir: Path,
    *,
    pairs_per_opponent: int = 50,
    require_capability_pass: bool = True,
) -> dict[str, object]:
    if pairs_per_opponent <= 0:
        raise ValueError("pairs_per_opponent must be positive")
    report_dir = report_dir.expanduser().resolve()
    episode_path = report_dir / "episodes.csv"
    summary_path = report_dir / "summary.json"
    manifest_path = report_dir / "manifest.json"
    summary = _load_json(summary_path)
    manifest = _load_json(manifest_path)

    expected_episodes = (
        len(M2_POLICIES) * len(DUEL_OPPONENTS) * pairs_per_opponent * 2
    )
    if not (summary.get("official") is True and summary.get("complete") is True):
        raise ValueError("Official M2 summary is incomplete")
    if require_capability_pass and summary.get("passed") is not True:
        raise ValueError("Official M2 summary did not pass")
    if summary.get("episodes") != expected_episodes or summary.get(
        "expected_episodes"
    ) != expected_episodes:
        raise ValueError("Official M2 summary episode count is incorrect")
    if summary.get("protocol_inconsistencies") != 0:
        raise ValueError("Official M2 summary contains protocol inconsistencies")
    if summary.get("artifact_inconsistencies") != 0:
        raise ValueError("Official M2 summary contains artifact inconsistencies")
    gates = summary.get("gates")
    if not isinstance(gates, dict) or not REQUIRED_GATES <= gates.keys():
        raise ValueError("Official M2 summary is missing required gates")
    if any(gates[name] is not True for name in INTEGRITY_GATES):
        raise ValueError("Official M2 summary contains a failed integrity gate")
    capability_passed = summary.get("passed") is True and all(
        gates[name] is True for name in REQUIRED_GATES
    )
    if require_capability_pass and not capability_passed:
        raise ValueError("Official M2 summary contains a failed gate")
    policies = summary.get("policies")
    if not isinstance(policies, dict) or set(policies) != set(M2_POLICIES):
        raise ValueError("Official M2 summary policy set is incorrect")

    if not (
        manifest.get("official") is True
        and manifest.get("split") == "test"
        and manifest.get("git_dirty") is False
    ):
        raise ValueError("Official M2 manifest provenance is invalid")
    if manifest.get("policies") != list(M2_POLICIES):
        raise ValueError("Official M2 manifest policy order is incorrect")
    if manifest.get("opponents") != list(DUEL_OPPONENTS):
        raise ValueError("Official M2 manifest opponent order is incorrect")
    if manifest.get("pairs_per_opponent") != pairs_per_opponent:
        raise ValueError("Official M2 manifest pair count is incorrect")
    if manifest.get("episodes_sha256") != sha256_file(episode_path):
        raise ValueError("Official episode CSV hash does not match its manifest")
    if manifest.get("summary_sha256") != sha256_file(summary_path):
        raise ValueError("Official summary hash does not match its manifest")

    with episode_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required_fields = {"policy", "opponent", "pair_index", "learner_side", "seed"}
    if not rows or not required_fields <= rows[0].keys():
        raise ValueError("Official episode CSV is missing required fields")
    if len(rows) != expected_episodes:
        raise ValueError("Official episode CSV row count is incorrect")

    identities = [
        (row["policy"], row["opponent"], row["pair_index"], row["learner_side"])
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Official episode CSV contains duplicate identities")
    expected_per_policy = len(DUEL_OPPONENTS) * pairs_per_opponent * 2
    if Counter(row["policy"] for row in rows) != Counter(
        {policy: expected_per_policy for policy in M2_POLICIES}
    ):
        raise ValueError("Official episode CSV policy balance is incorrect")
    if Counter((row["policy"], row["opponent"]) for row in rows) != Counter(
        {
            (policy, opponent): pairs_per_opponent * 2
            for policy in M2_POLICIES
            for opponent in DUEL_OPPONENTS
        }
    ):
        raise ValueError("Official episode CSV opponent balance is incorrect")

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["opponent"], row["pair_index"])].append(row)
    expected_groups = len(M2_POLICIES) * len(DUEL_OPPONENTS) * pairs_per_opponent
    if len(groups) != expected_groups or any(
        len(pair) != 2
        or {row["learner_side"] for row in pair} != {"host", "opponent"}
        or len({row["seed"] for row in pair}) != 1
        for pair in groups.values()
    ):
        raise ValueError("Official episode CSV is not exactly side-swapped")

    checkpoint_hashes = manifest.get("checkpoint_sha256")
    if not isinstance(checkpoint_hashes, dict):
        raise ValueError("Official manifest is missing checkpoint hashes")
    for policy in ("ppo", "bc"):
        training = _load_json(report_dir / f"{policy}-training-summary.json")
        if checkpoint_hashes.get(policy) != training.get(
            "selected_checkpoint_sha256"
        ):
            raise ValueError(
                f"Official {policy} checkpoint differs from validation selection"
            )

    retries = summary.get("environment_retries", 0)
    if not isinstance(retries, int) or retries < 0:
        raise ValueError("Official environment retry count is invalid")
    return {
        "checkpoint_sha256": {
            policy: checkpoint_hashes[policy] for policy in ("ppo", "bc")
        },
        "environment_retries": retries,
        "episodes": len(rows),
        "official": True,
        "pair_groups": len(groups),
        "capability_passed": capability_passed,
        "integrity_passed": True,
        "passed": capability_passed,
    }


def audit_repository_provenance(
    root: Path, report_dir: Path, result: dict[str, object]
) -> dict[str, object]:
    root = root.expanduser().resolve()
    report_dir = report_dir.expanduser().resolve()
    manifest = _load_json(report_dir / "manifest.json")
    paths = {
        "config_sha256": root / "configs/m2/evaluation.yaml",
        "split_sha256": root / "configs/m2/test.json",
        "scenario_manifest_sha256": root
        / "assets/scenarios/crystal_run/manifest.json",
    }
    for field, path in paths.items():
        if manifest.get(field) != sha256_file(path):
            raise ValueError(f"Official manifest {field} no longer matches the repository")

    scenario = _load_json(paths["scenario_manifest_sha256"])
    if manifest.get("scenario_hash") != scenario.get("wad_sha256"):
        raise ValueError("Official scenario hash differs from the tracked WAD")
    config = yaml.safe_load(paths["config_sha256"].read_text(encoding="utf-8"))
    checkpoint_hashes = manifest["checkpoint_sha256"]
    if not isinstance(checkpoint_hashes, dict):
        raise ValueError("Official manifest checkpoint hashes are invalid")
    for policy in ("ppo", "bc"):
        checkpoint = root / config["policies"][policy]["checkpoint"]
        if checkpoint_hashes.get(policy) != sha256_file(checkpoint):
            raise ValueError(f"Local {policy} checkpoint differs from official evidence")

    commit = manifest.get("git_commit")
    if not isinstance(commit, str) or not commit:
        raise ValueError("Official manifest is missing its Git commit")
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError("Official evaluation commit is absent from Git history")
    return {**result, "evaluation_commit": commit, "repository_provenance": True}
