from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file

IDENTITY_COMPONENTS = {
    "scenario_config": Path(
        "assets/scenarios/crystal_run_extraction_randomized/"
        "crystal_run_extraction_randomized.cfg"
    ),
    "scenario_manifest": Path(
        "assets/scenarios/crystal_run_extraction_randomized/manifest.json"
    ),
    "scenario_wad": Path(
        "assets/scenarios/crystal_run_extraction_randomized/"
        "crystal_run_extraction_randomized.wad"
    ),
    "layout_generator": Path("src/botcolosseo/envs/extraction_layouts.py"),
    "game_rules": Path("src/botcolosseo/envs/extraction_rules.py"),
    "teacher": Path("src/botcolosseo/agents/extraction_teachers.py"),
    "evaluation_protocol": Path("configs/extraction/randomized/evaluation.yaml"),
    "metric_implementation": Path("src/botcolosseo/evaluation/extraction.py"),
    "gate_implementation": Path("src/botcolosseo/evaluation/extraction_gates.py"),
}
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def experiment_identity_sha256(payload: dict[str, object]) -> str:
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "experiment_identity_sha256"
    }
    return _canonical_sha256(unsigned)


def current_git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_experiment_identity(
    *,
    root: Path,
    git_commit: str,
) -> dict[str, object]:
    root = root.resolve()
    if not GIT_COMMIT_PATTERN.fullmatch(git_commit):
        raise ValueError("Experiment identity requires a full lowercase Git commit")
    components: dict[str, object] = {}
    for name, relative in IDENTITY_COMPONENTS.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Experiment identity component is missing: {relative}")
        components[name] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
        }

    manifest_path = root / IDENTITY_COMPONENTS["scenario_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario_hash = manifest.get("wad_sha256")
    if scenario_hash != components["scenario_wad"]["sha256"]:  # type: ignore[index]
        raise ValueError("Scenario manifest does not bind the tracked WAD")

    payload: dict[str, object] = {
        "schema_version": 1,
        "identity_scope": "scenario_rules_teacher_protocol_metrics_and_code",
        "git_commit": git_commit,
        "scenario_hash": scenario_hash,
        "metric_schema_version": 2,
        "components": components,
    }
    payload["experiment_identity_sha256"] = experiment_identity_sha256(payload)
    return payload


def validate_experiment_identity(
    *,
    root: Path,
    payload: dict[str, object],
) -> None:
    git_commit = payload.get("git_commit")
    if (
        payload.get("schema_version") != 1
        or payload.get("metric_schema_version") != 2
        or not isinstance(git_commit, str)
    ):
        raise ValueError("Experiment identity schema does not match")
    expected = build_experiment_identity(root=root, git_commit=git_commit)
    if payload != expected:
        raise ValueError("Experiment identity components drifted")
