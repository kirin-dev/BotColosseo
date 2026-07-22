from __future__ import annotations

import json
from pathlib import Path

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.cli.evaluate_m3 import load_episode_rows
from botcolosseo.evaluation.m3 import evaluate_m3_records
from botcolosseo.training.historical_pool import load_pool

_INTEGRITY_GATES = {
    "official",
    "complete",
    "pool_size",
    "protocol_clean",
    "artifact_clean",
    "heldout_core_strata_complete",
    "confidence_intervals_finite",
}


def _object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _audit_repository_bindings(identity: dict[str, object], *, root: Path) -> None:
    bindings = (
        ("config", "config_sha256"),
        ("test_manifest", "test_manifest_sha256"),
        ("heldout_manifest", "heldout_manifest_sha256"),
        ("selection_report", "selection_report_sha256"),
        ("selected_checkpoint", "selected_checkpoint_sha256"),
        ("pool", "pool_file_sha256"),
        ("m2_baseline", "m2_baseline_sha256"),
        ("m2_selection_report", "m2_selection_report_sha256"),
    )
    for path_field, hash_field in bindings:
        value = identity.get(path_field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"M3 run identity is missing {path_field}")
        path = Path(value)
        path = path if path.is_absolute() else root / path
        if not path.is_file() or identity.get(hash_field) != sha256_file(path):
            raise ValueError(f"M3 repository artifact {path_field} does not match")
    scenario_manifest = root / "assets/scenarios/crystal_run/manifest.json"
    if (
        not scenario_manifest.is_file()
        or identity.get("scenario_manifest_sha256")
        != sha256_file(scenario_manifest)
        or _object(scenario_manifest).get("wad_sha256") != identity.get("scenario_hash")
    ):
        raise ValueError("M3 scenario binding does not match")
    selection = _object(
        Path(str(identity["selection_report"]))
        if Path(str(identity["selection_report"])).is_absolute()
        else root / str(identity["selection_report"])
    )
    selected = selection.get("selected")
    if (
        selection.get("split") != "validation"
        or selection.get("test_cases_accessed") is not False
        or not isinstance(selected, dict)
        or selected.get("checkpoint_sha256")
        != identity.get("selected_checkpoint_sha256")
    ):
        raise ValueError("M3 selected checkpoint is not validation-bound")
    pool_path = Path(str(identity["pool"]))
    pool_path = pool_path if pool_path.is_absolute() else root / pool_path
    pool = load_pool(pool_path, artifact_root=root)
    if (
        pool.manifest_sha256 != identity.get("pool_manifest_sha256")
        or [entry.policy_id for entry in pool.entries]
        != identity.get("historical_policy_ids")
    ):
        raise ValueError("M3 historical pool binding does not match")
    baseline_selection_path = Path(str(identity["m2_selection_report"]))
    baseline_selection_path = (
        baseline_selection_path
        if baseline_selection_path.is_absolute()
        else root / baseline_selection_path
    )
    baseline_selection = _object(baseline_selection_path)
    selected_baseline = baseline_selection.get(
        "selected_checkpoint_sha256",
        baseline_selection.get("checkpoint_sha256"),
    )
    if selected_baseline != identity.get("m2_baseline_sha256"):
        raise ValueError("M3 M2 baseline is not validation-bound")


def audit_m3_evidence(
    report_dir: Path,
    *,
    artifact_root: Path | None = None,
    require_capability_pass: bool = True,
) -> dict[str, object]:
    if not isinstance(require_capability_pass, bool):
        raise ValueError("M3 capability-pass requirement must be boolean")
    report_dir = report_dir.expanduser().resolve()
    paths = {
        "identity": report_dir / "run-identity.json",
        "episodes": report_dir / "episodes.jsonl",
        "summary": report_dir / "summary.json",
        "manifest": report_dir / "manifest.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"M3 evidence is missing: {', '.join(missing)}")
    identity = _object(paths["identity"])
    stored_summary = _object(paths["summary"])
    manifest = _object(paths["manifest"])
    required_identity = {
        "official",
        "historical_policy_ids",
        "scenario_hash",
        "selected_checkpoint_sha256",
        "pool_manifest_sha256",
        "m2_baseline_sha256",
    }
    if not required_identity <= identity.keys() or identity["official"] is not True:
        raise ValueError("M3 run identity is incomplete or unofficial")
    if identity.get("git_dirty") not in (None, False):
        raise ValueError("Official M3 evidence was produced from a dirty worktree")
    if manifest.get("official") is not True or manifest.get("schema_version") != 1:
        raise ValueError("M3 evidence manifest is not official")
    hashes = {
        "run_identity_sha256": sha256_file(paths["identity"]),
        "episodes_sha256": sha256_file(paths["episodes"]),
        "summary_sha256": sha256_file(paths["summary"]),
    }
    for field, digest in hashes.items():
        if manifest.get(field) != digest:
            raise ValueError(f"M3 manifest {field} does not match")
    for field in (
        "selected_checkpoint_sha256",
        "pool_manifest_sha256",
        "m2_baseline_sha256",
    ):
        if manifest.get(field) != identity.get(field):
            raise ValueError(f"M3 manifest {field} differs from the run identity")
    rows = load_episode_rows(paths["episodes"])
    recomputed = evaluate_m3_records(
        rows,
        historical_policy_ids=identity["historical_policy_ids"],
        expected_scenario_hash=str(identity["scenario_hash"]),
    )
    recomputed_payload = recomputed.to_dict()
    if recomputed_payload != stored_summary:
        raise ValueError("M3 summary differs from raw-row recomputation")
    failed_gates = sorted(
        name for name, passed in recomputed.gates.items() if not passed
    )
    failed_integrity = sorted(_INTEGRITY_GATES.intersection(failed_gates))
    if failed_integrity:
        raise ValueError(
            f"Recomputed M3 evidence failed integrity gates: {failed_integrity}"
        )
    if require_capability_pass and not recomputed.passed:
        raise ValueError("Recomputed M3 evidence did not pass")
    if manifest.get("episodes") != recomputed.episodes:
        raise ValueError("M3 manifest episode count does not match raw rows")
    if artifact_root is not None:
        _audit_repository_bindings(identity, root=artifact_root.expanduser().resolve())
    return {
        "episodes": recomputed.episodes,
        "official": True,
        "integrity_passed": True,
        "capability_passed": recomputed.passed,
        "passed": recomputed.passed,
        "failed_gates": failed_gates,
        "pool_size": recomputed.pool_size,
        "selected_checkpoint_sha256": identity["selected_checkpoint_sha256"],
    }
