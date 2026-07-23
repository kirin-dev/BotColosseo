from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from botcolosseo.agents.league_opponents import sha256_file

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TOP_FIELDS = {
    "schema_version",
    "stage",
    "test_cases_accessed",
    "difficulty",
    "cases",
    "scenario",
    "sources",
}
_ARTIFACT_FIELDS = {"path", "expected_sha256"}
_SCENARIO_FIELDS = {"path", "expected_sha256", "wad_sha256"}
_BASE_FIELDS = {"manifest", "expected_sha256"}
_HYBRID_FIELDS = {
    "governor_config",
    "governor_config_sha256",
    "hard_manifest",
    "hard_manifest_sha256",
}


@dataclass(frozen=True)
class HybridDifficultyStyleSource:
    style: Literal["defensive", "explorer"]
    governor_config: Path
    governor_config_sha256: str
    hard_manifest: Path
    hard_manifest_sha256: str


@dataclass(frozen=True)
class HybridDifficultyProductConfig:
    config_path: Path
    config_sha256: str
    difficulty_config: Path
    difficulty_config_sha256: str
    cases: Path
    cases_sha256: str
    scenario_manifest: Path
    scenario_manifest_sha256: str
    scenario_hash: str
    base_aggressive_manifest: Path
    base_aggressive_manifest_sha256: str
    defensive: HybridDifficultyStyleSource
    explorer: HybridDifficultyStyleSource
    test_cases_accessed: Literal[False]


def load_hybrid_difficulty_product_config(
    path: Path,
    *,
    root: Path,
) -> HybridDifficultyProductConfig:
    root = root.resolve()
    config_path = _resolve(root, path)
    payload_bytes = config_path.read_bytes()
    payload = yaml.safe_load(payload_bytes)
    _fields(payload, _TOP_FIELDS, "Hybrid difficulty config")
    if (
        payload["schema_version"] != 1
        or payload["stage"] != "m5_hybrid_all_style_difficulty"
        or payload["test_cases_accessed"] is not False
    ):
        raise ValueError("Hybrid difficulty publication boundary is invalid")

    difficulty_path, difficulty_hash = _artifact(
        payload["difficulty"],
        root=root,
        label="difficulty config",
    )
    cases_path, cases_hash = _artifact(
        payload["cases"],
        root=root,
        label="validation cases",
    )
    scenario = payload["scenario"]
    _fields(scenario, _SCENARIO_FIELDS, "Hybrid difficulty scenario")
    scenario_path = _path(root, scenario["path"], "scenario manifest")
    scenario_manifest_hash = _hash(
        scenario["expected_sha256"],
        "scenario manifest",
    )
    scenario_hash = _hash(scenario["wad_sha256"], "scenario WAD")
    _verify(scenario_path, scenario_manifest_hash, "scenario manifest")
    scenario_payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(scenario_payload, dict) or scenario_payload.get(
        "wad_sha256"
    ) != scenario_hash:
        raise ValueError("Hybrid difficulty scenario identity is invalid")

    sources = payload["sources"]
    _fields(
        sources,
        {"base_aggressive", "defensive", "explorer"},
        "Hybrid difficulty sources",
    )
    base = sources["base_aggressive"]
    _fields(base, _BASE_FIELDS, "Hybrid difficulty Base/Aggressive source")
    base_manifest = _path(root, base["manifest"], "Base/Aggressive manifest")
    base_manifest_hash = _hash(base["expected_sha256"], "Base/Aggressive manifest")
    _verify(base_manifest, base_manifest_hash, "Base/Aggressive manifest")
    defensive = _style_source("defensive", sources["defensive"], root=root)
    explorer = _style_source("explorer", sources["explorer"], root=root)
    return HybridDifficultyProductConfig(
        config_path=config_path,
        config_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        difficulty_config=difficulty_path,
        difficulty_config_sha256=difficulty_hash,
        cases=cases_path,
        cases_sha256=cases_hash,
        scenario_manifest=scenario_path,
        scenario_manifest_sha256=scenario_manifest_hash,
        scenario_hash=scenario_hash,
        base_aggressive_manifest=base_manifest,
        base_aggressive_manifest_sha256=base_manifest_hash,
        defensive=defensive,
        explorer=explorer,
        test_cases_accessed=False,
    )


def _style_source(
    style: Literal["defensive", "explorer"],
    payload: object,
    *,
    root: Path,
) -> HybridDifficultyStyleSource:
    _fields(payload, _HYBRID_FIELDS, f"Hybrid difficulty {style} source")
    config = _path(root, payload["governor_config"], f"{style} governor config")
    config_hash = _hash(
        payload["governor_config_sha256"],
        f"{style} governor config",
    )
    manifest = _path(root, payload["hard_manifest"], f"{style} Hard manifest")
    manifest_hash = _hash(
        payload["hard_manifest_sha256"],
        f"{style} Hard manifest",
    )
    _verify(config, config_hash, f"{style} governor config")
    _verify(manifest, manifest_hash, f"{style} Hard manifest")
    return HybridDifficultyStyleSource(
        style=style,
        governor_config=config,
        governor_config_sha256=config_hash,
        hard_manifest=manifest,
        hard_manifest_sha256=manifest_hash,
    )


def _artifact(
    payload: object,
    *,
    root: Path,
    label: str,
) -> tuple[Path, str]:
    _fields(payload, _ARTIFACT_FIELDS, f"Hybrid difficulty {label}")
    path = _path(root, payload["path"], label)
    digest = _hash(payload["expected_sha256"], label)
    _verify(path, digest, label)
    return path, digest


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Hybrid difficulty {label} path is invalid")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Hybrid difficulty {label} path is not repository-local")
    return path


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"Hybrid difficulty {label} SHA-256 is invalid")
    return value


def _verify(path: Path, expected: str, label: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"Hybrid difficulty {label} hash drifted")


def _fields(payload: object, expected: set[str], label: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{label} fields do not match schema")
