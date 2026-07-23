from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.evaluation.showcase import canonical_json
from botcolosseo.evaluation.user_study import (
    RESPONSE_FIELDS,
    STYLE_CHOICES,
    STYLES,
)


def generate_synthetic_user_study(
    package_dir: Path,
    config_path: Path,
    *,
    responses_path: Path,
    provenance_path: Path,
) -> dict[str, object]:
    package_dir = package_dir.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    responses_path = responses_path.expanduser().resolve()
    provenance_path = provenance_path.expanduser().resolve()
    config = _read_json(config_path)
    package = _read_json(package_dir / "manifest.json")
    answer_key_path = package_dir / "answer-key.json"
    answer_key = _read_json(answer_key_path)
    assignments_path = package_dir / "assignments.csv"
    if (
        config.get("schema_version") != 1
        or package.get("stage") != "m6-user-study-package"
        or package.get("test_cases_accessed") is not False
        or package.get("answer_key_sha256") != sha256_file(answer_key_path)
        or package.get("assignments_sha256") != sha256_file(assignments_path)
    ):
        raise ValueError("Synthetic user-study source identity is invalid")
    respondent_count = config.get("respondent_count")
    plans = config.get("clips")
    if (
        type(respondent_count) is not int
        or respondent_count <= 0
        or not isinstance(plans, list)
        or len(plans) != 6
        or any(not isinstance(plan, dict) for plan in plans)
    ):
        raise ValueError("Synthetic user-study config is invalid")
    plan_lookup = _validate_plans(plans, respondent_count)
    clip_lookup = _clip_lookup(answer_key)
    assignments = _assignments(assignments_path)
    if len(assignments) < respondent_count:
        raise ValueError("Synthetic study has fewer assignments than respondents")

    rows = []
    for respondent_index, assignment_id in enumerate(
        sorted(assignments)[:respondent_count]
    ):
        for clip_id in assignments[assignment_id]:
            style, variant = clip_lookup[clip_id]
            plan = plan_lookup[(style, variant)]
            rows.append(
                {
                    "respondent_id": f"synthetic-r{respondent_index + 1:02d}",
                    "assignment_id": assignment_id,
                    "clip_id": clip_id,
                    "style_choice": plan["style_choices"][respondent_index],
                    "style_clarity": plan["style_clarity"][respondent_index],
                    "perceived_difficulty": plan["perceived_difficulty"][
                        respondent_index
                    ],
                }
            )
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    with responses_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESPONSE_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "schema_version": 1,
        "stage": "m6-synthetic-user-study-preflight",
        "synthetic_data": True,
        "human_participants": False,
        "respondent_count": respondent_count,
        "response_count": len(rows),
        "purpose": config["purpose"],
        "generated_on": config["generated_on"],
        "seed": config["seed"],
        "config_path": _portable_source_path(config_path),
        "config_sha256": sha256_file(config_path),
        "package_manifest_sha256": sha256_file(package_dir / "manifest.json"),
        "answer_key_sha256": sha256_file(answer_key_path),
        "assignments_sha256": sha256_file(assignments_path),
        "responses_sha256": sha256_file(responses_path),
        "clip_mapping": [
            {
                "clip_id": clip_id,
                "true_style": style,
                "variant": variant,
            }
            for clip_id, (style, variant) in clip_lookup.items()
        ],
        "test_cases_accessed": False,
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_bytes(canonical_json(provenance))
    return provenance


def _validate_plans(
    plans: list[dict[str, object]],
    respondent_count: int,
) -> dict[tuple[str, int], dict[str, list[object]]]:
    result = {}
    expected = {(style, variant) for style in STYLES for variant in (1, 2)}
    for plan in plans:
        style = plan.get("style")
        variant = plan.get("variant")
        identity = (style, variant)
        if style not in STYLES or type(variant) is not int or identity in result:
            raise ValueError("Synthetic clip-plan identity is invalid")
        choices = plan.get("style_choices")
        clarity = plan.get("style_clarity")
        difficulty = plan.get("perceived_difficulty")
        if (
            not isinstance(choices, list)
            or len(choices) != respondent_count
            or any(choice not in STYLE_CHOICES for choice in choices)
            or not _valid_ratings(clarity, respondent_count)
            or not _valid_ratings(difficulty, respondent_count)
        ):
            raise ValueError("Synthetic clip-plan responses are invalid")
        result[(style, variant)] = {
            "style_choices": choices,
            "style_clarity": clarity,
            "perceived_difficulty": difficulty,
        }
    if set(result) != expected:
        raise ValueError("Synthetic study must plan two clips per style")
    return result


def _valid_ratings(value: object, count: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == count
        and all(type(rating) is int and 1 <= rating <= 5 for rating in value)
    )


def _clip_lookup(answer_key: Mapping[str, object]) -> dict[str, tuple[str, int]]:
    rows = answer_key.get("clips")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("Synthetic study answer key is invalid")
    style_counts: dict[str, int] = defaultdict(int)
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Synthetic study answer-key row is invalid")
        clip_id = row.get("clip_id")
        style = row.get("true_style")
        if not isinstance(clip_id, str) or style not in STYLES or clip_id in result:
            raise ValueError("Synthetic study clip identity is invalid")
        style_counts[style] += 1
        result[clip_id] = (style, style_counts[style])
    if set(style_counts) != set(STYLES) or set(style_counts.values()) != {2}:
        raise ValueError("Synthetic study requires two answer-key clips per style")
    return result


def _assignments(path: Path) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("assignment_id", "position", "clip_id"):
            raise ValueError("Synthetic study assignment schema is invalid")
        for row in reader:
            grouped[row["assignment_id"]].append(
                (int(row["position"]), row["clip_id"])
            )
    return {
        assignment_id: tuple(
            clip_id for _, clip_id in sorted(rows, key=lambda item: item[0])
        )
        for assignment_id, rows in grouped.items()
    }


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _portable_source_path(path: Path) -> str:
    for parent in path.parents:
        if (parent / ".git").exists():
            return path.relative_to(parent).as_posix()
    return path.name
