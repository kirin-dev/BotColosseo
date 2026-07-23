from __future__ import annotations

import csv
import itertools
import json
import math
import random
import re
import shutil
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from botcolosseo.agents.league_opponents import sha256_file

STYLES = ("aggressive", "defensive", "explorer")
STYLE_CHOICES = (*STYLES, "unsure")
RESPONSE_FIELDS = (
    "respondent_id",
    "assignment_id",
    "clip_id",
    "style_choice",
    "style_clarity",
    "perceived_difficulty",
)
_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z")


def prepare_user_study(
    clips: Mapping[str, Path],
    *,
    output_dir: Path,
    assignment_count: int = 10,
    seed: int = 20260723,
) -> dict[str, object]:
    if tuple(clips) != STYLES:
        raise ValueError("User study clips must use the frozen style order")
    if assignment_count <= 0:
        raise ValueError("User study requires at least one assignment")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a user-study package")
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = output_dir / "clips"
    clip_dir.mkdir()

    rng = random.Random(seed)
    opaque_tokens = rng.sample(range(10_000, 100_000), len(STYLES))
    clip_rows: list[dict[str, object]] = []
    style_to_clip: dict[str, str] = {}
    for style, token in zip(STYLES, opaque_tokens, strict=True):
        source = clips[style].expanduser().resolve()
        if not source.is_file() or source.suffix.lower() != ".mp4":
            raise ValueError(f"User study clip is not an MP4: {source}")
        clip_id = f"clip-{token}"
        target = clip_dir / f"{clip_id}.mp4"
        shutil.copyfile(source, target)
        style_to_clip[style] = clip_id
        clip_rows.append(
            {
                "clip_id": clip_id,
                "true_style": style,
                "path": target.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            }
        )

    permutations = list(itertools.permutations(STYLES))
    rng.shuffle(permutations)
    assignments: list[dict[str, object]] = []
    for index in range(assignment_count):
        assignment_id = f"assignment-{index + 1:03d}"
        order = permutations[index % len(permutations)]
        assignments.append(
            {
                "assignment_id": assignment_id,
                "clip_order": [style_to_clip[style] for style in order],
            }
        )

    public_assignments = output_dir / "assignments.csv"
    with public_assignments.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("assignment_id", "position", "clip_id")
        )
        writer.writeheader()
        for assignment in assignments:
            for position, clip_id in enumerate(
                assignment["clip_order"], start=1  # type: ignore[arg-type]
            ):
                writer.writerow(
                    {
                        "assignment_id": assignment["assignment_id"],
                        "position": position,
                        "clip_id": clip_id,
                    }
                )

    response_template = output_dir / "response-template.csv"
    with response_template.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=RESPONSE_FIELDS).writeheader()

    answer_key = output_dir / "answer-key.json"
    _write_json(
        answer_key,
        {
            "schema_version": 1,
            "clips": clip_rows,
            "do_not_share_until_collection_closes": True,
        },
    )
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "stage": "m6-user-study-package",
        "seed": seed,
        "assignment_count": assignment_count,
        "styles": list(STYLES),
        "choices": list(STYLE_CHOICES),
        "response_fields": list(RESPONSE_FIELDS),
        "assignments_sha256": sha256_file(public_assignments),
        "response_template_sha256": sha256_file(response_template),
        "answer_key_sha256": sha256_file(answer_key),
        "clips": [
            {key: value for key, value in row.items() if key != "true_style"}
            for row in clip_rows
        ],
        "test_cases_accessed": False,
    }
    _write_json(manifest_path, manifest)
    return manifest


def analyze_user_study(
    package_dir: Path,
    responses_path: Path,
) -> dict[str, object]:
    package_dir = package_dir.expanduser().resolve()
    responses_path = responses_path.expanduser().resolve()
    manifest = _read_json(package_dir / "manifest.json")
    key = _read_json(package_dir / "answer-key.json")
    _validate_package(package_dir, manifest, key)
    assignments = _load_assignments(package_dir / "assignments.csv")
    clip_to_style = {
        str(row["clip_id"]): str(row["true_style"])
        for row in _require_rows(key.get("clips"), "answer key clips")
    }
    responses = _load_responses(responses_path, assignments, clip_to_style)

    confusion = {
        style: {choice: 0 for choice in STYLE_CHOICES} for style in STYLES
    }
    clarity: dict[str, list[int]] = defaultdict(list)
    difficulty: dict[str, list[int]] = defaultdict(list)
    for row in responses:
        true_style = clip_to_style[row["clip_id"]]
        confusion[true_style][row["style_choice"]] += 1
        clarity[true_style].append(int(row["style_clarity"]))
        difficulty[true_style].append(int(row["perceived_difficulty"]))

    per_style: dict[str, dict[str, object]] = {}
    correct_total = 0
    for style in STYLES:
        count = sum(confusion[style].values())
        correct = confusion[style][style]
        correct_total += correct
        low, high = _wilson(correct, count)
        per_style[style] = {
            "responses": count,
            "correct": correct,
            "recognition_rate": correct / count,
            "recognition_wilson_95": [low, high],
            "mean_style_clarity": sum(clarity[style]) / count,
            "mean_perceived_difficulty": sum(difficulty[style]) / count,
        }
    respondent_count = len({row["respondent_id"] for row in responses})
    macro = sum(
        float(per_style[style]["recognition_rate"]) for style in STYLES
    ) / len(STYLES)
    low, high = _wilson(correct_total, len(responses))
    return {
        "schema_version": 1,
        "stage": "m6-user-study-analysis",
        "respondents": respondent_count,
        "responses": len(responses),
        "styles": list(STYLES),
        "confusion_matrix": confusion,
        "per_style": per_style,
        "macro_recognition_rate": macro,
        "micro_recognition_rate": correct_total / len(responses),
        "micro_recognition_wilson_95": [low, high],
        "responses_sha256": sha256_file(responses_path),
        "package_manifest_sha256": sha256_file(package_dir / "manifest.json"),
        "small_sample_product_study": respondent_count < 30,
        "test_cases_accessed": False,
    }


def _validate_package(
    package_dir: Path,
    manifest: Mapping[str, object],
    key: Mapping[str, object],
) -> None:
    if (
        manifest.get("stage") != "m6-user-study-package"
        or manifest.get("test_cases_accessed") is not False
        or manifest.get("answer_key_sha256")
        != sha256_file(package_dir / "answer-key.json")
        or manifest.get("assignments_sha256")
        != sha256_file(package_dir / "assignments.csv")
        or manifest.get("response_template_sha256")
        != sha256_file(package_dir / "response-template.csv")
    ):
        raise ValueError("User-study package identity is invalid")
    keyed = {
        str(row["clip_id"]): row
        for row in _require_rows(key.get("clips"), "answer key clips")
    }
    public_rows = _require_rows(manifest.get("clips"), "manifest clips")
    if set(keyed) != {str(row["clip_id"]) for row in public_rows}:
        raise ValueError("User-study clip identities do not match")
    for row in public_rows:
        clip_id = str(row["clip_id"])
        path = package_dir / str(row["path"])
        if (
            not path.is_file()
            or row.get("sha256") != sha256_file(path)
            or keyed[clip_id].get("sha256") != row.get("sha256")
        ):
            raise ValueError("User-study clip hash is invalid")


def _load_assignments(path: Path) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("assignment_id", "position", "clip_id"):
            raise ValueError("User-study assignment schema is invalid")
        for row in reader:
            grouped[row["assignment_id"]].append(
                (int(row["position"]), row["clip_id"])
            )
    assignments = {
        assignment_id: tuple(
            clip_id for _, clip_id in sorted(rows, key=lambda item: item[0])
        )
        for assignment_id, rows in grouped.items()
    }
    if not assignments or any(len(set(order)) != len(STYLES) for order in assignments.values()):
        raise ValueError("User-study assignments are incomplete")
    return assignments


def _load_responses(
    path: Path,
    assignments: Mapping[str, tuple[str, ...]],
    clip_to_style: Mapping[str, str],
) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RESPONSE_FIELDS:
            raise ValueError("User-study response schema is invalid")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("User-study responses are empty")
    by_respondent: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        respondent = row["respondent_id"]
        assignment = row["assignment_id"]
        clip_id = row["clip_id"]
        if _ID.fullmatch(respondent) is None:
            raise ValueError("Respondent IDs must be anonymous short identifiers")
        if (
            assignment not in assignments
            or clip_id not in assignments[assignment]
            or clip_id not in clip_to_style
            or row["style_choice"] not in STYLE_CHOICES
        ):
            raise ValueError("User-study response has an unknown categorical value")
        for field in ("style_clarity", "perceived_difficulty"):
            try:
                rating = int(row[field])
            except ValueError as error:
                raise ValueError("User-study ratings must be integers") from error
            if str(rating) != row[field] or not 1 <= rating <= 5:
                raise ValueError("User-study ratings must be integers in [1, 5]")
        identity = (respondent, clip_id)
        if identity in seen:
            raise ValueError("User-study response contains a duplicate clip answer")
        seen.add(identity)
        by_respondent[respondent].append(row)
    for respondent, respondent_rows in by_respondent.items():
        assignment_ids = {row["assignment_id"] for row in respondent_rows}
        clip_ids = {row["clip_id"] for row in respondent_rows}
        if len(assignment_ids) != 1:
            raise ValueError(f"Respondent {respondent} used multiple assignments")
        assignment = next(iter(assignment_ids))
        if clip_ids != set(assignments[assignment]):
            raise ValueError(f"Respondent {respondent} has incomplete responses")
    return rows


def _wilson(successes: int, count: int) -> tuple[float, float]:
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("Wilson interval requires a non-empty valid count")
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z * z / count
    centre = proportion + z * z / (2 * count)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / count + z * z / (4 * count * count)
    )
    return (centre - margin) / denominator, (centre + margin) / denominator


def _require_rows(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value or any(
        not isinstance(row, dict) for row in value
    ):
        raise ValueError(f"User-study {label} are invalid")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
