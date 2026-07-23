from __future__ import annotations

import csv
import json
import math
import random
import re
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

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
_PARTICIPANT_INSTRUCTIONS = """# BotColosseo 匿名风格判断

## 你要做什么

观看 6 段匿名视频，判断每段视频中“第一视角 Bot”的行为风格。画面中出现的
人形角色是对手，不是需要判断的 Bot。每段可选 `aggressive`（进攻型）、
`defensive`（防守型）、`explorer`（探索型）或 `unsure`（无法判断）。

## 游戏任务

这是 1v1 的 Crystal Run。Bot 需要找到能量核心、拾取核心并送回己方区域得分；
先得到 3 分的一方获胜。携带核心的角色被消灭时会掉落核心。

## Bot 能做的动作

- 前进、后退、左右平移；
- 左转、右转，以及边前进边转向；
- 原地攻击、前进攻击、转向攻击。

攻击只有实际命中才会降低对手血量。需要持续有效命中直至 `OPP HP` 降到 0
才能消灭对手；伤害会因命中情况变化，因此没有固定的“几枪必杀”。死亡角色
随后会以满血复活。

## 怎么看画面

- `SELF HP`：第一视角 Bot 的血量；
- `OPP HP`：对手血量；
- `CORE SELF / OPP / FREE`：核心由 Bot / 对手携带，或处于自由状态；
- `SCORE`：Bot-对手比分；
- `ACTION`：第一视角 Bot 当前执行的动作；
- `EVENT`：命中、拾取、掉落、死亡、复活或得分等中立事件。

`OPP HP`、核心归属和事件标签仅供观看者理解，未提供给 Bot 的策略网络。

## 每段视频填写

1. `style_choice`：四个风格选项之一；
2. `style_clarity`：风格清晰度，1（很不清楚）到 5（很清楚）；
3. `perceived_difficulty`：你认为该对手有多难打，1（很容易）到 5（很难）。

可以多次选择同一风格。请独立判断，不要查看文件名之外的项目资料。
"""


def prepare_user_study(
    clips: Mapping[str, Sequence[Path]],
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
    clip_count = len(STYLES) * 2
    opaque_tokens = iter(rng.sample(range(10_000, 100_000), clip_count))
    clip_rows: list[dict[str, object]] = []
    style_to_clips: dict[str, list[str]] = {}
    seen_sources: set[Path] = set()
    for style in STYLES:
        sources = tuple(clips[style])
        if len(sources) != 2:
            raise ValueError(f"User study requires two {style} clips")
        style_to_clips[style] = []
        for source_path in sources:
            source = source_path.expanduser().resolve()
            if (
                not source.is_file()
                or source.suffix.lower() != ".mp4"
                or source in seen_sources
            ):
                raise ValueError(f"User study clip is not a unique MP4: {source}")
            seen_sources.add(source)
            clip_id = f"clip-{next(opaque_tokens)}"
            target = clip_dir / f"{clip_id}.mp4"
            shutil.copyfile(source, target)
            style_to_clips[style].append(clip_id)
            clip_rows.append(
                {
                    "clip_id": clip_id,
                    "true_style": style,
                    "path": target.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(target),
                    "bytes": target.stat().st_size,
                }
            )

    base_order = [
        clip_id for style in STYLES for clip_id in style_to_clips[style]
    ]
    rng.shuffle(base_order)
    assignments: list[dict[str, object]] = []
    for index in range(assignment_count):
        assignment_id = f"assignment-{index + 1:03d}"
        block, rotation = divmod(index, clip_count)
        cycle = base_order if block % 2 == 0 else list(reversed(base_order))
        order = cycle[rotation:] + cycle[:rotation]
        assignments.append(
            {
                "assignment_id": assignment_id,
                "clip_order": order,
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
    instructions = output_dir / "participant-instructions.md"
    instructions.write_text(_PARTICIPANT_INSTRUCTIONS, encoding="utf-8")

    answer_key = output_dir / "answer-key.json"
    _write_json(
        answer_key,
        {
            "schema_version": 2,
            "clips": clip_rows,
            "do_not_share_until_collection_closes": True,
        },
    )
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": 2,
        "stage": "m6-user-study-package",
        "seed": seed,
        "assignment_count": assignment_count,
        "styles": list(STYLES),
        "clips_per_style": 2,
        "clip_count": clip_count,
        "choices": list(STYLE_CHOICES),
        "response_fields": list(RESPONSE_FIELDS),
        "assignments_sha256": sha256_file(public_assignments),
        "response_template_sha256": sha256_file(response_template),
        "participant_instructions_sha256": sha256_file(instructions),
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
    *,
    synthetic_data: bool = False,
) -> dict[str, object]:
    package_dir = package_dir.expanduser().resolve()
    responses_path = responses_path.expanduser().resolve()
    manifest = _read_json(package_dir / "manifest.json")
    key = _read_json(package_dir / "answer-key.json")
    _validate_package(package_dir, manifest, key)
    clip_to_style = {
        str(row["clip_id"]): str(row["true_style"])
        for row in _require_rows(key.get("clips"), "answer key clips")
    }
    assignments = _load_assignments(
        package_dir / "assignments.csv",
        expected_clip_ids=set(clip_to_style),
    )
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
        "synthetic_data": synthetic_data,
        "human_participants": not synthetic_data,
        "test_cases_accessed": False,
    }


def render_user_study_chart(
    summary: Mapping[str, object],
    output: Path,
) -> Path:
    if (
        summary.get("stage") != "m6-user-study-analysis"
        or summary.get("test_cases_accessed") is not False
    ):
        raise ValueError("User-study chart requires audited analysis")
    per_style = summary.get("per_style")
    confusion = summary.get("confusion_matrix")
    if not isinstance(per_style, Mapping) or not isinstance(confusion, Mapping):
        raise ValueError("User-study chart data are incomplete")
    rates: list[float] = []
    intervals: list[tuple[float, float]] = []
    matrix: list[list[int]] = []
    for style in STYLES:
        row = per_style.get(style)
        counts = confusion.get(style)
        if not isinstance(row, Mapping) or not isinstance(counts, Mapping):
            raise ValueError("User-study chart style data are incomplete")
        rate = row.get("recognition_rate")
        interval = row.get("recognition_wilson_95")
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not isinstance(interval, list)
            or len(interval) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in interval
            )
        ):
            raise ValueError("User-study chart rates are invalid")
        rates.append(float(rate))
        intervals.append((float(interval[0]), float(interval[1])))
        count_row = []
        for choice in STYLE_CHOICES:
            value = counts.get(choice)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("User-study confusion counts are invalid")
            count_row.append(value)
        matrix.append(count_row)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    figure = Figure(figsize=(11, 4.2), dpi=150, facecolor="white")
    FigureCanvasAgg(figure)
    try:
        recognition = figure.add_subplot(1, 2, 1)
        lower = [
            rate - interval[0]
            for rate, interval in zip(rates, intervals, strict=True)
        ]
        upper = [
            interval[1] - rate
            for rate, interval in zip(rates, intervals, strict=True)
        ]
        recognition.bar(
            STYLES,
            rates,
            yerr=(lower, upper),
            capsize=4,
            color=("#ef4444", "#3b82f6", "#10b981"),
        )
        recognition.set_ylim(0, 1)
        recognition.set_ylabel("Recognition rate")
        title_prefix = "Synthetic preflight" if summary.get("synthetic_data") else "Blind"
        recognition.set_title(f"{title_prefix} style recognition (Wilson 95% CI)")
        recognition.grid(axis="y", alpha=0.2)
        for index, rate in enumerate(rates):
            recognition.text(index, min(rate + 0.04, 0.96), f"{rate:.0%}", ha="center")

        confusion_axis = figure.add_subplot(1, 2, 2)
        confusion_axis.imshow(matrix, cmap="Blues", aspect="auto")
        confusion_axis.set_xticks(range(len(STYLE_CHOICES)), STYLE_CHOICES, rotation=25)
        confusion_axis.set_yticks(range(len(STYLES)), STYLES)
        confusion_axis.set_xlabel("Selected label")
        confusion_axis.set_ylabel("True style")
        confusion_axis.set_title("Confusion matrix (counts)")
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                confusion_axis.text(
                    column_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    color="black",
                )
        if summary.get("synthetic_data") is True:
            figure.text(
                0.5,
                0.01,
                "SYNTHETIC PREFLIGHT — NO HUMAN PARTICIPANTS",
                ha="center",
                color="#b91c1c",
                weight="bold",
            )
        figure.tight_layout()
        figure.savefig(temporary, facecolor=figure.get_facecolor())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        figure.clear()
    return output


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
        or manifest.get("participant_instructions_sha256")
        != sha256_file(package_dir / "participant-instructions.md")
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


def _load_assignments(
    path: Path,
    *,
    expected_clip_ids: set[str],
) -> dict[str, tuple[str, ...]]:
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
    if not assignments or any(
        set(order) != expected_clip_ids or len(order) != len(expected_clip_ids)
        for order in assignments.values()
    ):
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
