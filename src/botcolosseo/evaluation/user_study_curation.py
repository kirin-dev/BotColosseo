from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.envs.video import read_video_frames, write_mp4
from botcolosseo.evaluation.showcase import canonical_json
from botcolosseo.evaluation.user_study import STYLES


def curate_user_study_clips(
    config_path: Path,
    *,
    output_dir: Path,
    root: Path,
) -> dict[str, object]:
    config_path = (root / config_path).resolve()
    config = _read_json(config_path)
    selections = config.get("clips")
    if (
        config.get("schema_version") != 1
        or not isinstance(selections, list)
        or len(selections) != 6
        or any(not isinstance(row, dict) for row in selections)
    ):
        raise ValueError("User-study curation config is invalid")
    identities = [(row.get("style"), row.get("variant")) for row in selections]
    expected = [(style, variant) for style in STYLES for variant in (1, 2)]
    if identities != expected:
        raise ValueError("User-study curation must contain two ordered clips per style")

    output_dir = (root / output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite curated user-study clips")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for selection in selections:
        output_rows.append(
            _curate_one(selection, output_dir=output_dir, root=root)
        )
    manifest = {
        "schema_version": 1,
        "stage": "m6-curated-validation-clips",
        "selection": config["selection"],
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "clips_per_style": 2,
        "clip_count": 6,
        "clips": output_rows,
        "observer_fields_not_available_to_policy": True,
        "observer_only_fields": [
            "opponent_health",
            "core_owner",
            "neutral_event_labels",
        ],
        "test_cases_accessed": False,
    }
    (output_dir / "manifest.json").write_bytes(canonical_json(manifest))
    return manifest


def _curate_one(
    selection: Mapping[str, object],
    *,
    output_dir: Path,
    root: Path,
) -> dict[str, object]:
    source_manifest_path = (root / str(selection["source_manifest"])).resolve()
    if sha256_file(source_manifest_path) != selection.get("source_manifest_sha256"):
        raise ValueError("Candidate manifest hash does not match curation config")
    source_manifest = _read_json(source_manifest_path)
    if (
        source_manifest.get("stage") != "m6-user-study-candidates"
        or source_manifest.get("test_cases_accessed") is not False
    ):
        raise ValueError("Candidate manifest is not eligible for curation")
    candidates = source_manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Candidate manifest has no candidate rows")
    matches = [
        row
        for row in candidates
        if row.get("style") == selection.get("style")
        and row.get("rank") == selection.get("source_rank")
        and row.get("case_id") == selection.get("case_id")
        and row.get("path") == selection.get("source_path")
        and row.get("sha256") == selection.get("source_sha256")
    ]
    if len(matches) != 1:
        raise ValueError("Curated clip identity is absent from candidate manifest")
    candidate = matches[0]
    source_path = source_manifest_path.parent / str(selection["source_path"])
    if sha256_file(source_path) != selection.get("source_sha256"):
        raise ValueError("Candidate video hash does not match curation config")
    frames = read_video_frames(source_path)
    if len(frames) != candidate.get("frame_count"):
        raise ValueError("Candidate video frame count has drifted")
    fps = candidate.get("fps")
    start = selection.get("start_frame")
    end = selection.get("end_frame")
    if (
        type(fps) is not int
        or type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(frames)
    ):
        raise ValueError("Curated clip frame window is invalid")
    duration = (end - start) / fps
    if not 24.5 <= duration <= 40:
        raise ValueError("Curated clip duration must be between 24.5 and 40 seconds")

    style = str(selection["style"])
    variant = int(selection["variant"])
    target = output_dir / f"{style}-{variant}.mp4"
    write_mp4(frames[start:end], target, fps=fps)
    steps = candidate.get("replay", {}).get("observer_steps")
    if not isinstance(steps, list) or len(steps) != len(frames):
        raise ValueError("Candidate observer trace is incomplete")
    selected_steps = steps[start:end]
    events = Counter(
        event
        for step in selected_steps
        for event in step.get("events", [])
        if isinstance(event, str)
    )
    return {
        "style": style,
        "variant": variant,
        "path": target.relative_to(output_dir).as_posix(),
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
        "fps": fps,
        "frame_count": end - start,
        "duration_seconds": duration,
        "source_manifest": selection["source_manifest"],
        "source_manifest_sha256": selection["source_manifest_sha256"],
        "source_path": selection["source_path"],
        "source_sha256": selection["source_sha256"],
        "source_case_id": selection["case_id"],
        "source_rank": selection["source_rank"],
        "source_frame_window": [start, end],
        "selection_reason": selection["reason"],
        "visible_summary": {
            "events": dict(sorted(events.items())),
            "minimum_self_health": min(
                int(step["self_health"]) for step in selected_steps
            ),
            "minimum_opponent_health": min(
                int(step["opponent_health"]) for step in selected_steps
            ),
        },
        "protocol_inconsistent": candidate["replay"]["protocol_inconsistent"],
    }


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload
