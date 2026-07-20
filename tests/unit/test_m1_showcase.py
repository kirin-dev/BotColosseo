import json
from pathlib import Path

import pytest

from botcolosseo.demo.m1_showcase import (
    arena_labels,
    load_m1_summary,
    overlay_text,
    select_frame_indices,
)
from botcolosseo.scenarios.regions import RegionGraph


def test_summary_loader_requires_official_passing_evidence(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    for payload in ({"official": False, "passed": True}, {"official": True, "passed": False}):
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            load_m1_summary(path)


def test_arena_labels_cover_all_regions_and_routes() -> None:
    graph = RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml"))

    labels = arena_labels(graph)

    assert labels["regions"] == {region.name for region in graph.regions}
    assert labels["routes"] == {route.name for route in graph.routes}


def test_frame_selection_is_deterministic_and_capped() -> None:
    first = select_frame_indices(113, cap=40)

    assert first == select_frame_indices(113, cap=40)
    assert len(first) == 40
    assert first[0] == 0
    assert first[-1] == 112
    assert select_frame_indices(3, cap=40) == (0, 1, 2)


def test_overlay_contains_public_fields_without_coordinates() -> None:
    text = overlay_text(
        task="moving_hit", teacher="aggressive_script", event="valid_hit", success=True
    )

    assert "moving_hit" in text
    assert "aggressive_script" in text
    assert "valid_hit" in text
    assert "success=True" in text
    assert "position" not in text.lower()
    assert "coordinate" not in text.lower()
