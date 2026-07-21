import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from botcolosseo.evaluation.crossplay import CrossplayRow, write_crossplay_csv_atomic
from botcolosseo.evaluation.m3_figures import (
    load_crossplay_evidence,
    load_pool_history,
    render_crossplay_heatmap,
    render_evidence_bundle,
    render_pool_history,
)


def _rows() -> list[CrossplayRow]:
    rows = []
    for left, right in (
        ("policy-a", "policy-a"),
        ("policy-a", "policy-b"),
        ("policy-b", "policy-b"),
    ):
        for pair in range(5):
            for side in ("host", "opponent"):
                difference = 0 if left == right else 1
                rows.append(
                    CrossplayRow(
                        left_policy=left,
                        right_policy=right,
                        split="validation",
                        pair_index=pair,
                        seed=100 + pair,
                        left_side=side,
                        outcome="draw" if difference == 0 else "win",
                        left_objective_completed=True,
                        right_objective_completed=left == right,
                        left_score=1,
                        right_score=1 - difference,
                        decisions=20,
                        terminated=True,
                        truncated=False,
                        peer_tic_lag_max=0,
                        protocol_inconsistent=False,
                        action_tic_inconsistent=False,
                        score_event_inconsistent=False,
                        scenario_hash="scenario",
                        environment_attempts=1,
                    )
                )
    return rows


def _history(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "snapshots": [
            {
                "environment_steps": 0,
                "pool_size": 2,
                "pfsp_probabilities": {"policy-a": 0.75, "policy-b": 0.25},
            },
            {
                "environment_steps": 200_000,
                "pool_size": 3,
                "pfsp_probabilities": {"policy-a": 0.25, "policy-b": 0.75},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_crossplay_loader_has_deterministic_labels_and_matrix_order(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "crossplay.csv"
    write_crossplay_csv_atomic(tuple(reversed(_rows())), csv_path)

    evidence = load_crossplay_evidence(csv_path)

    assert evidence.policy_ids == ("policy-a", "policy-b")
    assert evidence.win_rate == ((0.0, 1.0), (0.0, 0.0))
    assert evidence.executed_rows == 30


def test_crossplay_loader_fails_on_missing_or_conflicting_cells(tmp_path: Path) -> None:
    rows = _rows()
    missing = tmp_path / "missing.csv"
    write_crossplay_csv_atomic(rows[:-1], missing)
    with pytest.raises(ValueError, match="complete"):
        load_crossplay_evidence(missing)

    conflicting = tmp_path / "conflicting.csv"
    write_crossplay_csv_atomic([*rows, replace(rows[-1], left_score=2, outcome="win")], conflicting)
    with pytest.raises(ValueError, match="duplicate"):
        load_crossplay_evidence(conflicting)


def test_figures_have_frozen_dimensions_and_bundle_metrics_match_raw_csv(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "crossplay.csv"
    write_crossplay_csv_atomic(_rows(), csv_path)
    history_path = _history(tmp_path / "history.json")
    crossplay = load_crossplay_evidence(csv_path)
    history = load_pool_history(history_path)
    heatmap = tmp_path / "heatmap.png"
    pool_figure = tmp_path / "pool.png"

    render_crossplay_heatmap(crossplay, heatmap)
    render_pool_history(history, pool_figure)
    bundle = render_evidence_bundle(
        crossplay_csv=csv_path,
        pool_history_path=history_path,
        heatmap_output=tmp_path / "bundle-heatmap.png",
        pool_output=tmp_path / "bundle-pool.png",
        matrix_output=tmp_path / "matrix.json",
    )

    assert Image.open(heatmap).size == (1440, 1120)
    assert Image.open(pool_figure).size == (1440, 800)
    assert bundle["policy_ids"] == list(crossplay.policy_ids)
    assert bundle["executed_rows"] == crossplay.executed_rows
    assert bundle["win_rate"] == {
        "policy-a": {"policy-a": 0.0, "policy-b": 1.0},
        "policy-b": {"policy-a": 0.0, "policy-b": 0.0},
    }
    assert json.loads((tmp_path / "matrix.json").read_text(encoding="utf-8")) == bundle
