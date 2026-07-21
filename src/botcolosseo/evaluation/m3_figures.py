from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.cli.train_ppo import _atomic_json
from botcolosseo.evaluation.crossplay import CrossplayRow, summarize_payoff_matrix


@dataclass(frozen=True)
class CrossplayFigureEvidence:
    policy_ids: tuple[str, ...]
    executed_rows: int
    win_rate: tuple[tuple[float, ...], ...]
    draw_rate: tuple[tuple[float, ...], ...]
    objective_rate: tuple[tuple[float, ...], ...]
    score_difference: tuple[tuple[float, ...], ...]
    source_sha256: str


@dataclass(frozen=True)
class PoolHistorySnapshot:
    environment_steps: int
    pool_size: int
    pfsp_probabilities: dict[str, float]


@dataclass(frozen=True)
class PoolHistoryEvidence:
    snapshots: tuple[PoolHistorySnapshot, ...]
    policy_ids: tuple[str, ...]
    source_sha256: str


_INTEGER_FIELDS = {
    "pair_index",
    "seed",
    "left_score",
    "right_score",
    "decisions",
    "peer_tic_lag_max",
    "environment_attempts",
}
_BOOLEAN_FIELDS = {
    "left_objective_completed",
    "right_objective_completed",
    "terminated",
    "truncated",
    "protocol_inconsistent",
    "action_tic_inconsistent",
    "score_event_inconsistent",
}


def _boolean(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Invalid cross-play boolean: {value}")


def load_crossplay_evidence(path: Path) -> CrossplayFigureEvidence:
    path = path.expanduser().resolve()
    with path.open(newline="", encoding="utf-8") as source:
        payloads = list(csv.DictReader(source))
    expected_fields = set(CrossplayRow.__dataclass_fields__) | {"score_difference"}
    if not payloads or set(payloads[0]) != expected_fields:
        raise ValueError("Cross-play CSV fields do not match schema")
    rows: list[CrossplayRow] = []
    for payload in payloads:
        values: dict[str, object] = {}
        for field in CrossplayRow.__dataclass_fields__:
            raw = payload[field]
            if field in _INTEGER_FIELDS:
                values[field] = int(raw)
            elif field in _BOOLEAN_FIELDS:
                values[field] = _boolean(raw)
            else:
                values[field] = raw
        row = CrossplayRow(**values)  # type: ignore[arg-type]
        if int(payload["score_difference"]) != row.score_difference:
            raise ValueError("Cross-play CSV score difference does not match")
        rows.append(row)
    policy_ids = tuple(
        sorted({row.left_policy for row in rows} | {row.right_policy for row in rows})
    )
    matrix = summarize_payoff_matrix(rows, policy_ids=policy_ids)

    def values(name: str) -> tuple[tuple[float, ...], ...]:
        cells = matrix[name]
        if not isinstance(cells, dict):
            raise ValueError("Cross-play matrix values are invalid")
        return tuple(
            tuple(float(cells[left][right]) for right in policy_ids)  # type: ignore[index]
            for left in policy_ids
        )

    return CrossplayFigureEvidence(
        policy_ids=policy_ids,
        executed_rows=int(matrix["executed_rows"]),
        win_rate=values("win_rate"),
        draw_rate=values("draw_rate"),
        objective_rate=values("objective_rate"),
        score_difference=values("score_difference"),
        source_sha256=sha256_file(path),
    )


def load_pool_history(path: Path) -> PoolHistoryEvidence:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "snapshots"}
        or payload["schema_version"] != 1
        or not isinstance(payload["snapshots"], list)
        or not payload["snapshots"]
    ):
        raise ValueError("PFSP pool history fields do not match schema")
    snapshots: list[PoolHistorySnapshot] = []
    for row in payload["snapshots"]:
        if not isinstance(row, dict) or set(row) != {
            "environment_steps",
            "pool_size",
            "pfsp_probabilities",
        }:
            raise ValueError("PFSP pool snapshot fields do not match schema")
        probabilities = row["pfsp_probabilities"]
        if (
            type(row["environment_steps"]) is not int
            or row["environment_steps"] < 0
            or type(row["pool_size"]) is not int
            or not 1 <= row["pool_size"] <= 12
            or not isinstance(probabilities, dict)
            or not probabilities
            or any(
                not isinstance(policy_id, str)
                or not policy_id
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
                for policy_id, value in probabilities.items()
            )
            or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-9)
        ):
            raise ValueError("PFSP pool snapshot values are invalid")
        snapshots.append(
            PoolHistorySnapshot(
                environment_steps=row["environment_steps"],
                pool_size=row["pool_size"],
                pfsp_probabilities={
                    key: float(value) for key, value in sorted(probabilities.items())
                },
            )
        )
    if [row.environment_steps for row in snapshots] != sorted(
        {row.environment_steps for row in snapshots}
    ):
        raise ValueError("PFSP pool history steps must be unique and sorted")
    policy_ids = tuple(
        sorted(
            {
                policy_id
                for snapshot in snapshots
                for policy_id in snapshot.pfsp_probabilities
            }
        )
    )
    return PoolHistoryEvidence(tuple(snapshots), policy_ids, sha256_file(path))


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def render_crossplay_heatmap(
    evidence: CrossplayFigureEvidence, output: Path
) -> Path:
    import numpy as np

    plt = _matplotlib()
    matrix = np.asarray(evidence.win_rate, dtype=float)
    fig, axis = plt.subplots(figsize=(9, 7), dpi=160)
    figure = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    axis.set_xticks(range(len(evidence.policy_ids)), evidence.policy_ids, rotation=35, ha="right")
    axis.set_yticks(range(len(evidence.policy_ids)), evidence.policy_ids)
    axis.set_xlabel("Opponent policy")
    axis.set_ylabel("Focal policy")
    axis.set_title("M3 Validation Cross-play Win Rate", loc="left", fontweight="bold")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.0%}",
                ha="center",
                va="center",
                color="black" if 0.2 < matrix[row, column] < 0.8 else "white",
                fontsize=8,
            )
    fig.colorbar(figure, ax=axis, label="Win rate")
    fig.tight_layout()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    fig.savefig(temporary)
    plt.close(fig)
    temporary.replace(output)
    return output


def render_pool_history(evidence: PoolHistoryEvidence, output: Path) -> Path:
    plt = _matplotlib()
    steps = [snapshot.environment_steps / 1_000_000 for snapshot in evidence.snapshots]
    probabilities = [
        [snapshot.pfsp_probabilities.get(policy_id, 0.0) for snapshot in evidence.snapshots]
        for policy_id in evidence.policy_ids
    ]
    fig, axis = plt.subplots(figsize=(9, 5), dpi=160)
    axis.stackplot(steps, *probabilities, labels=evidence.policy_ids, alpha=0.85)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Environment steps (millions)")
    axis.set_ylabel("PFSP sampling probability")
    axis.set_title("M3 Opponent Distribution and Pool Growth", loc="left", fontweight="bold")
    pool_axis = axis.twinx()
    pool_axis.plot(
        steps,
        [snapshot.pool_size for snapshot in evidence.snapshots],
        color="#111827",
        marker="o",
        linewidth=2.0,
        label="pool size",
    )
    pool_axis.set_ylabel("Active historical policies")
    pool_axis.set_ylim(0, 12.5)
    axis.legend(frameon=False, loc="upper left", ncol=2, fontsize=8)
    pool_axis.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    fig.savefig(temporary)
    plt.close(fig)
    temporary.replace(output)
    return output


def _matrix_payload(evidence: CrossplayFigureEvidence) -> dict[str, object]:
    def cells(values: tuple[tuple[float, ...], ...]) -> dict[str, dict[str, float]]:
        return {
            left: {
                right: values[row][column]
                for column, right in enumerate(evidence.policy_ids)
            }
            for row, left in enumerate(evidence.policy_ids)
        }

    return {
        "schema_version": 1,
        "policy_ids": list(evidence.policy_ids),
        "executed_rows": evidence.executed_rows,
        "crossplay_csv_sha256": evidence.source_sha256,
        "win_rate": cells(evidence.win_rate),
        "draw_rate": cells(evidence.draw_rate),
        "objective_rate": cells(evidence.objective_rate),
        "score_difference": cells(evidence.score_difference),
    }


def render_evidence_bundle(
    *,
    crossplay_csv: Path,
    pool_history_path: Path,
    heatmap_output: Path,
    pool_output: Path,
    matrix_output: Path,
) -> dict[str, object]:
    crossplay = load_crossplay_evidence(crossplay_csv)
    history = load_pool_history(pool_history_path)
    render_crossplay_heatmap(crossplay, heatmap_output)
    render_pool_history(history, pool_output)
    payload = _matrix_payload(crossplay)
    payload["pool_history_sha256"] = history.source_sha256
    matrix_output = matrix_output.expanduser().resolve()
    matrix_output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(payload, matrix_output)
    return payload
