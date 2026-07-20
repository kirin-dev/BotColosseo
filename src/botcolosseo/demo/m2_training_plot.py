from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class M2TrainingEvidence:
    bc_updates: tuple[int, ...]
    bc_losses: tuple[float, ...]
    bc_accuracies: tuple[float, ...]
    bc_selected_update: int
    ppo_steps: tuple[int, ...]
    ppo_objective_rates: tuple[float, ...]
    ppo_win_rates: tuple[float, ...]
    ppo_selected_steps: int


def _load_validation_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("official_test_result") is not False or payload.get(
        "test_cases_accessed"
    ) is not False:
        raise ValueError(f"Training plot requires validation-only evidence: {path}")
    return payload


def load_training_evidence(bc_path: Path, ppo_path: Path) -> M2TrainingEvidence:
    bc = _load_validation_summary(bc_path)
    ppo = _load_validation_summary(ppo_path)
    bc_selection = bc["selection"]
    ppo_selection = ppo["selection"]
    if not isinstance(bc_selection, dict) or not isinstance(ppo_selection, dict):
        raise ValueError("Training summaries must contain selection objects")
    if ppo_selection.get("split") != "validation":
        raise ValueError("PPO curve must come from the validation split")

    bc_curve = bc_selection["validation_curve"]
    ppo_curve = ppo_selection["candidates"]
    if not isinstance(bc_curve, list) or not bc_curve:
        raise ValueError("BC validation curve is empty")
    if not isinstance(ppo_curve, list) or not ppo_curve:
        raise ValueError("PPO validation candidates are empty")
    ppo_selected = ppo["selected"]
    if not isinstance(ppo_selected, dict):
        raise ValueError("PPO summary must contain a selected candidate")

    evidence = M2TrainingEvidence(
        bc_updates=tuple(int(row["update"]) for row in bc_curve),
        bc_losses=tuple(float(row["loss"]) for row in bc_curve),
        bc_accuracies=tuple(float(row["accuracy"]) for row in bc_curve),
        bc_selected_update=int(bc_selection["selected_update"]),
        ppo_steps=tuple(int(row["environment_steps"]) for row in ppo_curve),
        ppo_objective_rates=tuple(float(row["objective_rate"]) for row in ppo_curve),
        ppo_win_rates=tuple(float(row["win_rate"]) for row in ppo_curve),
        ppo_selected_steps=int(ppo_selected["environment_steps"]),
    )
    if evidence.bc_selected_update not in evidence.bc_updates:
        raise ValueError("Selected BC update is absent from its validation curve")
    if evidence.ppo_selected_steps not in evidence.ppo_steps:
        raise ValueError("Selected PPO step is absent from its validation candidates")
    return evidence


def render_training_plot(evidence: M2TrainingEvidence, output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue = "#2563EB"
    orange = "#EA580C"
    green = "#059669"
    muted = "#64748B"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=160)
    fig.patch.set_facecolor("#F8FAFC")
    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    bc_axis = axes[0]
    accuracy_axis = bc_axis.twinx()
    bc_axis.plot(evidence.bc_updates, evidence.bc_losses, color=blue, linewidth=2.2)
    accuracy_axis.plot(
        evidence.bc_updates,
        evidence.bc_accuracies,
        color=green,
        linewidth=2.0,
        alpha=0.9,
    )
    bc_axis.axvline(
        evidence.bc_selected_update,
        color=orange,
        linestyle="--",
        linewidth=1.6,
        label=f"selected: {evidence.bc_selected_update:,}",
    )
    bc_axis.set_title("Behavioral Cloning", loc="left", fontweight="bold")
    bc_axis.set_xlabel("Optimizer updates")
    bc_axis.set_ylabel("Validation cross-entropy", color=blue)
    accuracy_axis.set_ylabel("Validation accuracy", color=green)
    accuracy_axis.set_ylim(0.8, 1.0)
    bc_axis.legend(frameon=False, loc="upper right")

    ppo_axis = axes[1]
    step_millions = tuple(step / 1_000_000 for step in evidence.ppo_steps)
    ppo_axis.plot(
        step_millions,
        evidence.ppo_objective_rates,
        marker="o",
        color=green,
        linewidth=2.2,
        label="objective rate",
    )
    ppo_axis.plot(
        step_millions,
        evidence.ppo_win_rates,
        marker="o",
        color=blue,
        linewidth=2.2,
        label="win rate",
    )
    selected_x = evidence.ppo_selected_steps / 1_000_000
    ppo_axis.axvline(
        selected_x,
        color=orange,
        linestyle="--",
        linewidth=1.6,
        label=f"selected: {selected_x:.1f}M",
    )
    ppo_axis.set_title("PPO Checkpoint Selection", loc="left", fontweight="bold")
    ppo_axis.set_xlabel("Environment steps (millions)")
    ppo_axis.set_ylabel("Validation rate (30 games / checkpoint)")
    ppo_axis.set_ylim(0.65, 1.02)
    ppo_axis.legend(frameon=False, loc="lower right")

    fig.suptitle("M2 Training and Validation Evidence", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "Validation-only checkpoint selection — not official test performance",
        ha="center",
        color=muted,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93), w_pad=3.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    fig.savefig(temporary, bbox_inches="tight")
    plt.close(fig)
    temporary.replace(output)
    return output
