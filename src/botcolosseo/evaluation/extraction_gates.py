from __future__ import annotations

import math
from dataclasses import dataclass

from botcolosseo.evaluation.extraction import ExtractionEpisodeMetrics


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    value: float
    threshold: str


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: tuple[GateCheck, ...]


def _rate(items: tuple[ExtractionEpisodeMetrics, ...], attribute: str) -> float:
    return sum(bool(getattr(item, attribute)) for item in items) / len(items)


def strong_validation_gate(
    validation: tuple[ExtractionEpisodeMetrics, ...],
    heldout: tuple[ExtractionEpisodeMetrics, ...],
    solo: tuple[ExtractionEpisodeMetrics, ...],
) -> GateResult:
    if len(validation) != 240 or len(heldout) != 120 or len(solo) != 40:
        raise ValueError("Strong gate requires the frozen validation budgets")
    opponents = sorted({item.opponent_style for item in validation})
    if opponents != ["aggressive", "defensive", "explorer", "strong"]:
        raise ValueError("Strong gate opponent coverage is incomplete")
    if (
        {item.opponent_style for item in solo} != {"idle"}
        or {item.learner_side for item in solo} != {"host", "opponent"}
        or len({(item.seed, item.learner_side) for item in solo}) != len(solo)
    ):
        raise ValueError("Strong gate solo evaluation is contaminated")
    win_rate = _rate(validation, "won")
    extraction_rate = _rate(validation, "extracted")
    solo_extraction_rate = _rate(solo, "extracted")
    heldout_extraction = _rate(heldout, "extracted")
    value_advantage = sum(
        item.extracted_value_advantage for item in validation
    ) / len(validation)
    worst_opponent_win = min(
        _rate(
            tuple(item for item in validation if item.opponent_style == opponent),
            "won",
        )
        for opponent in opponents
    )
    protocol_errors = sum(
        item.max_peer_tic_lag > 2 or item.truncated
        for item in (*validation, *heldout, *solo)
    )
    checks = (
        GateCheck(
            "solo_extraction",
            solo_extraction_rate >= 0.90,
            solo_extraction_rate,
            ">=0.90",
        ),
        GateCheck("script_average_win", win_rate >= 0.70, win_rate, ">=0.70"),
        GateCheck(
            "script_worst_case_win",
            worst_opponent_win >= 0.55,
            worst_opponent_win,
            ">=0.55",
        ),
        GateCheck("validation_extraction", extraction_rate >= 0.75, extraction_rate, ">=0.75"),
        GateCheck("heldout_extraction", heldout_extraction >= 0.70, heldout_extraction, ">=0.70"),
        GateCheck("extracted_value_advantage", value_advantage > 0, value_advantage, ">0"),
        GateCheck("protocol_integrity", protocol_errors == 0, float(protocol_errors), "==0"),
    )
    return GateResult(all(item.passed for item in checks), checks)


def _style_score(style: str, episode: ExtractionEpisodeMetrics) -> float:
    if style == "aggressive":
        return episode.valid_hits + 2 * episode.cache_looted
    if style == "defensive":
        return -float(episode.attack_decisions)
    if style == "explorer":
        return episode.unique_route_cells + 0.5 * episode.loot_pickups
    raise ValueError("Unknown Extraction style")


def _paired_lower_confidence_bound(values: tuple[float, ...]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("Paired style CI requires at least two cases")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    lower = mean - 1.96 * math.sqrt(variance / len(values))
    return mean, lower


def style_validation_gate(
    *,
    style: str,
    strong: tuple[ExtractionEpisodeMetrics, ...],
    styled: tuple[ExtractionEpisodeMetrics, ...],
) -> GateResult:
    if len(strong) != 240 or len(styled) != 240:
        raise ValueError("Style gate requires paired frozen validation budgets")
    strong_by_case = {
        (item.seed, item.learner_side, item.opponent_style): item for item in strong
    }
    styled_by_case = {
        (item.seed, item.learner_side, item.opponent_style): item for item in styled
    }
    if set(strong_by_case) != set(styled_by_case):
        raise ValueError("Style and Strong validation cases are not paired")
    ordered = tuple(sorted(strong_by_case))
    strong_successes = tuple(
        key
        for key in ordered
        if strong_by_case[key].won and strong_by_case[key].extracted
    )
    retention = (
        sum(
            styled_by_case[key].won and styled_by_case[key].extracted
            for key in strong_successes
        )
        / len(strong_successes)
        if strong_successes
        else 0.0
    )
    strong_extraction = _rate(strong, "extracted")
    style_extraction = _rate(styled, "extracted")
    strong_value = sum(item.extracted_value for item in strong) / len(strong)
    style_value = sum(item.extracted_value for item in styled) / len(styled)
    value_ratio = style_value / strong_value if strong_value > 0 else 0.0
    differences = tuple(
        _style_score(style, styled_by_case[key])
        - _style_score(style, strong_by_case[key])
        for key in ordered
    )
    style_difference, ci_lower = _paired_lower_confidence_bound(differences)
    integrity_errors = sum(
        item.max_peer_tic_lag > 2 or item.truncated for item in styled
    )
    checks = (
        GateCheck("paired_task_retention", retention >= 0.85, retention, ">=0.85"),
        GateCheck(
            "extraction_rate_delta",
            style_extraction - strong_extraction >= -0.10,
            style_extraction - strong_extraction,
            ">=-0.10",
        ),
        GateCheck("mean_value_ratio", value_ratio >= 0.85, value_ratio, ">=0.85"),
        GateCheck("style_paired_difference", style_difference > 0, style_difference, ">0"),
        GateCheck("style_ci_lower", ci_lower > 0, ci_lower, ">0"),
        GateCheck("protocol_integrity", integrity_errors == 0, float(integrity_errors), "==0"),
    )
    return GateResult(all(item.passed for item in checks), checks)
