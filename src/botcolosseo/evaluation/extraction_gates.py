from __future__ import annotations

import math
import random
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
    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator > 0 else 0.0

    if style == "aggressive":
        return (
            2.0 * episode.aggressive_chains
            + 0.5
            * ratio(episode.kill_to_cache_conversions, episode.kills)
            + 0.25
            * ratio(
                episode.favorable_encounter_initiations,
                episode.encounter_opportunities,
            )
            + 0.25 * ratio(episode.valid_hits, episode.attack_decisions)
        )
    if style == "defensive":
        return (
            ratio(
                episode.successful_disengagements,
                episode.disengagement_opportunities,
            )
            + float(episode.meaningful_extractions)
            - 0.25 * ratio(
                episode.combat_with_meaningful_value,
                episode.attack_decisions,
            )
            - 0.5 * episode.timeout_with_value
        )
    if style == "explorer":
        return (
            min(episode.meaningful_loot_regions, 7) / 7
            + episode.upgrade_to_extraction_conversions
            + 0.25
            * ratio(episode.urgency_extractions, episode.urgency_opportunities)
            - 0.5 * episode.timeout_with_value
        )
    raise ValueError("Unknown Extraction style")


def aggressive_showcase_direction_counts(
    strong: tuple[ExtractionEpisodeMetrics, ...],
    styled: tuple[ExtractionEpisodeMetrics, ...],
) -> dict[str, int]:
    """Count paired directional changes used only by Showcase admission."""
    if len(strong) != 240 or len(styled) != 240:
        raise ValueError("Aggressive Showcase requires paired 240-episode budgets")
    strong_by_case = {
        (item.seed, item.learner_side, item.opponent_style): item for item in strong
    }
    styled_by_case = {
        (item.seed, item.learner_side, item.opponent_style): item for item in styled
    }
    if (
        len(strong_by_case) != len(strong)
        or len(styled_by_case) != len(styled)
        or set(strong_by_case) != set(styled_by_case)
    ):
        raise ValueError("Aggressive Showcase evidence is not uniquely paired")

    positive_pairs = 0
    negative_pairs = 0
    unchanged_pairs = 0
    new_complete_chains = 0
    lost_complete_chains = 0
    for key in sorted(strong_by_case):
        strong_episode = strong_by_case[key]
        styled_episode = styled_by_case[key]
        difference = _style_score("aggressive", styled_episode) - _style_score(
            "aggressive", strong_episode
        )
        if difference > 0:
            positive_pairs += 1
        elif difference < 0:
            negative_pairs += 1
        else:
            unchanged_pairs += 1
        if styled_episode.aggressive_chains > 0:
            if strong_episode.aggressive_chains == 0:
                new_complete_chains += 1
        elif strong_episode.aggressive_chains > 0:
            lost_complete_chains += 1

    return {
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "unchanged_pairs": unchanged_pairs,
        "new_complete_chains": new_complete_chains,
        "lost_complete_chains": lost_complete_chains,
    }


def _paired_bootstrap_interval(
    values: tuple[float, ...],
    *,
    samples: int = 10_000,
    seed: int = 20260726,
) -> tuple[float, float, float]:
    if len(values) < 2:
        raise ValueError("Paired style CI requires at least two cases")
    if samples < 1_000:
        raise ValueError("Paired style bootstrap requires at least 1,000 samples")
    mean = sum(values) / len(values)
    generator = random.Random(seed)
    bootstrapped = sorted(
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    lower = bootstrapped[math.floor(0.025 * (samples - 1))]
    upper = bootstrapped[math.ceil(0.975 * (samples - 1))]
    return mean, lower, upper


def _mean(items: tuple[ExtractionEpisodeMetrics, ...], attribute: str) -> float:
    return sum(float(getattr(item, attribute)) for item in items) / len(items)


def _ratio_total(
    items: tuple[ExtractionEpisodeMetrics, ...],
    numerator: str,
    denominator: str,
) -> float:
    denominator_total = sum(int(getattr(item, denominator)) for item in items)
    if denominator_total == 0:
        return 0.0
    return (
        sum(int(getattr(item, numerator)) for item in items) / denominator_total
    )


def _opponent_retention_margin(
    strong: tuple[ExtractionEpisodeMetrics, ...],
    styled: tuple[ExtractionEpisodeMetrics, ...],
) -> float:
    opponents = sorted({item.opponent_style for item in strong})
    return min(
        _rate(
            tuple(item for item in styled if item.opponent_style == opponent),
            "won",
        )
        - max(
            0.40,
            _rate(
                tuple(item for item in strong if item.opponent_style == opponent),
                "won",
            )
            - 0.20,
        )
        for opponent in opponents
    )


def _reward_hacking_checks(
    style: str,
    *,
    strong: tuple[ExtractionEpisodeMetrics, ...],
    styled: tuple[ExtractionEpisodeMetrics, ...],
) -> tuple[GateCheck, ...]:
    if style == "aggressive":
        chain_delta = _mean(styled, "aggressive_chains") - _mean(
            strong, "aggressive_chains"
        )
        precision_delta = _ratio_total(
            styled, "valid_hits", "attack_decisions"
        ) - _ratio_total(strong, "valid_hits", "attack_decisions")
        return (
            GateCheck(
                "anti_hack_complete_combat_chain",
                chain_delta > 0,
                chain_delta,
                ">0",
            ),
            GateCheck(
                "anti_hack_no_empty_fire_regression",
                precision_delta >= -0.05,
                precision_delta,
                ">=-0.05",
            ),
        )
    if style == "defensive":
        meaningful_ratio = (
            _mean(styled, "meaningful_extractions")
            / _mean(strong, "meaningful_extractions")
            if _mean(strong, "meaningful_extractions") > 0
            else 0.0
        )
        timeout_delta = _mean(styled, "timeout_with_value") - _mean(
            strong, "timeout_with_value"
        )
        return (
            GateCheck(
                "anti_hack_not_inactive",
                meaningful_ratio >= 0.85,
                meaningful_ratio,
                ">=0.85",
            ),
            GateCheck(
                "anti_hack_no_timeout_value_loss",
                timeout_delta <= 0.02,
                timeout_delta,
                "<=0.02",
            ),
        )
    if style == "explorer":
        conversion_delta = _mean(
            styled, "upgrade_to_extraction_conversions"
        ) - _mean(strong, "upgrade_to_extraction_conversions")
        timeout_delta = _mean(styled, "timeout_with_value") - _mean(
            strong, "timeout_with_value"
        )
        return (
            GateCheck(
                "anti_hack_real_upgrade_conversion",
                conversion_delta > 0,
                conversion_delta,
                ">0",
            ),
            GateCheck(
                "anti_hack_no_high_value_wandering",
                timeout_delta <= 0.02,
                timeout_delta,
                "<=0.02",
            ),
        )
    raise ValueError("Unknown Extraction style")


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
    style_difference, ci_lower, ci_upper = _paired_bootstrap_interval(differences)
    integrity_errors = sum(
        item.max_peer_tic_lag > 2 or item.truncated for item in styled
    )
    opponent_margin = _opponent_retention_margin(strong, styled)
    hacking_checks = _reward_hacking_checks(
        style,
        strong=strong,
        styled=styled,
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
        GateCheck(
            "worst_opponent_retention",
            opponent_margin >= 0,
            opponent_margin,
            ">=0.00",
        ),
        GateCheck("style_paired_difference", style_difference > 0, style_difference, ">0"),
        GateCheck("style_ci_lower", ci_lower > 0, ci_lower, ">0"),
        GateCheck("style_ci_upper", ci_upper > 0, ci_upper, ">0"),
        *hacking_checks,
        GateCheck(
            "reward_hacking_counterexamples",
            all(check.passed for check in hacking_checks),
            float(sum(not check.passed for check in hacking_checks)),
            "==0",
        ),
        GateCheck("protocol_integrity", integrity_errors == 0, float(integrity_errors), "==0"),
    )
    return GateResult(all(item.passed for item in checks), checks)


def style_heldout_gate(
    *,
    strong: tuple[ExtractionEpisodeMetrics, ...],
    styled: tuple[ExtractionEpisodeMetrics, ...],
) -> GateResult:
    if len(strong) != 120 or len(styled) != 120:
        raise ValueError("Style heldout gate requires frozen 120-episode budgets")
    strong_by_case = {
        (item.seed, item.learner_side, item.opponent_style): item for item in strong
    }
    styled_by_case = {
        (item.seed, item.learner_side, item.opponent_style): item for item in styled
    }
    if set(strong_by_case) != set(styled_by_case):
        raise ValueError("Style and Strong heldout cases are not paired")
    extraction_delta = _rate(styled, "extracted") - _rate(strong, "extracted")
    opponent_margin = _opponent_retention_margin(strong, styled)
    integrity_errors = sum(
        item.max_peer_tic_lag > 2 or item.truncated for item in styled
    )
    checks = (
        GateCheck(
            "heldout_extraction_delta",
            extraction_delta >= -0.10,
            extraction_delta,
            ">=-0.10",
        ),
        GateCheck(
            "heldout_worst_opponent_retention",
            opponent_margin >= 0,
            opponent_margin,
            ">=0.00",
        ),
        GateCheck(
            "heldout_protocol_integrity",
            integrity_errors == 0,
            float(integrity_errors),
            "==0",
        ),
    )
    return GateResult(all(item.passed for item in checks), checks)
