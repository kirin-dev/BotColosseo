from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from botcolosseo.agents.extraction_model import load_extraction_policy
from botcolosseo.agents.extraction_teachers import StyledExtractionTeacher
from botcolosseo.data.extraction_demonstrations import ExtractionCase
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_protocol import ExtractionEventType
from botcolosseo.envs.ipc import WorkerTimeout
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.extraction_bc import extraction_observation_tensors

ATTACK_ACTIONS = frozenset(
    (
        MacroAction.ATTACK,
        MacroAction.FORWARD_ATTACK,
        MacroAction.TURN_LEFT_ATTACK,
        MacroAction.TURN_RIGHT_ATTACK,
    )
)


@dataclass(frozen=True)
class ExtractionEpisodeMetrics:
    seed: int
    learner_side: str
    opponent_style: str
    decisions: int
    extracted_value: int
    extracted: bool
    died: bool
    valid_hits: int
    kills: int
    cache_looted: int
    attack_decisions: int
    unique_route_cells: int
    terminated: bool
    truncated: bool
    loot_pickups: int = 0
    won: bool = False
    opponent_extracted: bool = False
    opponent_extracted_value: int = 0
    extracted_value_advantage: int = 0
    max_peer_tic_lag: int = 0
    environment_attempts: int = 1
    encounter_opportunities: int = 0
    favorable_encounter_initiations: int = 0
    kill_to_cache_conversions: int = 0
    cache_to_extraction_conversions: int = 0
    aggressive_chains: int = 0
    disengagement_opportunities: int = 0
    successful_disengagements: int = 0
    meaningful_extractions: int = 0
    combat_with_meaningful_value: int = 0
    meaningful_loot_regions: int = 0
    backpack_upgrades: int = 0
    upgrade_to_extraction_conversions: int = 0
    urgency_opportunities: int = 0
    urgency_extractions: int = 0
    timeout_with_value: int = 0


def is_aggressive_showcase_chain(episode: ExtractionEpisodeMetrics) -> bool:
    """Return whether one replay proves the complete Aggressive product story."""
    return (
        episode.aggressive_chains >= 1
        and episode.extracted
        and episode.extracted_value > 0
    )


def _learner_event_count(
    events: Counter[tuple[str, ExtractionEventType]],
    *,
    side: str,
    event_type: ExtractionEventType,
) -> int:
    return events[(side, event_type)]


@torch.no_grad()
def evaluate_extraction_episode(
    *,
    root: Path,
    checkpoint: Path,
    style: str,
    case: ExtractionCase,
    device: torch.device,
    max_decisions: int = 700,
    policy_model: torch.nn.Module | None = None,
) -> ExtractionEpisodeMetrics:
    scenario_hash = json.loads(
        (
            root / "assets/scenarios/crystal_run_extraction/manifest.json"
        ).read_text(encoding="utf-8")
    )["wad_sha256"]
    if policy_model is None:
        model, _ = load_extraction_policy(
            checkpoint,
            style=style,
            scenario_hash=scenario_hash,
            device=device,
        )
    else:
        model = policy_model
    env = SynchronousExtractionEnv(
        config_path=(
            root
            / "assets/scenarios/crystal_run_extraction/crystal_run_extraction.cfg"
            if case.layout_id == "base"
            else root
            / "assets/scenarios/crystal_run_extraction/"
            "crystal_run_extraction_heldout.cfg"
        ),
        seed=case.seed,
        max_decisions=max_decisions,
    )
    opponent_side = "opponent" if case.learner_side == "host" else "host"
    opponent = (
        None
        if case.opponent_style == "idle"
        else StyledExtractionTeacher(
            side=opponent_side,
            style=case.opponent_style,
        )
    )
    events: Counter[tuple[str, ExtractionEventType]] = Counter()
    route_cells: set[tuple[int, int]] = set()
    loot_region_cells: set[tuple[int, int]] = set()
    attack_decisions = 0
    encounter_opportunities = 0
    favorable_encounter_initiations = 0
    encounter_active = False
    encounter_initiated = False
    disengagement_opportunities = 0
    successful_disengagements = 0
    disengagement_active = False
    killed_opponent = False
    looted_cache_after_kill = False
    kill_to_cache_conversions = 0
    cache_to_extraction_conversions = 0
    aggressive_chains = 0
    meaningful_extractions = 0
    combat_with_meaningful_value = 0
    backpack_upgrades = 0
    upgraded_backpack = False
    upgrade_to_extraction_conversions = 0
    urgency_opportunities = 0
    urgency_active = False
    urgency_extractions = 0
    timeout_with_value = 0
    hidden = model.initial_state(1, device=device)
    try:
        observations, _ = env.reset()
        if opponent is not None:
            opponent.reset()
        episode_start = True
        terminated = False
        truncated = False
        decisions = 0
        final_winner = 0
        max_peer_tic_lag = 0
        for _ in range(max_decisions):
            decisions += 1
            observation = (
                observations.host
                if case.learner_side == "host"
                else observations.opponent
            )
            output = model(
                *extraction_observation_tensors(
                    observation,
                    episode_start=episode_start,
                    device=device,
                ),
                hidden,
            )
            learner_action = MacroAction(int(output.logits[0, 0].argmax()))
            hidden = output.hidden
            if learner_action in ATTACK_ACTIONS:
                attack_decisions += 1
            state = env.privileged_state()
            x, y, opponent_x, opponent_y = (
                (state.host_x, state.host_y)
                if case.learner_side == "host"
                else (state.opponent_x, state.opponent_y)
            ) + (
                (state.opponent_x, state.opponent_y)
                if case.learner_side == "host"
                else (state.host_x, state.host_y)
            )
            route_cells.add((math.floor(x / 160.0), math.floor(y / 160.0)))
            opponent_distance = math.dist((x, y), (opponent_x, opponent_y))
            favorable_resources = observation.health >= 40 and observation.ammo > 5
            if opponent_distance <= 384 and favorable_resources and not encounter_active:
                encounter_opportunities += 1
                encounter_active = True
                encounter_initiated = False
            if opponent_distance >= 512:
                encounter_active = False
                encounter_initiated = False
            low_resources = observation.health <= 40 or observation.ammo <= 5
            if low_resources and opponent_distance <= 384 and not disengagement_active:
                disengagement_opportunities += 1
                disengagement_active = True
            if disengagement_active and opponent_distance >= 512:
                successful_disengagements += 1
                disengagement_active = False
            if (
                observation.carried_value >= 50
                and observation.remaining_time <= 20
                and not urgency_active
            ):
                urgency_opportunities += 1
                urgency_active = True
            if (
                learner_action in ATTACK_ACTIONS
                and observation.carried_value >= 50
            ):
                combat_with_meaningful_value += 1
            opponent_action = (
                MacroAction.IDLE
                if opponent is None
                else opponent.act(state)
            )
            host_action, away_action = (
                (learner_action, opponent_action)
                if case.learner_side == "host"
                else (opponent_action, learner_action)
            )
            step = env.step(host_action, away_action)
            final_winner = step.winner
            max_peer_tic_lag = max(max_peer_tic_lag, step.peer_tic_lag)
            observations = type(observations)(step.host, step.opponent)
            for event in step.events:
                events[(event.side, event.type)] += event.count
            learner_events = tuple(
                event for event in step.events if event.side == case.learner_side
            )
            opponent_events = tuple(
                event for event in step.events if event.side == opponent_side
            )
            valid_hit = any(
                event.type is ExtractionEventType.VALID_HIT
                for event in learner_events
            )
            if encounter_active and not encounter_initiated and valid_hit:
                favorable_encounter_initiations += 1
                encounter_initiated = True
            if any(
                event.type is ExtractionEventType.DEATH
                for event in opponent_events
            ):
                killed_opponent = True
            if any(
                event.type is ExtractionEventType.LOOT_PICKUP
                for event in learner_events
            ):
                loot_region_cells.add(
                    (math.floor(x / 160.0), math.floor(y / 160.0))
                )
            if any(
                event.type is ExtractionEventType.LOOT_DROP
                for event in learner_events
            ):
                backpack_upgrades += 1
                upgraded_backpack = True
            cache_looted = any(
                event.type is ExtractionEventType.CACHE_LOOTED
                for event in learner_events
            )
            if cache_looted and killed_opponent and not looted_cache_after_kill:
                kill_to_cache_conversions += 1
                looted_cache_after_kill = True
            extracted_now = any(
                event.type is ExtractionEventType.EXTRACTED
                for event in learner_events
            )
            if extracted_now:
                if observation.carried_value >= 25:
                    meaningful_extractions += 1
                if looted_cache_after_kill:
                    cache_to_extraction_conversions += 1
                    aggressive_chains += 1
                if upgraded_backpack:
                    upgrade_to_extraction_conversions += 1
                if urgency_active:
                    urgency_extractions += 1
            if any(
                event.type is ExtractionEventType.TIMEOUT for event in step.events
            ) and observation.carried_value > 0:
                timeout_with_value += 1
            episode_start = False
            terminated, truncated = step.terminated, step.truncated
            learner = step.host if case.learner_side == "host" else step.opponent
            life = env.protocol_snapshot().public_state(case.learner_side).life_state
            if life != 1 or terminated or truncated:
                break
        learner = (
            observations.host
            if case.learner_side == "host"
            else observations.opponent
        )
        life = env.protocol_snapshot().public_state(case.learner_side).life_state
        opponent_public = env.protocol_snapshot().public_state(opponent_side)
        won = final_winner == (1 if case.learner_side == "host" else 2)
        return ExtractionEpisodeMetrics(
            seed=case.seed,
            learner_side=case.learner_side,
            opponent_style=case.opponent_style,
            decisions=decisions,
            extracted_value=learner.banked_value,
            extracted=life == 3,
            died=life == 2,
            valid_hits=_learner_event_count(
                events,
                side=case.learner_side,
                event_type=ExtractionEventType.VALID_HIT,
            ),
            kills=_learner_event_count(
                events,
                side=opponent_side,
                event_type=ExtractionEventType.DEATH,
            ),
            cache_looted=_learner_event_count(
                events,
                side=case.learner_side,
                event_type=ExtractionEventType.CACHE_LOOTED,
            ),
            loot_pickups=_learner_event_count(
                events,
                side=case.learner_side,
                event_type=ExtractionEventType.LOOT_PICKUP,
            ),
            attack_decisions=attack_decisions,
            unique_route_cells=len(route_cells),
            terminated=terminated,
            truncated=truncated,
            won=won,
            opponent_extracted=opponent_public.life_state == 3,
            opponent_extracted_value=opponent_public.banked_value,
            extracted_value_advantage=(
                learner.banked_value - opponent_public.banked_value
            ),
            max_peer_tic_lag=max_peer_tic_lag,
            encounter_opportunities=encounter_opportunities,
            favorable_encounter_initiations=favorable_encounter_initiations,
            kill_to_cache_conversions=kill_to_cache_conversions,
            cache_to_extraction_conversions=cache_to_extraction_conversions,
            aggressive_chains=aggressive_chains,
            disengagement_opportunities=disengagement_opportunities,
            successful_disengagements=successful_disengagements,
            meaningful_extractions=meaningful_extractions,
            combat_with_meaningful_value=combat_with_meaningful_value,
            meaningful_loot_regions=len(loot_region_cells),
            backpack_upgrades=backpack_upgrades,
            upgrade_to_extraction_conversions=upgrade_to_extraction_conversions,
            urgency_opportunities=urgency_opportunities,
            urgency_extractions=urgency_extractions,
            timeout_with_value=timeout_with_value,
        )
    finally:
        env.close()


def evaluate_extraction_episode_with_retries(
    *,
    startup_attempts: int = 3,
    **kwargs: object,
) -> ExtractionEpisodeMetrics:
    """Retry only transient worker startup timeouts for one fixed case."""
    if startup_attempts <= 0:
        raise ValueError("Extraction startup attempts must be positive")
    for attempt in range(1, startup_attempts + 1):
        try:
            episode = evaluate_extraction_episode(**kwargs)
            return replace(episode, environment_attempts=attempt)
        except WorkerTimeout:
            if attempt == startup_attempts:
                raise
    raise AssertionError("Extraction evaluation retry loop did not return")


def summarize_extraction_episodes(
    episodes: tuple[ExtractionEpisodeMetrics, ...],
) -> dict[str, object]:
    if not episodes:
        raise ValueError("Extraction evaluation requires episodes")
    count = len(episodes)
    by_opponent: dict[str, dict[str, float | int]] = {}
    for opponent in sorted({item.opponent_style for item in episodes}):
        selected = tuple(
            item for item in episodes if item.opponent_style == opponent
        )
        by_opponent[opponent] = {
            "episodes": len(selected),
            "win_rate": sum(item.won for item in selected) / len(selected),
            "extraction_rate": sum(item.extracted for item in selected)
            / len(selected),
            "opponent_extraction_rate": sum(
                item.opponent_extracted for item in selected
            )
            / len(selected),
        }
    paired_units: dict[tuple[int, str], list[ExtractionEpisodeMetrics]] = {}
    for item in episodes:
        paired_units.setdefault((item.seed, item.opponent_style), []).append(item)

    def paired_interval(attribute: str) -> dict[str, float | int]:
        values = tuple(
            sum(float(getattr(item, attribute)) for item in unit) / len(unit)
            for unit in paired_units.values()
        )
        generator = random.Random(f"extraction-v3:{attribute}")
        samples = 10_000
        bootstrapped = sorted(
            sum(generator.choice(values) for _ in values) / len(values)
            for _ in range(samples)
        )
        return {
            "lower": bootstrapped[math.floor(0.025 * (samples - 1))],
            "mean": sum(values) / len(values),
            "samples": samples,
            "upper": bootstrapped[math.ceil(0.975 * (samples - 1))],
        }

    return {
        "attack_decisions_mean": sum(item.attack_decisions for item in episodes)
        / count,
        "cache_looted_total": sum(item.cache_looted for item in episodes),
        "death_rate": sum(item.died for item in episodes) / count,
        "episodes": [asdict(item) for item in episodes],
        "extraction_rate": sum(item.extracted for item in episodes) / count,
        "kill_total": sum(item.kills for item in episodes),
        "loot_pickups_total": sum(item.loot_pickups for item in episodes),
        "mean_extracted_value": sum(item.extracted_value for item in episodes)
        / count,
        "mean_extracted_value_advantage": sum(
            item.extracted_value_advantage for item in episodes
        )
        / count,
        "opponent_extraction_rate": sum(
            item.opponent_extracted for item in episodes
        )
        / count,
        "prevent_opponent_extraction_rate": 1
        - sum(item.opponent_extracted for item in episodes) / count,
        "protocol_inconsistencies": sum(
            item.max_peer_tic_lag > 2 for item in episodes
        ),
        "route_cells_mean": sum(item.unique_route_cells for item in episodes)
        / count,
        "style_metrics": {
            "aggressive_chain_rate": sum(item.aggressive_chains for item in episodes)
            / count,
            "extraction_denial_rate": 1
            - sum(item.opponent_extracted for item in episodes) / count,
            "backpack_upgrade_rate": sum(item.backpack_upgrades for item in episodes)
            / count,
            "cache_to_extraction_conversion_rate": sum(
                item.cache_to_extraction_conversions for item in episodes
            )
            / count,
            "combat_with_meaningful_value_mean": sum(
                item.combat_with_meaningful_value for item in episodes
            )
            / count,
            "favorable_encounter_initiation_rate": (
                sum(item.favorable_encounter_initiations for item in episodes)
                / sum(item.encounter_opportunities for item in episodes)
                if sum(item.encounter_opportunities for item in episodes)
                else 0.0
            ),
            "meaningful_extraction_rate": sum(
                item.meaningful_extractions for item in episodes
            )
            / count,
            "meaningful_loot_regions_mean": sum(
                item.meaningful_loot_regions for item in episodes
            )
            / count,
            "successful_disengagement_rate": (
                sum(item.successful_disengagements for item in episodes)
                / sum(item.disengagement_opportunities for item in episodes)
                if sum(item.disengagement_opportunities for item in episodes)
                else 0.0
            ),
            "timeout_with_value_rate": sum(
                item.timeout_with_value for item in episodes
            )
            / count,
            "upgrade_to_extraction_conversion_rate": sum(
                item.upgrade_to_extraction_conversions for item in episodes
            )
            / count,
            "urgency_extraction_rate": (
                sum(item.urgency_extractions for item in episodes)
                / sum(item.urgency_opportunities for item in episodes)
                if sum(item.urgency_opportunities for item in episodes)
                else 0.0
            ),
            "valid_hit_per_attack": (
                sum(item.valid_hits for item in episodes)
                / sum(item.attack_decisions for item in episodes)
                if sum(item.attack_decisions for item in episodes)
                else 0.0
            ),
        },
        "valid_hits_total": sum(item.valid_hits for item in episodes),
        "win_rate": sum(item.won for item in episodes) / count,
        "by_opponent": by_opponent,
        "paired_bootstrap_95": {
            "extracted_value": paired_interval("extracted_value"),
            "extracted_value_advantage": paired_interval(
                "extracted_value_advantage"
            ),
            "extraction_rate": paired_interval("extracted"),
            "win_rate": paired_interval("won"),
        },
    }
