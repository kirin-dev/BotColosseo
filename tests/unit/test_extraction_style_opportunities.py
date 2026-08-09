from __future__ import annotations

from dataclasses import replace

import numpy as np

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_protocol import ExtractionEvent, ExtractionEventType
from botcolosseo.envs.extraction_types import (
    ExtractionActorObservation,
    ExtractionPrivilegedState,
)
from botcolosseo.training.extraction_style_opportunities import (
    AggressiveOpportunityConfig,
    AggressiveOpportunityLedger,
    DefensiveOpportunityConfig,
    DefensiveOpportunityLedger,
    ExplorerOpportunityConfig,
    ExplorerOpportunityLedger,
)


def observation(
    *, health: float = 100, ammo: float = 30, carried: int = 0, remaining: float = 70
) -> ExtractionActorObservation:
    return ExtractionActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=health,
        ammo=ammo,
        carried_value=carried,
        free_slots=3 if carried == 0 else 2,
        minimum_slot_value=0 if carried == 0 else 10,
        banked_value=0,
        extraction_open=True,
        extraction_progress=0,
        remaining_time=remaining,
        previous_action=0,
    )


def state(
    *,
    host_x: float = 0,
    host_y: float = 0,
    opponent_x: float = 300,
    opponent_y: float = 0,
    host_health: int = 100,
    opponent_health: int = 100,
    host_slots: tuple[int, int, int] = (0, 0, 0),
    world_loot_mask: int = 127,
) -> ExtractionPrivilegedState:
    return ExtractionPrivilegedState(
        host_x=host_x,
        host_y=host_y,
        host_angle=0,
        opponent_x=opponent_x,
        opponent_y=opponent_y,
        opponent_angle=180,
        host_health=host_health,
        opponent_health=opponent_health,
        host_slots=host_slots,
        opponent_slots=(0, 0, 0),
        host_banked=0,
        opponent_banked=0,
        cache_owner=0,
        cache_slots=(0, 0, 0),
        cache_x=0,
        cache_y=0,
        world_loot_mask=world_loot_mask,
        round_state=1,
        winner=0,
        engine_tic=100,
    )


def event(kind: ExtractionEventType, *, side: str = "host") -> ExtractionEvent:
    return ExtractionEvent(
        type=kind,
        side=side,
        count=1,
        value=25,
        episode_id=0,
        decision_index=1,
        engine_tic=101,
    )


def test_aggressive_opportunity_requires_favorable_fight_and_rewards_full_chain() -> None:
    ledger = AggressiveOpportunityLedger(
        AggressiveOpportunityConfig(), learner_side="host", scale=1
    )
    assert not ledger.opportunity(
        observation_before=observation(health=30), state_before=state()
    ).active

    opportunity = ledger.opportunity(
        observation_before=observation(), state_before=state()
    )
    assert opportunity.active
    preferred = next(iter(opportunity.preferred_actions))
    initiation = ledger.apply(
        preferred,
        (),
        observation_before=observation(),
        state_before=state(),
        state_after=state(),
    )
    hit = ledger.apply(
        preferred,
        (event(ExtractionEventType.VALID_HIT),),
        observation_before=observation(),
        state_before=state(),
        state_after=state(),
    )
    kill = ledger.apply(
        preferred,
        (event(ExtractionEventType.DEATH, side="opponent"),),
        observation_before=observation(),
        state_before=state(),
        state_after=state(opponent_health=0),
    )
    cache = ledger.apply(
        MacroAction.MOVE_FORWARD,
        (event(ExtractionEventType.CACHE_LOOTED),),
        observation_before=observation(carried=25),
        state_before=state(opponent_health=0),
        state_after=state(opponent_health=0, host_slots=(25, 0, 0)),
    )
    extracted = ledger.apply(
        MacroAction.IDLE,
        (event(ExtractionEventType.EXTRACTED),),
        observation_before=observation(carried=25),
        state_before=state(opponent_health=0, host_slots=(25, 0, 0)),
        state_after=state(opponent_health=0),
    )

    assert initiation.components["pbrs_progress"] > 0
    assert hit.components["pbrs_progress"] > 0
    assert kill.components["pbrs_progress"] > 0
    assert cache.components["pbrs_progress"] > 0
    assert extracted.components["completion_utility"] > 0
    assert extracted.total > 0


def test_defensive_opportunity_is_risk_conditioned_and_reaches_value_conversion() -> None:
    ledger = DefensiveOpportunityLedger(
        DefensiveOpportunityConfig(), learner_side="host", scale=1
    )
    assert not ledger.opportunity(
        observation_before=observation(), state_before=state()
    ).active
    assert not ledger.opportunity(
        observation_before=observation(health=30),
        state_before=state(host_health=30),
    ).active
    opportunity = ledger.opportunity(
        observation_before=observation(carried=25),
        state_before=state(host_slots=(25, 0, 0)),
    )
    assert opportunity.active
    assert not (opportunity.preferred_actions & {MacroAction.ATTACK})

    retreat = ledger.apply(
        next(iter(opportunity.preferred_actions)),
        (),
        observation_before=observation(carried=25),
        state_before=state(host_slots=(25, 0, 0)),
        state_after=state(opponent_x=700, host_slots=(25, 0, 0)),
    )
    started = ledger.apply(
        MacroAction.IDLE,
        (event(ExtractionEventType.EXTRACTION_STARTED),),
        observation_before=observation(carried=25),
        state_before=state(opponent_x=700, host_slots=(25, 0, 0)),
        state_after=state(opponent_x=700, host_slots=(25, 0, 0)),
    )
    extracted = ledger.apply(
        MacroAction.IDLE,
        (event(ExtractionEventType.EXTRACTED),),
        observation_before=observation(carried=25),
        state_before=state(opponent_x=700, host_slots=(25, 0, 0)),
        state_after=state(opponent_x=700),
    )

    assert retreat.components["pbrs_progress"] > 0
    assert started.components["pbrs_progress"] > 0
    assert extracted.components["completion_utility"] > 0

    unconditioned = DefensiveOpportunityLedger(
        DefensiveOpportunityConfig(), learner_side="host", scale=1
    ).apply(
        MacroAction.IDLE,
        (event(ExtractionEventType.EXTRACTED),),
        observation_before=observation(carried=25),
        state_before=state(opponent_x=900, host_slots=(25, 0, 0)),
        state_after=state(opponent_x=900),
    )
    assert "completion_utility" not in unconditioned.components


def test_defensive_completion_requires_actual_disengagement() -> None:
    ledger = DefensiveOpportunityLedger(
        DefensiveOpportunityConfig(), learner_side="host", scale=1
    )
    opportunity = ledger.opportunity(
        observation_before=observation(carried=25),
        state_before=state(host_slots=(25, 0, 0)),
    )
    ledger.apply(
        next(iter(opportunity.preferred_actions)),
        (),
        observation_before=observation(carried=25),
        state_before=state(host_slots=(25, 0, 0)),
        state_after=state(host_slots=(25, 0, 0)),
    )
    ledger.apply(
        MacroAction.IDLE,
        (event(ExtractionEventType.EXTRACTION_STARTED),),
        observation_before=observation(carried=25),
        state_before=state(host_slots=(25, 0, 0)),
        state_after=state(host_slots=(25, 0, 0)),
    )
    extracted = ledger.apply(
        MacroAction.IDLE,
        (event(ExtractionEventType.EXTRACTED),),
        observation_before=observation(carried=25),
        state_before=state(host_slots=(25, 0, 0)),
        state_after=state(host_slots=(25, 0, 0)),
    )

    assert "completion_utility" not in extracted.components


def test_defensive_opportunity_steers_away_until_safe_then_converts() -> None:
    ledger = DefensiveOpportunityLedger(
        DefensiveOpportunityConfig(), learner_side="host", scale=1
    )
    initial = ledger.opportunity(
        observation_before=observation(carried=25),
        state_before=state(host_slots=(25, 0, 0)),
    )

    assert initial.preferred_actions == frozenset({MacroAction.TURN_RIGHT})

    ledger.apply(
        MacroAction.TURN_RIGHT,
        (),
        observation_before=observation(carried=25),
        state_before=state(host_slots=(25, 0, 0)),
        state_after=state(opponent_x=550, host_slots=(25, 0, 0)),
    )
    continuing = ledger.opportunity(
        observation_before=observation(carried=25),
        state_before=state(opponent_x=550, host_slots=(25, 0, 0)),
    )

    assert continuing.active
    assert continuing.phase == "disengage_from_risk"

    ledger.apply(
        next(iter(continuing.preferred_actions)),
        (),
        observation_before=observation(carried=25),
        state_before=state(opponent_x=550, host_slots=(25, 0, 0)),
        state_after=state(opponent_x=700, host_slots=(25, 0, 0)),
    )
    conversion = ledger.opportunity(
        observation_before=observation(carried=25),
        state_before=state(opponent_x=700, host_slots=(25, 0, 0)),
    )

    assert conversion.active
    assert conversion.phase == "convert_safety"


def test_explorer_opportunity_targets_available_loot_then_stops_to_extract() -> None:
    ledger = ExplorerOpportunityLedger(
        ExplorerOpportunityConfig(),
        learner_side="host",
        scale=1,
        layout_variant=0,
    )
    first = ledger.opportunity(
        observation_before=observation(), state_before=state()
    )
    assert first.active and first.phase == "novel_loot_route"

    pickup_one = ledger.apply(
        next(iter(first.preferred_actions)),
        (event(ExtractionEventType.LOOT_PICKUP),),
        observation_before=observation(),
        state_before=state(),
        state_after=state(host_x=-192, host_y=96, world_loot_mask=63),
    )
    pickup_two = ledger.apply(
        MacroAction.MOVE_FORWARD,
        (event(ExtractionEventType.LOOT_PICKUP),),
        observation_before=observation(carried=25),
        state_before=state(host_x=-192, host_y=96, world_loot_mask=63),
        state_after=state(host_x=384, host_y=-224, world_loot_mask=31),
    )
    upgrade = ledger.apply(
        MacroAction.MOVE_FORWARD,
        (event(ExtractionEventType.LOOT_DROP),),
        observation_before=observation(carried=60),
        state_before=state(host_slots=(10, 25, 25)),
        state_after=state(host_slots=(10, 25, 50)),
    )
    convert = ledger.opportunity(
        observation_before=observation(carried=85),
        state_before=state(host_slots=(10, 25, 50)),
    )
    extracted = ledger.apply(
        next(iter(convert.preferred_actions)),
        (event(ExtractionEventType.EXTRACTED),),
        observation_before=observation(carried=85),
        state_before=state(host_slots=(10, 25, 50)),
        state_after=state(),
    )

    assert pickup_one.components["pbrs_progress"] > 0
    assert pickup_two.total <= 0
    assert upgrade.components["pbrs_progress"] > 0
    assert convert.phase == "convert_exploration"
    assert extracted.components["completion_utility"] > 0


def test_explorer_completion_requires_real_backpack_upgrade() -> None:
    ledger = ExplorerOpportunityLedger(
        ExplorerOpportunityConfig(),
        learner_side="host",
        scale=1,
        layout_variant=0,
    )
    ledger.apply(
        MacroAction.MOVE_FORWARD,
        (event(ExtractionEventType.LOOT_PICKUP),),
        observation_before=observation(),
        state_before=state(),
        state_after=state(host_x=-192, host_y=96, world_loot_mask=63),
    )
    ledger.apply(
        MacroAction.MOVE_FORWARD,
        (event(ExtractionEventType.LOOT_PICKUP),),
        observation_before=observation(carried=25),
        state_before=state(host_x=-192, host_y=96, world_loot_mask=63),
        state_after=state(host_x=384, host_y=-224, world_loot_mask=31),
    )
    extracted = ledger.apply(
        MacroAction.IDLE,
        (event(ExtractionEventType.EXTRACTED),),
        observation_before=observation(carried=50),
        state_before=state(host_slots=(25, 25, 0)),
        state_after=state(),
    )

    assert "completion_utility" not in extracted.components


def test_terminal_potential_is_cleared_even_when_chain_fails() -> None:
    config = replace(AggressiveOpportunityConfig(), completion_utility=0)
    ledger = AggressiveOpportunityLedger(config, learner_side="host", scale=1)
    opportunity = ledger.opportunity(
        observation_before=observation(), state_before=state()
    )
    ledger.apply(
        next(iter(opportunity.preferred_actions)),
        (),
        observation_before=observation(),
        state_before=state(),
        state_after=state(),
    )
    terminal = ledger.apply(
        MacroAction.IDLE,
        (event(ExtractionEventType.DEATH),),
        observation_before=observation(),
        state_before=state(),
        state_after=state(host_health=0),
    )

    assert terminal.components["pbrs_progress"] < 0
