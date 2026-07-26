from __future__ import annotations

import numpy as np

from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_protocol import ExtractionEvent, ExtractionEventType
from botcolosseo.envs.extraction_types import (
    ExtractionActorObservation,
    ExtractionPrivilegedState,
)
from botcolosseo.training.extraction_rewards import (
    AggressiveExtractionRewardConfig,
    AggressiveExtractionRewardLedger,
    DefensiveExtractionRewardConfig,
    DefensiveExtractionRewardLedger,
    ExplorerExtractionRewardConfig,
    ExplorerExtractionRewardLedger,
    ExtractionTaskRewardConfig,
    ExtractionTaskRewardLedger,
)


def observation(
    *,
    health: float = 100,
    ammo: float = 30,
    carried: int = 0,
) -> ExtractionActorObservation:
    return ExtractionActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=health,
        ammo=ammo,
        carried_value=carried,
        free_slots=3 if carried == 0 else 2,
        minimum_slot_value=0 if carried == 0 else 10,
        banked_value=0,
        extraction_open=False,
        extraction_progress=0,
        remaining_time=75,
        previous_action=0,
    )


def state(
    *,
    host_x: float = 0,
    opponent_x: float = 100,
    host_slots: tuple[int, int, int] = (0, 0, 0),
) -> ExtractionPrivilegedState:
    return ExtractionPrivilegedState(
        host_x=host_x,
        host_y=0,
        host_angle=0,
        opponent_x=opponent_x,
        opponent_y=0,
        opponent_angle=180,
        host_health=100,
        opponent_health=100,
        host_slots=host_slots,
        opponent_slots=(0, 0, 0),
        host_banked=0,
        opponent_banked=0,
        cache_owner=0,
        cache_slots=(0, 0, 0),
        cache_x=0,
        cache_y=0,
        world_loot_mask=0,
        round_state=1,
        winner=0,
        engine_tic=100,
    )


def event(
    kind: ExtractionEventType,
    *,
    side: str = "host",
    value: int = 0,
) -> ExtractionEvent:
    return ExtractionEvent(
        type=kind,
        side=side,
        count=1,
        value=value,
        episode_id=0,
        decision_index=1,
        engine_tic=101,
    )


def test_task_reward_does_not_turn_kills_into_terminal_value() -> None:
    ledger = ExtractionTaskRewardLedger(
        ExtractionTaskRewardConfig(),
        learner_side="host",
    )

    reward = ledger.apply(
        MacroAction.IDLE,
        (event(ExtractionEventType.DEATH, side="opponent"),),
        observation_before=observation(),
        state_before=state(),
        state_after=state(),
        scale=1,
    )

    assert reward.total == 0


def test_aggressive_reward_requires_valid_conversion_and_penalizes_spam() -> None:
    ledger = AggressiveExtractionRewardLedger(
        AggressiveExtractionRewardConfig(),
        learner_side="host",
        scale=1,
    )

    spam = ledger.apply(
        MacroAction.ATTACK,
        (),
        observation_before=observation(health=30, ammo=3),
        state_before=state(),
        state_after=state(),
    )
    hit = ledger.apply(
        MacroAction.FORWARD_ATTACK,
        (event(ExtractionEventType.VALID_HIT),),
        observation_before=observation(),
        state_before=state(),
        state_after=state(),
    )
    cache = ledger.apply(
        MacroAction.MOVE_FORWARD,
        (event(ExtractionEventType.CACHE_LOOTED, value=25),),
        observation_before=observation(carried=10),
        state_before=state(),
        state_after=state(host_slots=(10, 25, 0)),
    )
    extracted = ledger.apply(
        MacroAction.IDLE,
        (event(ExtractionEventType.EXTRACTED),),
        observation_before=observation(carried=35),
        state_before=state(host_slots=(10, 25, 0)),
        state_after=state(),
    )

    assert spam.total < 0
    assert hit.components["valid_hit"] > 0
    assert cache.components["cache_looted"] > 0
    assert extracted.components["cache_to_extraction"] > 0


def test_defensive_reward_rejects_empty_camping_and_rewards_value_protection() -> None:
    ledger = DefensiveExtractionRewardLedger(
        DefensiveExtractionRewardConfig(),
        learner_side="host",
        scale=1,
    )

    idle = ledger.apply(
        MacroAction.IDLE,
        (),
        observation_before=observation(),
        state_before=state(),
        state_after=state(),
    )
    extraction = ledger.apply(
        MacroAction.IDLE,
        (event(ExtractionEventType.EXTRACTED),),
        observation_before=observation(carried=50),
        state_before=state(host_slots=(50, 0, 0)),
        state_after=state(),
    )

    assert idle.components["empty_idle"] < 0
    assert extraction.components["meaningful_extraction"] > 0


def test_explorer_reward_needs_loot_regions_and_upgrade_not_raw_distance() -> None:
    ledger = ExplorerExtractionRewardLedger(
        ExplorerExtractionRewardConfig(),
        learner_side="host",
        scale=1,
    )

    movement = ledger.apply(
        MacroAction.MOVE_FORWARD,
        (),
        observation_before=observation(),
        state_before=state(host_x=0),
        state_after=state(host_x=500),
    )
    pickup = ledger.apply(
        MacroAction.MOVE_FORWARD,
        (event(ExtractionEventType.LOOT_PICKUP, value=50),),
        observation_before=observation(carried=30),
        state_before=state(host_x=500),
        state_after=state(host_x=500, host_slots=(10, 10, 50)),
    )
    upgrade = ledger.apply(
        MacroAction.MOVE_FORWARD,
        (event(ExtractionEventType.LOOT_DROP, value=10),),
        observation_before=observation(carried=30),
        state_before=state(host_x=500),
        state_after=state(host_x=500, host_slots=(10, 10, 50)),
    )
    extraction = ledger.apply(
        MacroAction.IDLE,
        (event(ExtractionEventType.EXTRACTED),),
        observation_before=observation(carried=70),
        state_before=state(host_x=500),
        state_after=state(host_x=500),
    )

    assert movement.total == 0
    assert pickup.components["novel_loot_region"] > 0
    assert upgrade.components["backpack_upgrade"] > 0
    assert extraction.components["upgrade_to_extraction"] > 0
