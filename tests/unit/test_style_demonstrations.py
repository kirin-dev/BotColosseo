from pathlib import Path

import numpy as np
import pytest

from botcolosseo.data.defensive_demonstrations import (
    DefensiveStepEvidence,
    label_defensive_windows,
)
from botcolosseo.data.demonstrations import (
    DemonstrationBuffer,
    load_demonstration_shard,
    write_demonstration_shard,
)
from botcolosseo.data.style_demonstrations import aggressive_supervision
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.envs.duel_types import DuelActorObservation


def _event(side: str) -> DuelEvent:
    return DuelEvent(DuelEventType.VALID_HIT, side, 0, 0, 4)


def _typed_event(side: str, event_type: DuelEventType, index: int) -> DuelEvent:
    return DuelEvent(event_type, side, 0, index, 4 * (index + 1))


def _observation() -> DuelActorObservation:
    return DuelActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=100.0,
        armor=0.0,
        ammo=10.0,
        own_score=0,
        opponent_score=0,
        has_core=False,
        previous_action=0,
    )


def test_success_filter_keeps_only_learner_hits_and_bounded_non_attacks() -> None:
    assert aggressive_supervision(
        MacroAction.ATTACK,
        (_event("host"),),
        learner_side="host",
        non_attack_index=0,
        non_attack_stride=4,
    ) == (True, "successful_attack")
    assert aggressive_supervision(
        MacroAction.FORWARD_ATTACK,
        (_event("opponent"),),
        learner_side="host",
        non_attack_index=0,
        non_attack_stride=4,
    ) == (False, "rejected_attack")
    assert aggressive_supervision(
        MacroAction.MOVE_FORWARD,
        (),
        learner_side="host",
        non_attack_index=4,
        non_attack_stride=4,
    ) == (True, "selected_non_attack")
    assert aggressive_supervision(
        MacroAction.TURN_LEFT,
        (),
        learner_side="host",
        non_attack_index=5,
        non_attack_stride=4,
    ) == (False, "skipped_non_attack")


def test_masked_style_shard_requires_explicit_loader_opt_in(tmp_path: Path) -> None:
    buffer = DemonstrationBuffer(capacity=2, allow_masked=True)
    for index, valid in enumerate((True, False)):
        buffer.append(
            _observation(),
            teacher_action=int(MacroAction.ATTACK),
            episode_start=index == 0,
            opponent_id=0,
            task_id=1,
            train_seed=7,
            valid=valid,
        )
    path = write_demonstration_shard(
        buffer.arrays(), tmp_path / "style.npz", require_all_valid=False
    )

    with pytest.raises(ValueError, match="all-true"):
        load_demonstration_shard(path)
    loaded = load_demonstration_shard(path, require_all_valid=False)
    assert loaded["valid_mask"].tolist() == [True, False]


def test_defensive_window_keeps_verified_denial_and_rejects_concession() -> None:
    denial = DefensiveStepEvidence(
        risk=True,
        risk_after=False,
        opponent_carrier=True,
        loose_core_in_defensive_half=False,
        events=(
            _typed_event("host", DuelEventType.VALID_HIT, 0),
            _typed_event("opponent", DuelEventType.DROP, 0),
        ),
    )
    conceded = DefensiveStepEvidence(
        risk=True,
        risk_after=False,
        opponent_carrier=True,
        loose_core_in_defensive_half=False,
        events=(_typed_event("opponent", DuelEventType.SCORE, 1),),
    )

    labels = label_defensive_windows((denial, conceded), learner_side="host")

    assert labels.selected == (True, False)
    assert labels.reasons == ("successful_denial", "rejected_risk_window")
    assert labels.successful_windows == 1
    assert labels.denial_recovery_windows == 1


def test_defensive_window_keeps_recovery_and_safe_resolution_with_progress() -> None:
    recovery = DefensiveStepEvidence(
        risk=True,
        risk_after=False,
        opponent_carrier=False,
        loose_core_in_defensive_half=True,
        events=(_typed_event("host", DuelEventType.PICKUP, 0),),
    )
    resolved = DefensiveStepEvidence(
        risk=True,
        risk_after=False,
        opponent_carrier=False,
        loose_core_in_defensive_half=False,
        events=(),
    )
    progress = DefensiveStepEvidence(
        risk=False,
        risk_after=False,
        opponent_carrier=False,
        loose_core_in_defensive_half=False,
        events=(_typed_event("host", DuelEventType.PICKUP, 2),),
    )

    labels = label_defensive_windows(
        (recovery, resolved, progress), learner_side="host"
    )

    assert labels.selected == (True, True, False)
    assert labels.reasons[:2] == ("successful_recovery", "resolved_to_objective")
    assert labels.successful_windows == 2
    assert labels.denial_recovery_windows == 1


def test_defensive_window_splits_drop_from_following_recovery() -> None:
    denial = DefensiveStepEvidence(
        risk=True,
        risk_after=True,
        opponent_carrier=True,
        loose_core_in_defensive_half=False,
        events=(
            _typed_event("host", DuelEventType.VALID_HIT, 0),
            _typed_event("opponent", DuelEventType.DROP, 0),
        ),
    )
    recovery = DefensiveStepEvidence(
        risk=True,
        risk_after=False,
        opponent_carrier=False,
        loose_core_in_defensive_half=True,
        events=(_typed_event("host", DuelEventType.PICKUP, 1),),
    )

    labels = label_defensive_windows((denial, recovery), learner_side="host")

    assert labels.selected == (True, True)
    assert labels.reasons == ("successful_denial", "successful_recovery")
    assert labels.successful_windows == 2
    assert labels.denial_recovery_windows == 2
