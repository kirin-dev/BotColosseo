from dataclasses import replace

import numpy as np
import pytest

from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.envs.duel_types import DuelActorObservation, DuelStep
from botcolosseo.evaluation.sync_audit import SyncAuditAccumulator, compose_duel_frame


def observation(score: int = 0) -> DuelActorObservation:
    return DuelActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=100.0,
        armor=0.0,
        ammo=50.0,
        own_score=score,
        opponent_score=0,
        has_core=False,
        previous_action=0,
    )


def step(*, index: int, tic: int, lag: int = 0) -> DuelStep:
    return DuelStep(
        host=observation(),
        opponent=observation(),
        host_reward=0.0,
        opponent_reward=0.0,
        terminated=False,
        truncated=False,
        events=(),
        decision_index=index,
        engine_tic=tic,
        peer_tic_lag=lag,
    )


def test_accumulator_requires_four_tics_except_after_death() -> None:
    audit = SyncAuditAccumulator(target_decisions=3)
    audit.start_episode(initial_tic=10)
    audit.record(step(index=1, tic=14, lag=1))
    death = DuelEvent(DuelEventType.DEATH, "host", 0, 2, 18)
    audit.record(replace(step(index=2, tic=18), events=(death,)))
    audit.record(step(index=3, tic=58, lag=2))

    summary = audit.finish(cleaned_workers=True)

    assert summary["completed_decisions"] == 3
    assert summary["event_counts"] == {"host:death": 1}
    assert summary["max_peer_tic_lag"] == 2
    assert summary["passed"] is True


def test_accumulator_rejects_unexplained_tic_jump_and_wrong_total() -> None:
    audit = SyncAuditAccumulator(target_decisions=2)
    audit.start_episode(initial_tic=10)
    with pytest.raises(RuntimeError, match="four tics"):
        audit.record(step(index=1, tic=15))
    with pytest.raises(RuntimeError, match="exactly 2"):
        audit.finish(cleaned_workers=True)


def test_duel_video_is_side_by_side_and_public_overlay_only() -> None:
    frame = compose_duel_frame(step(index=1, tic=14), event_label="none")

    assert frame.shape == (200, 336, 3)
    assert frame.dtype == np.uint8
