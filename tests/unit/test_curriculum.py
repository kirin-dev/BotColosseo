from pathlib import Path

import pytest

from botcolosseo.envs.duel_protocol import DuelEvent, DuelEventType
from botcolosseo.envs.duel_rewards import DuelRewardLedger, load_reward_config
from botcolosseo.scenarios.duel_splits import generate_duel_splits
from botcolosseo.training.curriculum import CurriculumPhase, OpponentCurriculum


def curriculum() -> OpponentCurriculum:
    cases = generate_duel_splits(master_seed=17, pairs_per_opponent=3)["train"]
    return OpponentCurriculum(
        cases,
        phases=(
            CurriculumPhase(0, ("random_legal", "fixed_route")),
            CurriculumPhase(
                100, ("random_legal", "fixed_route", "objective_first")
            ),
            CurriculumPhase(
                300,
                (
                    "random_legal",
                    "fixed_route",
                    "objective_first",
                    "aggressive_script",
                    "defensive_script",
                ),
            ),
        ),
        shaping_decay_steps=200,
    )


def test_curriculum_unlocks_fixed_opponent_phases() -> None:
    schedule = curriculum()

    assert schedule.opponents(0) == ("random_legal", "fixed_route")
    assert schedule.opponents(99) == ("random_legal", "fixed_route")
    assert schedule.opponents(100) == (
        "random_legal",
        "fixed_route",
        "objective_first",
    )
    assert schedule.opponents(300)[-2:] == (
        "aggressive_script",
        "defensive_script",
    )


def test_curriculum_is_train_only_balanced_and_side_swapped() -> None:
    schedule = curriculum()
    selected = [schedule.case(0, episode) for episode in range(8)]

    assert all(case.split == "train" for case in selected)
    assert [case.opponent for case in selected] == [
        "random_legal",
        "random_legal",
        "fixed_route",
        "fixed_route",
    ] * 2
    for first, second in zip(selected[::2], selected[1::2], strict=True):
        assert first.seed == second.seed
        assert first.pair_index == second.pair_index
        assert (first.learner_side, second.learner_side) == ("host", "opponent")


def test_curriculum_rejects_non_train_case_access() -> None:
    validation = generate_duel_splits(master_seed=17, pairs_per_opponent=1)[
        "validation"
    ]

    with pytest.raises(ValueError, match="train"):
        OpponentCurriculum(
            validation,
            phases=(CurriculumPhase(0, ("random_legal",)),),
            shaping_decay_steps=100,
        )


def test_shaping_scale_decays_without_changing_task_rewards() -> None:
    schedule = curriculum()
    assert schedule.shaping_scale(0) == pytest.approx(1.0)
    assert schedule.shaping_scale(100) == pytest.approx(0.5)
    assert schedule.shaping_scale(200) == pytest.approx(0.0)
    assert schedule.shaping_scale(1_000) == pytest.approx(0.0)

    ledger = DuelRewardLedger(load_reward_config(Path("configs/m2/reward.yaml")))
    events = tuple(
        DuelEvent(kind, "host", 0, index, index * 4)
        for index, kind in enumerate(
            (
                DuelEventType.PICKUP,
                DuelEventType.VALID_HIT,
                DuelEventType.SCORE,
                DuelEventType.DEATH,
            ),
            start=1,
        )
    )
    full = ledger.apply(events, shaping_scale=1.0)
    ledger.reset()
    decayed = ledger.apply(events, shaping_scale=0.0)

    assert full.host == pytest.approx(0.87)
    assert decayed.host == pytest.approx(0.75)
    assert decayed.opponent == pytest.approx(-0.75)
