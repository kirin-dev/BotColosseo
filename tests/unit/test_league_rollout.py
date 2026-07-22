from pathlib import Path

import numpy as np
import torch

from botcolosseo.agents.league_opponents import OpponentSpec
from botcolosseo.agents.model import ActorCriticOutput
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState, DuelStep
from botcolosseo.envs.synchronous_duel import DuelObservations
from botcolosseo.scenarios.league_splits import generate_league_splits
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.historical_pool import HistoricalPoolManifest, PoolEntry
from botcolosseo.training.league_rollout import (
    CheckpointDuelOpponentController,
    LeagueRolloutCollector,
    LeagueRolloutSchedule,
)
from botcolosseo.training.league_schedule import LeagueSchedule


def _observation(*, score: int = 0, previous_action: int = 0) -> DuelActorObservation:
    return DuelActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=100.0,
        armor=0.0,
        ammo=20.0,
        own_score=score,
        opponent_score=0,
        has_core=False,
        previous_action=previous_action,
    )


def _state() -> DuelPrivilegedState:
    return DuelPrivilegedState(
        host_x=-1.0,
        host_y=0.0,
        host_angle=0.0,
        host_region="home",
        opponent_x=1.0,
        opponent_y=0.0,
        opponent_angle=180.0,
        opponent_region="away",
        core_x=0.0,
        core_y=0.0,
        carrier=0,
        host_health=100.0,
        opponent_health=100.0,
        host_score=0,
        opponent_score=0,
        round_state=1,
        engine_tic=0,
    )


def _entry(index: int, *, anchor: bool) -> PoolEntry:
    return PoolEntry(
        policy_id=f"policy-{index}",
        checkpoint=f"runs/policy-{index}.pt",
        checkpoint_sha256=f"{index + 1:064x}",
        scenario_hash="scenario",
        config_hash="config",
        source_git_commit="a" * 40,
        parent_checkpoint_sha256="b" * 64,
        environment_steps=index * 200_000,
        admitted_at_utc=f"2026-07-21T0{index}:00:00Z",
        validation_report=f"reports/validation-{index}.json",
        validation_report_sha256=f"{index + 11:064x}",
        script_average_win_rate=0.75,
        script_worst_case_win_rate=0.60,
        objective_rate=0.90,
        payoff_by_policy={"axis": index / 2},
        anchor=anchor,
        admission_reason="anchor" if anchor else "diversity",
    )


def _league_schedule() -> LeagueSchedule:
    pool = HistoricalPoolManifest(
        schema_version=1,
        pool_version=0,
        parent_manifest_sha256=None,
        created_at_utc="2026-07-21T00:00:00Z",
        entries=(_entry(0, anchor=True), _entry(1, anchor=False)),
    )
    script = OpponentSpec(
        opponent_id="objective_first",
        kind="script",
        checkpoint=None,
        checkpoint_sha256=None,
        scenario_hash="scenario",
        selection_evidence="builtin:objective_first",
    )
    return LeagueSchedule(
        cases=generate_league_splits()["train"],
        scripts=(script,),
        pool=pool,
        win_rates={"policy-0": 0.2, "policy-1": 0.8},
        payoff_hash="c" * 64,
    )


class FakeCheckpointPolicy:
    def __init__(self) -> None:
        self.resets = 0
        self.observations: list[DuelActorObservation] = []

    def reset(self) -> None:
        self.resets += 1

    def act(self, observation: DuelActorObservation) -> MacroAction:
        self.observations.append(observation)
        return MacroAction.ATTACK


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, frames, scalars, previous_actions, masks, privileged, hidden):
        del scalars, previous_actions, masks
        batch, time = frames.shape[:2]
        logits = torch.zeros(batch, time, 13) + self.anchor
        values = torch.zeros(batch, time) + self.anchor
        return ActorCriticOutput(logits, values, hidden + 1.0 + self.anchor)


class FakeEnv:
    def __init__(self, case) -> None:
        self.case = case
        self.steps = 0
        self.closed = False

    def reset(self):
        return DuelObservations(_observation(), _observation(previous_action=3)), object()

    def teacher_state(self) -> DuelPrivilegedState:
        return _state()

    def set_shaping_scale(self, scale: float) -> None:
        self.scale = scale

    def step(self, host_action, opponent_action) -> DuelStep:
        del host_action, opponent_action
        self.steps += 1
        done = self.steps == 2
        return DuelStep(
            host=_observation(score=int(done and self.case.learner_side == "host")),
            opponent=_observation(
                score=int(done and self.case.learner_side == "opponent")
            ),
            host_reward=1.0,
            opponent_reward=-1.0,
            terminated=done,
            truncated=False,
            events=(),
            decision_index=self.steps,
            engine_tic=self.steps * 4,
            peer_tic_lag=0,
            pre_action_tics=0,
            action_tics=4,
        )

    def close(self) -> None:
        self.closed = True


def test_checkpoint_controller_never_resolves_privileged_state() -> None:
    policy = FakeCheckpointPolicy()
    controller = CheckpointDuelOpponentController(policy)
    controller.reset(seed=7)

    action = controller.act(
        _observation(),
        lambda: (_ for _ in ()).throw(AssertionError("privileged state accessed")),
    )

    assert action == MacroAction.ATTACK
    assert policy.resets == 1


def test_league_rollout_schedule_preserves_pair_and_decay() -> None:
    schedule = LeagueRolloutSchedule(_league_schedule(), shaping_decay_steps=100)

    first = schedule.assignment(4)
    second = schedule.assignment(5)

    assert first.pair_slot == second.pair_slot == 2
    assert first.opponent == second.opponent
    assert (first.case.learner_side, second.case.learner_side) == ("host", "opponent")
    assert schedule.case(10, 4).opponent == first.opponent.opponent_id
    assert schedule.shaping_scale(25) == 0.75


def test_league_collector_records_assignment_and_caches_checkpoint_policy() -> None:
    schedule = _league_schedule()
    historical_slot = next(
        slot for slot in range(100) if schedule.assignments(slot)[0].opponent.kind == "checkpoint"
    )
    policy = FakeCheckpointPolicy()
    loads: list[str] = []
    environments: list[FakeEnv] = []

    def load_policy(spec, *, device):
        del device
        loads.append(spec.opponent_id)
        return policy

    def make_environment(case):
        environment = FakeEnv(case)
        environments.append(environment)
        return environment

    collector = LeagueRolloutCollector(
        FakeModel(),
        schedule=schedule,
        graph=RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml")),
        device=torch.device("cpu"),
        shaping_decay_steps=100,
        episode_index=historical_slot * 2,
        environment_factory=make_environment,
        checkpoint_loader=load_policy,
        action_sampler=lambda distribution: distribution.logits.argmax(dim=-1),
    )
    try:
        collection = collector.collect(steps=2, start_environment_step=0)
    finally:
        collector.close()

    episode = collection.episodes[0]
    assert episode.pair_slot == historical_slot
    assert episode.opponent_kind == "checkpoint"
    assert episode.source in {"pfsp", "uniform_history"}
    assert episode.sampling_probability in {0.5, 0.1}
    assert loads == [episode.opponent]
    assert policy.resets == 1
    assert [item.previous_action for item in policy.observations] == [3, 0]
    assert environments[0].closed
