from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from botcolosseo.agents.duel_teachers import create_duel_teacher
from botcolosseo.agents.league_opponents import CheckpointOpponentPolicy, OpponentSpec
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState
from botcolosseo.scenarios.duel_splits import DuelCase
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.duel_rollout import (
    DuelOpponentController,
    DuelRolloutCollector,
    ScriptDuelOpponentController,
)
from botcolosseo.training.league_schedule import (
    LeagueEpisodeAssignment,
    LeagueSchedule,
)
from botcolosseo.training.rollout import RecurrentRollout


class PublicCheckpointPolicy(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: DuelActorObservation) -> MacroAction: ...


class CheckpointDuelOpponentController:
    def __init__(self, policy: PublicCheckpointPolicy) -> None:
        self._policy = policy

    def reset(self, *, seed: int) -> None:
        del seed
        self._policy.reset()

    def act(
        self,
        observation: DuelActorObservation,
        privileged_state: Callable[[], DuelPrivilegedState],
    ) -> MacroAction:
        del privileged_state
        return self._policy.act(observation)


class LeagueRolloutSchedule:
    def __init__(
        self, schedule: LeagueSchedule, *, shaping_decay_steps: int
    ) -> None:
        if shaping_decay_steps <= 0:
            raise ValueError("shaping_decay_steps must be positive")
        self.schedule = schedule
        self.shaping_decay_steps = shaping_decay_steps

    @property
    def opponent_specs(self) -> dict[str, OpponentSpec]:
        return dict(self.schedule.opponent_specs)

    def assignment(self, episode_index: int) -> LeagueEpisodeAssignment:
        if episode_index < 0:
            raise ValueError("episode_index must be nonnegative")
        pair_slot, side_index = divmod(episode_index, 2)
        return self.schedule.assignments(pair_slot)[side_index]

    def case(self, environment_steps: int, episode_index: int) -> DuelCase:
        if environment_steps < 0:
            raise ValueError("environment_steps must be nonnegative")
        assignment = self.assignment(episode_index)
        return assignment.case.to_duel_case(assignment.opponent.opponent_id)

    def shaping_scale(self, environment_steps: int) -> float:
        if environment_steps < 0:
            raise ValueError("environment_steps must be nonnegative")
        progress = min(environment_steps / self.shaping_decay_steps, 1.0)
        return 1.0 - progress


@dataclass(frozen=True)
class LeagueEpisodeResult:
    episode_index: int
    seed: int
    opponent: str
    opponent_kind: str
    source: str
    pair_slot: int
    sampling_probability: float
    learner_side: str
    decisions: int
    reward: float
    objective_completed: bool
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class LeagueRolloutCollection:
    rollout: RecurrentRollout
    environment_steps: int
    episodes: tuple[LeagueEpisodeResult, ...]
    event_counts: dict[str, int]
    reward_components: dict[str, float]


class LeagueRolloutCollector:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        schedule: LeagueSchedule,
        graph: RegionGraph,
        device: torch.device,
        shaping_decay_steps: int,
        config_path: Path = Path("assets/scenarios/crystal_run/crystal_run.cfg"),
        max_decisions: int = 525,
        episode_index: int = 0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        environment_factory: Callable[[DuelCase], Any] | None = None,
        action_sampler: Callable[[torch.distributions.Categorical], torch.Tensor]
        | None = None,
        checkpoint_loader: Callable[..., PublicCheckpointPolicy] = (
            CheckpointOpponentPolicy.load
        ),
        reward_shaper_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._schedule = LeagueRolloutSchedule(
            schedule, shaping_decay_steps=shaping_decay_steps
        )
        self._graph = graph
        self._device = device
        self._checkpoint_loader = checkpoint_loader
        self._checkpoint_cache: dict[str, PublicCheckpointPolicy] = {}
        self._collector = DuelRolloutCollector(
            model,
            curriculum=self._schedule,  # type: ignore[arg-type]
            graph=graph,
            device=device,
            config_path=config_path,
            max_decisions=max_decisions,
            episode_index=episode_index,
            gamma=gamma,
            gae_lambda=gae_lambda,
            environment_factory=environment_factory,
            opponent_factory=self._opponent_controller,
            action_sampler=action_sampler,
            reward_shaper_factory=reward_shaper_factory,
        )

    @property
    def episode_index(self) -> int:
        return self._collector.episode_index

    def _opponent_controller(
        self, case: DuelCase, graph: RegionGraph, side: str
    ) -> DuelOpponentController:
        spec = self._schedule.opponent_specs[case.opponent]
        if spec.kind == "script":
            teacher = create_duel_teacher(spec.opponent_id, graph, side=side)
            return ScriptDuelOpponentController(teacher)
        policy = self._checkpoint_cache.get(spec.opponent_id)
        if policy is None:
            policy = self._checkpoint_loader(spec, device=self._device)
            self._checkpoint_cache[spec.opponent_id] = policy
        return CheckpointDuelOpponentController(policy)

    def collect(
        self, *, steps: int, start_environment_step: int
    ) -> LeagueRolloutCollection:
        collection = self._collector.collect(
            steps=steps, start_environment_step=start_environment_step
        )
        episodes: list[LeagueEpisodeResult] = []
        for episode in collection.episodes:
            assignment = self._schedule.assignment(episode.episode_index)
            if assignment.opponent.opponent_id != episode.opponent:
                raise RuntimeError("League episode metadata does not match rollout opponent")
            episodes.append(
                LeagueEpisodeResult(
                    episode_index=episode.episode_index,
                    seed=episode.seed,
                    opponent=episode.opponent,
                    opponent_kind=assignment.opponent.kind,
                    source=assignment.source,
                    pair_slot=assignment.pair_slot,
                    sampling_probability=assignment.sampling_probability,
                    learner_side=episode.learner_side,
                    decisions=episode.decisions,
                    reward=episode.reward,
                    objective_completed=episode.objective_completed,
                    terminated=episode.terminated,
                    truncated=episode.truncated,
                )
            )
        return LeagueRolloutCollection(
            rollout=collection.rollout,
            environment_steps=collection.environment_steps,
            episodes=tuple(episodes),
            event_counts=collection.event_counts,
            reward_components=collection.reward_components,
        )

    def close(self) -> None:
        self._collector.close()
