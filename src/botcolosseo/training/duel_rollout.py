from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from botcolosseo.agents.checkpoint import CheckpointMetadata
from botcolosseo.agents.duel_teachers import create_duel_teacher
from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState
from botcolosseo.envs.synchronous_duel import (
    DuelObservations,
    SynchronousDuelEnv,
)
from botcolosseo.scenarios.duel_splits import DuelCase
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.curriculum import OpponentCurriculum
from botcolosseo.training.rollout import RecurrentRollout, RolloutBuffer, RolloutStep


@dataclass(frozen=True)
class DuelEpisodeResult:
    episode_index: int
    seed: int
    opponent: str
    learner_side: str
    decisions: int
    reward: float
    objective_completed: bool
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class DuelRolloutCollection:
    rollout: RecurrentRollout
    environment_steps: int
    episodes: tuple[DuelEpisodeResult, ...]
    event_counts: dict[str, int]
    reward_components: dict[str, float]


class DuelOpponentController(Protocol):
    def reset(self, *, seed: int) -> None: ...

    def act(
        self,
        observation: DuelActorObservation,
        privileged_state: Callable[[], DuelPrivilegedState],
    ) -> MacroAction: ...


class StyleRewardShaper(Protocol):
    def apply(
        self,
        action: MacroAction,
        events: tuple[Any, ...],
        *,
        has_core: bool,
    ) -> Any: ...


class ScriptDuelOpponentController:
    def __init__(self, teacher: Any) -> None:
        self._teacher = teacher

    def reset(self, *, seed: int) -> None:
        self._teacher.reset(seed=seed)

    def act(
        self,
        observation: DuelActorObservation,
        privileged_state: Callable[[], DuelPrivilegedState],
    ) -> MacroAction:
        del observation
        return MacroAction(self._teacher.act(privileged_state()))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bc_actor_checkpoint(
    path: Path,
    model: AsymmetricActorCritic,
    *,
    expected_scenario_hash: str,
    expected_checkpoint_sha: str | None = None,
) -> CheckpointMetadata:
    path = path.expanduser().resolve()
    if expected_checkpoint_sha is not None and _sha256(path) != expected_checkpoint_sha:
        raise ValueError("BC checkpoint hash does not match")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported BC checkpoint schema version")
    metadata = CheckpointMetadata(**payload["metadata"])
    if metadata.scenario_hash != expected_scenario_hash:
        raise ValueError("BC checkpoint scenario hash does not match")
    model.actor.load_state_dict(payload["model"])
    return metadata


def actor_observation_tensors(
    observation: DuelActorObservation,
    *,
    episode_start: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    frame = torch.from_numpy(np.array(observation.frame, copy=True))
    frame = frame.to(device).reshape(1, 1, 1, 84, 84)
    scalars = torch.tensor(
        (
            observation.health / 200.0,
            observation.armor / 200.0,
            min(observation.ammo, 100.0) / 100.0,
            min(observation.own_score, 3) / 3.0,
            min(observation.opponent_score, 3) / 3.0,
            float(observation.has_core),
        ),
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 6)
    previous_action = torch.tensor(
        [[observation.previous_action]], dtype=torch.long, device=device
    )
    mask = torch.tensor(
        [[0.0 if episode_start else 1.0]], dtype=torch.float32, device=device
    )
    return frame, scalars, previous_action, mask


def privileged_tensor(
    state: DuelPrivilegedState,
    *,
    learner_side: str,
    device: torch.device | str,
) -> torch.Tensor:
    if learner_side not in ("host", "opponent"):
        raise ValueError("learner_side must be host or opponent")
    if learner_side == "host":
        own = (state.host_x, state.host_y, state.host_angle)
        other = (state.opponent_x, state.opponent_y, state.opponent_angle)
        own_carrier, other_carrier = state.carrier == 1, state.carrier == 2
    else:
        own = (state.opponent_x, state.opponent_y, state.opponent_angle)
        other = (state.host_x, state.host_y, state.host_angle)
        own_carrier, other_carrier = state.carrier == 2, state.carrier == 1

    def player(values: tuple[float, float, float]) -> tuple[float, ...]:
        angle = math.radians(values[2])
        return values[0] / 1024.0, values[1] / 640.0, math.cos(angle), math.sin(angle)

    values = (
        *player(own),
        *player(other),
        state.core_x / 1024.0,
        state.core_y / 640.0,
        float(own_carrier),
        float(other_carrier),
    )
    return torch.tensor(values, dtype=torch.float32, device=device).reshape(1, 1, 12)


class DuelRolloutCollector:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        curriculum: OpponentCurriculum,
        graph: RegionGraph,
        device: torch.device,
        config_path: Path = Path(
            "assets/scenarios/crystal_run/crystal_run.cfg"
        ),
        max_decisions: int = 525,
        episode_index: int = 0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        environment_factory: Callable[[DuelCase], Any] | None = None,
        teacher_factory: Callable[..., Any] = create_duel_teacher,
        opponent_factory: Callable[
            [DuelCase, RegionGraph, str], DuelOpponentController
        ]
        | None = None,
        action_sampler: Callable[[torch.distributions.Categorical], torch.Tensor]
        | None = None,
        reward_shaper_factory: Callable[[str], StyleRewardShaper] | None = None,
    ) -> None:
        if (
            max_decisions <= 0
            or episode_index < 0
            or not 0.0 <= gamma <= 1.0
            or not 0.0 <= gae_lambda <= 1.0
        ):
            raise ValueError("Invalid collector episode settings")
        self.model = model
        self.curriculum = curriculum
        self.graph = graph
        self.device = device
        self.config_path = config_path
        self.max_decisions = max_decisions
        self.episode_index = episode_index
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._environment_factory = environment_factory or self._make_environment
        self._teacher_factory = teacher_factory
        self._opponent_factory = opponent_factory or self._make_script_opponent
        self._action_sampler = action_sampler or (lambda distribution: distribution.sample())
        self._reward_shaper_factory = reward_shaper_factory
        self._reward_shaper: StyleRewardShaper | None = None
        self._environment: Any | None = None
        self._opponent_controller: DuelOpponentController | None = None
        self._observations: DuelObservations | None = None
        self._case: DuelCase | None = None
        self._hidden: torch.Tensor | None = None
        self._episode_start = True
        self._episode_decisions = 0
        self._episode_reward = 0.0
        self._initial_score = 0

    def _make_environment(self, case: DuelCase) -> SynchronousDuelEnv:
        return SynchronousDuelEnv(
            config_path=self.config_path,
            region_graph=self.graph,
            seed=case.seed,
            max_decisions=self.max_decisions,
        )

    def _make_script_opponent(
        self, case: DuelCase, graph: RegionGraph, side: str
    ) -> DuelOpponentController:
        teacher = self._teacher_factory(case.opponent, graph, side=side)
        return ScriptDuelOpponentController(teacher)

    def _start_episode(self, environment_steps: int) -> None:
        case = self.curriculum.case(environment_steps, self.episode_index)
        environment = self._environment_factory(case)
        opponent_side = "opponent" if case.learner_side == "host" else "host"
        opponent = self._opponent_factory(case, self.graph, opponent_side)
        try:
            observations, _ = environment.reset()
            opponent.reset(seed=case.seed)
        except BaseException:
            environment.close()
            raise
        learner = observations.host if case.learner_side == "host" else observations.opponent
        self._environment = environment
        self._opponent_controller = opponent
        self._observations = observations
        self._case = case
        self._hidden = torch.zeros(1, 1, 256, device=self.device)
        self._episode_start = True
        self._episode_decisions = 0
        self._episode_reward = 0.0
        self._initial_score = learner.own_score
        self._reward_shaper = (
            None
            if self._reward_shaper_factory is None
            else self._reward_shaper_factory(case.learner_side)
        )

    def _learner_observation(self) -> DuelActorObservation:
        if self._case is None or self._observations is None:
            raise RuntimeError("Duel collector has no active episode")
        return (
            self._observations.host
            if self._case.learner_side == "host"
            else self._observations.opponent
        )

    def _opponent_observation(self) -> DuelActorObservation:
        if self._case is None or self._observations is None:
            raise RuntimeError("Duel collector has no active episode")
        return (
            self._observations.opponent
            if self._case.learner_side == "host"
            else self._observations.host
        )

    @torch.no_grad()
    def collect(
        self, *, steps: int, start_environment_step: int
    ) -> DuelRolloutCollection:
        if steps <= 0 or start_environment_step < 0:
            raise ValueError("Invalid rollout collection range")
        buffer = RolloutBuffer(capacity=steps, environments=1)
        episodes: list[DuelEpisodeResult] = []
        events: Counter[str] = Counter()
        reward_components: Counter[str] = Counter()
        try:
            for offset in range(steps):
                global_step = start_environment_step + offset
                if self._environment is None:
                    self._start_episode(global_step)
                environment = self._environment
                opponent = self._opponent_controller
                case = self._case
                hidden = self._hidden
                if environment is None or opponent is None or case is None or hidden is None:
                    raise RuntimeError("Duel collector episode state is incomplete")
                environment.set_shaping_scale(
                    self.curriculum.shaping_scale(global_step)
                )
                observation = self._learner_observation()
                inputs = actor_observation_tensors(
                    observation,
                    episode_start=self._episode_start,
                    device=self.device,
                )
                privileged = privileged_tensor(
                    environment.teacher_state(),
                    learner_side=case.learner_side,
                    device=self.device,
                )
                output = self.model(*inputs, privileged, hidden)
                distribution = torch.distributions.Categorical(logits=output.logits)
                action = self._action_sampler(distribution)
                log_prob = distribution.log_prob(action)
                learner_action = MacroAction(int(action[0, 0]))
                opponent_action = opponent.act(
                    self._opponent_observation(), environment.teacher_state
                )
                host_action, away_action = (
                    (learner_action, opponent_action)
                    if case.learner_side == "host"
                    else (opponent_action, learner_action)
                )
                step = environment.step(host_action, away_action)
                self._observations = DuelObservations(step.host, step.opponent)
                next_value = torch.zeros(1, device=self.device)
                if not step.terminated:
                    next_observation = self._learner_observation()
                    next_inputs = actor_observation_tensors(
                        next_observation, episode_start=False, device=self.device
                    )
                    next_privileged = privileged_tensor(
                        environment.teacher_state(),
                        learner_side=case.learner_side,
                        device=self.device,
                    )
                    next_output = self.model(
                        *next_inputs, next_privileged, output.hidden
                    )
                    next_value = next_output.values[:, 0]
                reward = (
                    step.host_reward
                    if case.learner_side == "host"
                    else step.opponent_reward
                )
                if self._reward_shaper is not None:
                    shaped = self._reward_shaper.apply(
                        learner_action,
                        step.events,
                        has_core=observation.has_core,
                    )
                    reward += float(shaped.total)
                    reward_components.update(shaped.components)
                buffer.append(
                    RolloutStep(
                        frames=inputs[0][:, 0].cpu(),
                        scalars=inputs[1][:, 0].cpu(),
                        previous_actions=inputs[2][:, 0].cpu(),
                        masks=inputs[3][:, 0].cpu(),
                        privileged=privileged[:, 0].cpu(),
                        hidden=hidden.cpu(),
                        actions=action[:, 0].cpu(),
                        rewards=torch.tensor([reward], dtype=torch.float32),
                        terminated=torch.tensor([step.terminated]),
                        truncated=torch.tensor([step.truncated]),
                        log_probs=log_prob[:, 0].cpu(),
                        values=output.values[:, 0].cpu(),
                        next_values=next_value.cpu(),
                    )
                )
                self._hidden = output.hidden.detach()
                self._episode_start = False
                self._episode_decisions += 1
                self._episode_reward += reward
                for event in step.events:
                    role = "learner" if event.side == case.learner_side else "opponent"
                    events[f"{role}:{event.type.value}"] += 1
                if step.terminated or step.truncated:
                    learner = self._learner_observation()
                    episodes.append(
                        DuelEpisodeResult(
                            episode_index=self.episode_index,
                            seed=case.seed,
                            opponent=case.opponent,
                            learner_side=case.learner_side,
                            decisions=self._episode_decisions,
                            reward=self._episode_reward,
                            objective_completed=learner.own_score > self._initial_score,
                            terminated=step.terminated,
                            truncated=step.truncated,
                        )
                    )
                    self._close_episode()
                    self.episode_index += 1
        except BaseException:
            self.close()
            raise
        return DuelRolloutCollection(
            rollout=buffer.finalize(
                gamma=self.gamma, gae_lambda=self.gae_lambda
            ),
            environment_steps=steps,
            episodes=tuple(episodes),
            event_counts=dict(sorted(events.items())),
            reward_components=dict(sorted(reward_components.items())),
        )

    def _close_episode(self) -> None:
        environment, self._environment = self._environment, None
        self._opponent_controller = None
        self._observations = None
        self._case = None
        self._hidden = None
        self._reward_shaper = None
        if environment is not None:
            environment.close()

    def close(self) -> None:
        self._close_episode()
