from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from botcolosseo.agents.extraction_model import (
    EXTRACTION_PRIVILEGED_DIM,
)
from botcolosseo.agents.extraction_teachers import StyledExtractionTeacher
from botcolosseo.data.extraction_demonstrations import ExtractionCase
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_rules import LifeState
from botcolosseo.envs.extraction_types import (
    ExtractionActorObservation,
    ExtractionObservations,
    ExtractionPrivilegedState,
)
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.extraction_bc import extraction_observation_tensors
from botcolosseo.training.extraction_rewards import (
    ExtractionReward,
    ExtractionTaskRewardConfig,
    ExtractionTaskRewardLedger,
)
from botcolosseo.training.rollout import RecurrentRollout, RolloutBuffer, RolloutStep


@dataclass(frozen=True)
class ExtractionEpisodeAssignment:
    case: ExtractionCase
    opponent_id: str
    opponent_kind: str = "script"
    sampling_probability: float = 1.0


class ExtractionSchedule(Protocol):
    def assignment(
        self, environment_steps: int, episode_index: int
    ) -> ExtractionEpisodeAssignment: ...

    def shaping_scale(self, environment_steps: int) -> float: ...


class ExtractionCaseSchedule:
    def __init__(
        self,
        cases: tuple[ExtractionCase, ...],
        *,
        shaping_decay_steps: int,
    ) -> None:
        if not cases or shaping_decay_steps <= 0:
            raise ValueError("Extraction schedule inputs must be positive")
        if any(case.split != "train" for case in cases):
            raise ValueError("Extraction training schedule requires train cases")
        self.cases = cases
        self.shaping_decay_steps = shaping_decay_steps

    def assignment(
        self, environment_steps: int, episode_index: int
    ) -> ExtractionEpisodeAssignment:
        if environment_steps < 0 or episode_index < 0:
            raise ValueError("Extraction schedule indices must be nonnegative")
        case = self.cases[episode_index % len(self.cases)]
        return ExtractionEpisodeAssignment(
            case=case,
            opponent_id=case.opponent_style,
            sampling_probability=1 / len(self.cases),
        )

    def shaping_scale(self, environment_steps: int) -> float:
        if environment_steps < 0:
            raise ValueError("Extraction environment steps must be nonnegative")
        return 1 - min(environment_steps / self.shaping_decay_steps, 1)


class ExtractionOpponentController(Protocol):
    def reset(self, *, seed: int) -> None: ...

    def act(
        self,
        observation: ExtractionActorObservation,
        privileged_state: Callable[[], ExtractionPrivilegedState],
    ) -> MacroAction: ...


class ScriptExtractionOpponentController:
    def __init__(self, teacher: StyledExtractionTeacher) -> None:
        self.teacher = teacher

    def reset(self, *, seed: int) -> None:
        del seed
        self.teacher.reset()

    def act(
        self,
        observation: ExtractionActorObservation,
        privileged_state: Callable[[], ExtractionPrivilegedState],
    ) -> MacroAction:
        del observation
        return MacroAction(self.teacher.act(privileged_state()))


class ExtractionStyleRewardLedger(Protocol):
    def apply(
        self,
        action: MacroAction,
        events: tuple[Any, ...],
        *,
        observation_before: ExtractionActorObservation,
        state_before: ExtractionPrivilegedState,
        state_after: ExtractionPrivilegedState,
    ) -> ExtractionReward: ...


def extraction_privileged_tensor(
    state: ExtractionPrivilegedState,
    *,
    learner_side: str,
    device: torch.device | str,
) -> torch.Tensor:
    if learner_side not in {"host", "opponent"}:
        raise ValueError("Extraction learner side is invalid")
    if learner_side == "host":
        own_pose = (state.host_x, state.host_y, state.host_angle)
        other_pose = (state.opponent_x, state.opponent_y, state.opponent_angle)
        own_health, other_health = state.host_health, state.opponent_health
        own_slots, other_slots = state.host_slots, state.opponent_slots
        own_banked, other_banked = state.host_banked, state.opponent_banked
    else:
        own_pose = (state.opponent_x, state.opponent_y, state.opponent_angle)
        other_pose = (state.host_x, state.host_y, state.host_angle)
        own_health, other_health = state.opponent_health, state.host_health
        own_slots, other_slots = state.opponent_slots, state.host_slots
        own_banked, other_banked = state.opponent_banked, state.host_banked

    def pose(values: tuple[float, float, float]) -> tuple[float, ...]:
        angle = math.radians(values[2])
        return (
            values[0] / 640,
            values[1] / 480,
            math.cos(angle),
            math.sin(angle),
        )

    values = (
        *pose(own_pose),
        *pose(other_pose),
        own_health / 100,
        other_health / 100,
        *(value / 50 for value in own_slots),
        *(value / 50 for value in other_slots),
        own_banked / 150,
        other_banked / 150,
        sum(state.cache_slots) / 150,
        state.world_loot_mask / 127,
    )
    if len(values) != EXTRACTION_PRIVILEGED_DIM:
        raise RuntimeError("Extraction privileged tensor schema drifted")
    return torch.tensor(values, dtype=torch.float32, device=device).reshape(
        1, 1, EXTRACTION_PRIVILEGED_DIM
    )


@dataclass(frozen=True)
class ExtractionTrainingEpisode:
    episode_index: int
    seed: int
    learner_side: str
    opponent_id: str
    opponent_kind: str
    sampling_probability: float
    decisions: int
    reward: float
    extracted: bool
    extracted_value: int
    died: bool
    won: bool
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class ExtractionRolloutCollection:
    rollout: RecurrentRollout
    environment_steps: int
    episodes: tuple[ExtractionTrainingEpisode, ...]
    event_counts: dict[str, int]
    reward_components: dict[str, float]


class ExtractionRolloutCollector:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        schedule: ExtractionSchedule,
        device: torch.device,
        config_path: Path = Path(
            "assets/scenarios/crystal_run_extraction/crystal_run_extraction.cfg"
        ),
        max_decisions: int = 700,
        episode_index: int = 0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        task_reward_config: ExtractionTaskRewardConfig | None = None,
        environment_factory: Callable[[ExtractionEpisodeAssignment], Any] | None = None,
        opponent_factory: Callable[
            [ExtractionEpisodeAssignment, str], ExtractionOpponentController
        ]
        | None = None,
        action_sampler: Callable[[torch.distributions.Categorical], torch.Tensor]
        | None = None,
        style_reward_factory: Callable[[str], ExtractionStyleRewardLedger]
        | None = None,
    ) -> None:
        if (
            max_decisions <= 0
            or episode_index < 0
            or not 0 <= gamma <= 1
            or not 0 <= gae_lambda <= 1
        ):
            raise ValueError("Extraction collector settings are invalid")
        self.model = model
        self.schedule = schedule
        self.device = device
        self.config_path = config_path
        self.max_decisions = max_decisions
        self.episode_index = episode_index
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.task_reward_config = task_reward_config or ExtractionTaskRewardConfig()
        self._environment_factory = environment_factory or self._make_environment
        self._opponent_factory = opponent_factory or self._make_script_opponent
        self._action_sampler = action_sampler or (
            lambda distribution: distribution.sample()
        )
        self._style_reward_factory = style_reward_factory
        self._environment: Any | None = None
        self._opponent: ExtractionOpponentController | None = None
        self._assignment: ExtractionEpisodeAssignment | None = None
        self._observations: ExtractionObservations | None = None
        self._hidden: torch.Tensor | None = None
        self._task_reward: ExtractionTaskRewardLedger | None = None
        self._style_reward: ExtractionStyleRewardLedger | None = None
        self._episode_start = True
        self._episode_decisions = 0
        self._episode_reward = 0.0

    def _make_environment(
        self, assignment: ExtractionEpisodeAssignment
    ) -> SynchronousExtractionEnv:
        return SynchronousExtractionEnv(
            config_path=self.config_path,
            seed=assignment.case.seed,
            max_decisions=self.max_decisions,
        )

    @staticmethod
    def _make_script_opponent(
        assignment: ExtractionEpisodeAssignment,
        side: str,
    ) -> ExtractionOpponentController:
        if assignment.opponent_kind != "script":
            raise ValueError("A checkpoint opponent requires an opponent factory")
        return ScriptExtractionOpponentController(
            StyledExtractionTeacher(side=side, style=assignment.opponent_id)
        )

    def _start_episode(self, environment_steps: int) -> None:
        assignment = self.schedule.assignment(environment_steps, self.episode_index)
        environment = self._environment_factory(assignment)
        opponent_side = (
            "opponent" if assignment.case.learner_side == "host" else "host"
        )
        opponent = self._opponent_factory(assignment, opponent_side)
        try:
            observations, _ = environment.reset()
            opponent.reset(seed=assignment.case.seed)
        except BaseException:
            environment.close()
            raise
        self._environment = environment
        self._opponent = opponent
        self._assignment = assignment
        self._observations = observations
        self._hidden = self.model.initial_state(1, device=self.device)
        self._task_reward = ExtractionTaskRewardLedger(
            self.task_reward_config,
            learner_side=assignment.case.learner_side,
        )
        self._style_reward = (
            None
            if self._style_reward_factory is None
            else self._style_reward_factory(assignment.case.learner_side)
        )
        self._episode_start = True
        self._episode_decisions = 0
        self._episode_reward = 0.0

    def _learner_observation(self) -> ExtractionActorObservation:
        if self._assignment is None or self._observations is None:
            raise RuntimeError("Extraction collector has no active episode")
        if self._assignment.case.learner_side == "host":
            return self._observations.host
        return self._observations.opponent

    def _opponent_observation(self) -> ExtractionActorObservation:
        if self._assignment is None or self._observations is None:
            raise RuntimeError("Extraction collector has no active episode")
        if self._assignment.case.learner_side == "host":
            return self._observations.opponent
        return self._observations.host

    @torch.no_grad()
    def collect(
        self, *, steps: int, start_environment_step: int
    ) -> ExtractionRolloutCollection:
        if steps <= 0 or start_environment_step < 0:
            raise ValueError("Extraction rollout range is invalid")
        buffer = RolloutBuffer(
            capacity=steps,
            environments=1,
            scalar_dim=9,
            privileged_dim=EXTRACTION_PRIVILEGED_DIM,
        )
        episodes: list[ExtractionTrainingEpisode] = []
        event_counts: Counter[str] = Counter()
        reward_components: Counter[str] = Counter()
        try:
            for offset in range(steps):
                global_step = start_environment_step + offset
                if self._environment is None:
                    self._start_episode(global_step)
                environment = self._environment
                opponent = self._opponent
                assignment = self._assignment
                hidden = self._hidden
                task_reward = self._task_reward
                if any(
                    item is None
                    for item in (
                        environment,
                        opponent,
                        assignment,
                        hidden,
                        task_reward,
                    )
                ):
                    raise RuntimeError("Extraction collector state is incomplete")
                observation = self._learner_observation()
                state_before = environment.privileged_state()
                inputs = extraction_observation_tensors(
                    observation,
                    episode_start=self._episode_start,
                    device=self.device,
                )
                privileged = extraction_privileged_tensor(
                    state_before,
                    learner_side=assignment.case.learner_side,
                    device=self.device,
                )
                output = self.model(*inputs, privileged, hidden)
                distribution = torch.distributions.Categorical(logits=output.logits)
                action = self._action_sampler(distribution)
                log_prob = distribution.log_prob(action)
                learner_action = MacroAction(int(action[0, 0]))
                opponent_action = opponent.act(
                    self._opponent_observation(),
                    environment.privileged_state,
                )
                host_action, away_action = (
                    (learner_action, opponent_action)
                    if assignment.case.learner_side == "host"
                    else (opponent_action, learner_action)
                )
                step = environment.step(host_action, away_action)
                self._observations = ExtractionObservations(
                    step.host,
                    step.opponent,
                )
                state_after = environment.privileged_state()
                next_value = torch.zeros(1, device=self.device)
                if not step.terminated:
                    next_inputs = extraction_observation_tensors(
                        self._learner_observation(),
                        episode_start=False,
                        device=self.device,
                    )
                    next_privileged = extraction_privileged_tensor(
                        state_after,
                        learner_side=assignment.case.learner_side,
                        device=self.device,
                    )
                    next_output = self.model(
                        *next_inputs,
                        next_privileged,
                        output.hidden,
                    )
                    next_value = next_output.values[:, 0]
                reward = (
                    step.host_reward
                    if assignment.case.learner_side == "host"
                    else step.opponent_reward
                )
                shaped = task_reward.apply(
                    learner_action,
                    step.events,
                    observation_before=observation,
                    state_before=state_before,
                    state_after=state_after,
                    scale=self.schedule.shaping_scale(global_step),
                )
                reward += shaped.total
                reward_components.update(shaped.components)
                if self._style_reward is not None:
                    styled = self._style_reward.apply(
                        learner_action,
                        step.events,
                        observation_before=observation,
                        state_before=state_before,
                        state_after=state_after,
                    )
                    reward += styled.total
                    reward_components.update(
                        {f"style:{name}": value for name, value in styled.components.items()}
                    )
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
                for item in step.events:
                    role = (
                        "learner"
                        if item.side == assignment.case.learner_side
                        else "opponent"
                    )
                    event_counts[f"{role}:{item.type.value}"] += item.count
                if step.terminated or step.truncated:
                    public = environment.protocol_snapshot().public_state(
                        assignment.case.learner_side
                    )
                    won = (
                        step.winner == 1
                        if assignment.case.learner_side == "host"
                        else step.winner == 2
                    )
                    episodes.append(
                        ExtractionTrainingEpisode(
                            episode_index=self.episode_index,
                            seed=assignment.case.seed,
                            learner_side=assignment.case.learner_side,
                            opponent_id=assignment.opponent_id,
                            opponent_kind=assignment.opponent_kind,
                            sampling_probability=assignment.sampling_probability,
                            decisions=self._episode_decisions,
                            reward=self._episode_reward,
                            extracted=public.life_state == int(LifeState.EXTRACTED),
                            extracted_value=public.banked_value,
                            died=public.life_state == int(LifeState.DEAD),
                            won=won,
                            terminated=step.terminated,
                            truncated=step.truncated,
                        )
                    )
                    self._close_episode()
                    self.episode_index += 1
        except BaseException:
            self.close()
            raise
        return ExtractionRolloutCollection(
            rollout=buffer.finalize(gamma=self.gamma, gae_lambda=self.gae_lambda),
            environment_steps=steps,
            episodes=tuple(episodes),
            event_counts=dict(sorted(event_counts.items())),
            reward_components=dict(sorted(reward_components.items())),
        )

    def _close_episode(self) -> None:
        environment, self._environment = self._environment, None
        self._opponent = None
        self._assignment = None
        self._observations = None
        self._hidden = None
        self._task_reward = None
        self._style_reward = None
        if environment is not None:
            environment.close()

    def close(self) -> None:
        self._close_episode()
