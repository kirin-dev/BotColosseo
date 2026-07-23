from __future__ import annotations

from dataclasses import dataclass

from botcolosseo.agents.duel_teachers import (
    EXPLORER_ROUTE_CYCLE,
    ProtectiveDefensiveTeacher,
    RouteExplorerTeacher,
)
from botcolosseo.envs.defensive_signals import defensive_risk
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState
from botcolosseo.scenarios.regions import RegionGraph


@dataclass(frozen=True)
class StyleSupervisionToken:
    teacher_action: int
    supervised: bool
    route_mode: int


class DefensiveStyleSupervisor:
    def __init__(self, graph: RegionGraph, *, side: str, episode_index: int) -> None:
        del episode_index
        self.side = side
        self._teacher = ProtectiveDefensiveTeacher(graph, side=side)

    def reset(self, *, seed: int, initial_score: int) -> None:
        del initial_score
        self._teacher.reset(seed=seed)

    def route_mode(self, observation: DuelActorObservation) -> int:
        del observation
        return -1

    def label(
        self,
        state: DuelPrivilegedState,
        observation: DuelActorObservation,
    ) -> StyleSupervisionToken:
        del observation
        return StyleSupervisionToken(
            teacher_action=int(self._teacher.act(state)),
            supervised=defensive_risk(state, self.side),
            route_mode=-1,
        )


class ExplorerStyleSupervisor:
    def __init__(self, graph: RegionGraph, *, side: str, episode_index: int) -> None:
        self._teacher = RouteExplorerTeacher(graph, side=side)
        self._initial_mode = episode_index % len(EXPLORER_ROUTE_CYCLE)
        self._initial_score: int | None = None

    def reset(self, *, seed: int, initial_score: int) -> None:
        self._teacher.reset(seed=seed)
        self._initial_score = initial_score

    def route_mode(self, observation: DuelActorObservation) -> int:
        if self._initial_score is None:
            raise RuntimeError("Explorer supervisor must be reset before use")
        if observation.own_score < self._initial_score:
            raise ValueError("Explorer own score decreased within an episode")
        return (
            self._initial_mode + observation.own_score - self._initial_score
        ) % len(EXPLORER_ROUTE_CYCLE)

    def label(
        self,
        state: DuelPrivilegedState,
        observation: DuelActorObservation,
    ) -> StyleSupervisionToken:
        route_mode = self.route_mode(observation)
        return StyleSupervisionToken(
            teacher_action=int(self._teacher.act_for_mode(state, route_mode)),
            supervised=True,
            route_mode=route_mode,
        )


def create_style_supervisor(
    style: str,
    graph: RegionGraph,
    *,
    side: str,
    episode_index: int,
) -> DefensiveStyleSupervisor | ExplorerStyleSupervisor:
    if style == "defensive":
        return DefensiveStyleSupervisor(graph, side=side, episode_index=episode_index)
    if style == "explorer":
        return ExplorerStyleSupervisor(graph, side=side, episode_index=episode_index)
    raise ValueError(f"Unsupported supervised style: {style}")
