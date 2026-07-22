from __future__ import annotations

from dataclasses import dataclass

from botcolosseo.scenarios.duel_splits import (
    DUEL_OPPONENTS,
    DuelCase,
)


@dataclass(frozen=True)
class CurriculumPhase:
    start_environment_step: int
    opponents: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.start_environment_step < 0 or not self.opponents:
            raise ValueError("Curriculum phases require a step and opponents")
        if len(self.opponents) != len(set(self.opponents)):
            raise ValueError("Curriculum opponents must be unique")
        unknown = set(self.opponents).difference(DUEL_OPPONENTS)
        if unknown:
            raise ValueError(f"Unknown curriculum opponents: {sorted(unknown)}")


class OpponentCurriculum:
    def __init__(
        self,
        cases: tuple[DuelCase, ...],
        *,
        phases: tuple[CurriculumPhase, ...],
        shaping_decay_steps: int,
    ) -> None:
        if not cases or any(case.split != "train" for case in cases):
            raise ValueError("Opponent curriculum may access train cases only")
        if not phases or phases[0].start_environment_step != 0:
            raise ValueError("Curriculum phases must start at environment step zero")
        starts = [phase.start_environment_step for phase in phases]
        if starts != sorted(set(starts)):
            raise ValueError("Curriculum phase starts must be strictly increasing")
        if shaping_decay_steps <= 0:
            raise ValueError("shaping_decay_steps must be positive")
        self._phases = phases
        self.shaping_decay_steps = shaping_decay_steps
        grouped: dict[str, dict[int, dict[str, DuelCase]]] = {}
        for case in cases:
            grouped.setdefault(case.opponent, {}).setdefault(case.pair_index, {})[
                case.learner_side
            ] = case
        required = set().union(*(phase.opponents for phase in phases))
        if missing := required.difference(grouped):
            raise ValueError(f"Curriculum cases missing opponents: {sorted(missing)}")
        self._pairs: dict[str, tuple[dict[str, DuelCase], ...]] = {}
        for opponent in required:
            pairs = tuple(grouped[opponent][index] for index in sorted(grouped[opponent]))
            if any(set(pair) != {"host", "opponent"} for pair in pairs):
                raise ValueError("Curriculum cases must contain paired side swaps")
            self._pairs[opponent] = pairs

    def opponents(self, environment_steps: int) -> tuple[str, ...]:
        if environment_steps < 0:
            raise ValueError("environment_steps must be nonnegative")
        selected = self._phases[0]
        for phase in self._phases[1:]:
            if environment_steps < phase.start_environment_step:
                break
            selected = phase
        return selected.opponents

    def case(self, environment_steps: int, episode_index: int) -> DuelCase:
        if episode_index < 0:
            raise ValueError("episode_index must be nonnegative")
        opponents = self.opponents(environment_steps)
        pair_slot = episode_index // 2
        opponent = opponents[pair_slot % len(opponents)]
        opponent_cycle = pair_slot // len(opponents)
        pairs = self._pairs[opponent]
        pair = pairs[opponent_cycle % len(pairs)]
        side = "host" if episode_index % 2 == 0 else "opponent"
        return pair[side]

    def shaping_scale(self, environment_steps: int) -> float:
        if environment_steps < 0:
            raise ValueError("environment_steps must be nonnegative")
        progress = min(environment_steps / self.shaping_decay_steps, 1.0)
        return 1.0 - progress
