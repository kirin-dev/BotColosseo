from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from botcolosseo.data.extraction_demonstrations import ExtractionCase
from botcolosseo.training.extraction_rollout import ExtractionEpisodeAssignment
from botcolosseo.training.pfsp import pfsp_probabilities


@dataclass(frozen=True)
class ExtractionHistoricalOpponent:
    opponent_id: str
    checkpoint: Path
    checkpoint_sha256: str
    environment_steps: int

    def __post_init__(self) -> None:
        if (
            not self.opponent_id
            or not self.checkpoint.is_file()
            or len(self.checkpoint_sha256) != 64
            or self.environment_steps <= 0
        ):
            raise ValueError("Invalid Extraction historical opponent")


class ExtractionPFSPSchedule:
    """Deterministic script/history mixture with online lightweight PFSP."""

    def __init__(
        self,
        cases: tuple[ExtractionCase, ...],
        *,
        shaping_decay_steps: int,
        master_seed: int,
        history_probability: float = 0.5,
    ) -> None:
        if (
            not cases
            or shaping_decay_steps <= 0
            or master_seed < 0
            or not 0 <= history_probability <= 1
        ):
            raise ValueError("Invalid Extraction PFSP schedule")
        if any(case.split != "train" for case in cases):
            raise ValueError("Extraction PFSP schedule requires train cases")
        self.cases = cases
        self.shaping_decay_steps = shaping_decay_steps
        self.master_seed = master_seed
        self.history_probability = history_probability
        self._history: dict[str, ExtractionHistoricalOpponent] = {}
        self._outcomes: dict[str, list[float]] = {}

    @property
    def historical_opponents(self) -> tuple[ExtractionHistoricalOpponent, ...]:
        return tuple(self._history[key] for key in sorted(self._history))

    @property
    def win_rates(self) -> dict[str, float]:
        return {
            opponent_id: (
                sum(outcomes) / len(outcomes) if outcomes else 0.5
            )
            for opponent_id, outcomes in sorted(self._outcomes.items())
        }

    def state_dict(self) -> dict[str, list[float]]:
        return {
            opponent_id: list(outcomes)
            for opponent_id, outcomes in sorted(self._outcomes.items())
        }

    def load_state_dict(self, state: dict[str, list[float]]) -> None:
        if set(state) != set(self._history):
            raise ValueError("Extraction PFSP state does not match opponent pool")
        restored: dict[str, list[float]] = {}
        for opponent_id, outcomes in state.items():
            if (
                not isinstance(outcomes, list)
                or len(outcomes) > 200
                or any(value not in {0.0, 0.5, 1.0} for value in outcomes)
            ):
                raise ValueError("Extraction PFSP outcomes are invalid")
            restored[opponent_id] = [float(value) for value in outcomes]
        self._outcomes = restored

    def add(self, opponent: ExtractionHistoricalOpponent) -> bool:
        existing = self._history.get(opponent.opponent_id)
        if existing is not None:
            if existing != opponent:
                raise ValueError("Extraction historical opponent identity drifted")
            return False
        if any(
            item.checkpoint_sha256 == opponent.checkpoint_sha256
            for item in self._history.values()
        ):
            raise ValueError("Extraction historical checkpoint is duplicated")
        self._history[opponent.opponent_id] = opponent
        self._outcomes[opponent.opponent_id] = []
        return True

    def _uniform(self, episode_index: int, stream: str) -> float:
        payload = f"{self.master_seed}:{episode_index}:{stream}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64

    def _historical_choice(
        self, episode_index: int
    ) -> tuple[ExtractionHistoricalOpponent, float]:
        probabilities = pfsp_probabilities(self.win_rates)
        draw = self._uniform(episode_index, "pfsp")
        cumulative = 0.0
        for opponent_id, probability in probabilities.items():
            cumulative += probability
            if draw < cumulative:
                return self._history[opponent_id], probability
        opponent_id = next(reversed(probabilities))
        return self._history[opponent_id], probabilities[opponent_id]

    def assignment(
        self, environment_steps: int, episode_index: int
    ) -> ExtractionEpisodeAssignment:
        if environment_steps < 0 or episode_index < 0:
            raise ValueError("Extraction PFSP indices must be nonnegative")
        case = self.cases[episode_index % len(self.cases)]
        pair_slot = episode_index // 2
        choose_history = (
            self._history
            and self._uniform(pair_slot, "source") < self.history_probability
        )
        if choose_history:
            opponent, conditional = self._historical_choice(pair_slot)
            return ExtractionEpisodeAssignment(
                case=case,
                opponent_id=opponent.opponent_id,
                opponent_kind="checkpoint",
                sampling_probability=self.history_probability * conditional,
            )
        script_probability = (
            1.0 if not self._history else 1.0 - self.history_probability
        )
        return ExtractionEpisodeAssignment(
            case=case,
            opponent_id=case.opponent_style,
            opponent_kind="script",
            sampling_probability=script_probability / len(self.cases),
        )

    def record_result(
        self,
        assignment: ExtractionEpisodeAssignment,
        *,
        won: bool,
        draw: bool,
    ) -> None:
        if assignment.opponent_kind != "checkpoint":
            return
        if assignment.opponent_id not in self._outcomes:
            raise ValueError("Extraction PFSP result references an unknown opponent")
        value = 0.5 if draw else float(won)
        outcomes = self._outcomes[assignment.opponent_id]
        outcomes.append(value)
        if len(outcomes) > 200:
            del outcomes[:-200]

    def shaping_scale(self, environment_steps: int) -> float:
        if environment_steps < 0:
            raise ValueError("Extraction environment steps must be nonnegative")
        return 1 - min(environment_steps / self.shaping_decay_steps, 1)
