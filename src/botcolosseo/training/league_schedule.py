from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from botcolosseo.agents.league_opponents import OpponentSpec
from botcolosseo.scenarios.league_splits import LeagueCase
from botcolosseo.training.historical_pool import HistoricalPoolManifest
from botcolosseo.training.pfsp import pfsp_probabilities, stable_uniform

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class LeagueEpisodeAssignment:
    pair_slot: int
    case: LeagueCase
    opponent: OpponentSpec
    source: Literal["script", "pfsp", "uniform_history"]
    sampling_probability: float


class LeagueSchedule:
    def __init__(
        self,
        *,
        cases: Sequence[LeagueCase],
        scripts: Sequence[OpponentSpec],
        pool: HistoricalPoolManifest,
        win_rates: Mapping[str, float],
        payoff_hash: str,
        master_seed: int = 20260721,
        script_weights: Mapping[str, float] | None = None,
    ) -> None:
        self._cases = tuple(cases)
        self._scripts = tuple(sorted(scripts, key=lambda spec: spec.opponent_id))
        self._pool = pool
        self._payoff_hash = payoff_hash
        self._master_seed = master_seed
        self._validate_cases()
        if not self._scripts or any(spec.kind != "script" for spec in self._scripts):
            raise ValueError("League schedule requires script opponent specs")
        self._script_weights = self._validate_script_weights(script_weights)
        scenario_hash = pool.entries[0].scenario_hash
        if any(spec.scenario_hash != scenario_hash for spec in self._scripts):
            raise ValueError("League opponent scenario hashes do not match")
        self._history = tuple(
            OpponentSpec(
                opponent_id=entry.policy_id,
                kind="checkpoint",
                checkpoint=entry.checkpoint,
                checkpoint_sha256=entry.checkpoint_sha256,
                scenario_hash=entry.scenario_hash,
                selection_evidence=entry.validation_report,
            )
            for entry in pool.entries
        )
        specs = {spec.opponent_id: spec for spec in (*self._scripts, *self._history)}
        if len(specs) != len(self._scripts) + len(self._history):
            raise ValueError("League opponent IDs must be unique")
        self._opponent_specs = MappingProxyType(specs)
        expected_ids = {spec.opponent_id for spec in self._history}
        if len(self._history) >= 2 and set(win_rates) != expected_ids:
            raise ValueError("PFSP payoff data is missing or stale")
        if len(self._history) >= 2 and _SHA256_PATTERN.fullmatch(payoff_hash) is None:
            raise ValueError("PFSP payoff hash is invalid")
        self._pfsp = (
            pfsp_probabilities(win_rates) if len(self._history) >= 2 else {}
        )

    @property
    def opponent_specs(self) -> Mapping[str, OpponentSpec]:
        return self._opponent_specs

    @property
    def pfsp_probabilities(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self._pfsp))

    def _validate_cases(self) -> None:
        if not self._cases or len(self._cases) % 2:
            raise ValueError("League schedule requires complete paired cases")
        for host, opponent in zip(self._cases[::2], self._cases[1::2], strict=True):
            if (
                host.split != "train"
                or opponent.split != "train"
                or host.pair_index != opponent.pair_index
                or host.seed != opponent.seed
                or (host.learner_side, opponent.learner_side) != ("host", "opponent")
            ):
                raise ValueError("League schedule cases must be train side-swapped pairs")

    def _uniform(self, pair_slot: int, stream: str) -> float:
        return stable_uniform(
            master_seed=self._master_seed,
            pair_slot=pair_slot,
            pool_hash=self._pool.manifest_sha256,
            payoff_hash=self._payoff_hash,
            stream=stream,
        )

    @staticmethod
    def _choose_uniform(
        specs: Sequence[OpponentSpec], draw: float
    ) -> OpponentSpec:
        index = min(int(draw * len(specs)), len(specs) - 1)
        return specs[index]

    def _validate_script_weights(
        self, weights: Mapping[str, float] | None
    ) -> tuple[float, ...] | None:
        if weights is None:
            return None
        expected = {spec.opponent_id for spec in self._scripts}
        if set(weights) != expected or any(
            isinstance(value, bool) or float(value) <= 0.0
            for value in weights.values()
        ):
            raise ValueError("Script weights must positively cover every script")
        total = sum(float(weights[spec.opponent_id]) for spec in self._scripts)
        return tuple(float(weights[spec.opponent_id]) / total for spec in self._scripts)

    def _choose_script(self, draw: float) -> OpponentSpec:
        if self._script_weights is None:
            return self._choose_uniform(self._scripts, draw)
        cumulative = 0.0
        for spec, probability in zip(
            self._scripts, self._script_weights, strict=True
        ):
            cumulative += probability
            if draw < cumulative:
                return spec
        return self._scripts[-1]

    def _choose_pfsp(self, draw: float) -> OpponentSpec:
        cumulative = 0.0
        by_id = {spec.opponent_id: spec for spec in self._history}
        for policy_id, probability in self._pfsp.items():
            cumulative += probability
            if draw < cumulative:
                return by_id[policy_id]
        return by_id[next(reversed(self._pfsp))]

    def assignments(
        self, pair_slot: int
    ) -> tuple[LeagueEpisodeAssignment, LeagueEpisodeAssignment]:
        if pair_slot < 0:
            raise ValueError("pair_slot must be nonnegative")
        case_offset = pair_slot % (len(self._cases) // 2) * 2
        paired_cases = self._cases[case_offset : case_offset + 2]
        source_draw = self._uniform(pair_slot, "source")
        if len(self._history) < 2 or source_draw < 0.40:
            source: Literal["script", "pfsp", "uniform_history"] = "script"
            source_probability = 1.0 if len(self._history) < 2 else 0.40
            selected = self._choose_script(
                self._uniform(pair_slot, "opponent:script")
            )
        elif source_draw < 0.90:
            source = "pfsp"
            source_probability = 0.50
            selected = self._choose_pfsp(self._uniform(pair_slot, "opponent:pfsp"))
        else:
            source = "uniform_history"
            source_probability = 0.10
            selected = self._choose_uniform(
                self._history, self._uniform(pair_slot, "opponent:uniform_history")
            )
        return tuple(
            LeagueEpisodeAssignment(
                pair_slot=pair_slot,
                case=case,
                opponent=selected,
                source=source,
                sampling_probability=source_probability,
            )
            for case in paired_cases
        )  # type: ignore[return-value]
