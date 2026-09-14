"""Immutable role populations and version-scoped empirical payoff records."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from botcolosseo.training.hierarchical_protocol import GameIdentity, own_payoffs


@dataclass(frozen=True)
class PopulationWindow:
    game: GameIdentity
    host_policies: tuple[str, ...]
    opponent_policies: tuple[str, ...]

    def __post_init__(self) -> None:
        for policies in (self.host_policies, self.opponent_policies):
            if (
                not 1 <= len(policies) <= 6
                or len(set(policies)) != len(policies)
                or not all(policies)
            ):
                raise ValueError("Each role needs one to six unique policy hashes")

    def with_response(self, role: str, policy_hash: str) -> PopulationWindow:
        if role not in {"host", "opponent"}:
            raise ValueError("Invalid policy role")
        field = f"{role}_policies"
        return replace(self, **{field: getattr(self, field) + (policy_hash,)})

    def next_executor(self, executor_hash: str) -> PopulationWindow:
        """Creates a new identity; does not itself authorize a promotion."""
        if executor_hash == self.game.executor_hash:
            raise ValueError("Executor upgrade requires a new hash")
        return replace(self, game=replace(self.game, executor_hash=executor_hash))


@dataclass(frozen=True)
class PayoffCase:
    game: GameIdentity
    host_policy: str
    opponent_policy: str
    seed: int
    layout: int
    repeat: int
    host_banked: float
    opponent_banked: float
    globally_settled: bool


def payoff_matrices(
    window: PopulationWindow,
    cases: list[PayoffCase],
    *,
    required_cases: frozenset[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Require the same declared (seed,layout,repeat) grid in every ordered cell.

    Case repeats remain repeats, not independent layouts. Role-swapped observations
    must be explicitly recorded under the swapped policy hashes, never inferred.
    """
    if not required_cases:
        raise ValueError("An explicit evaluation case grid is required")
    cells = {(h, o): {} for h in window.host_policies for o in window.opponent_policies}
    for case in cases:
        window.game.require_same_game(case.game)
        key = (case.host_policy, case.opponent_policy)
        case_id = (case.seed, case.layout, case.repeat)
        if key not in cells or case_id not in required_cases:
            raise ValueError("Unexpected policy pair or evaluation case")
        if case_id in cells[key]:
            raise ValueError("Duplicate payoff case")
        cells[key][case_id] = own_payoffs(
            case.host_banked, case.opponent_banked, globally_settled=case.globally_settled
        )
    shape = (len(window.host_policies), len(window.opponent_policies))
    a, b = np.zeros(shape), np.zeros(shape)
    for i, host in enumerate(window.host_policies):
        for j, opponent in enumerate(window.opponent_policies):
            cell = cells[(host, opponent)]
            if cell.keys() != required_cases:
                raise ValueError("Incomplete full cross-play matrix")
            a[i, j], b[i, j] = np.asarray([cell[key] for key in sorted(required_cases)]).mean(
                axis=0
            )
    return a, b
