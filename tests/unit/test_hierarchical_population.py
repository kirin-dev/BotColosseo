from dataclasses import replace

import pytest

from botcolosseo.training.hierarchical_population import (
    PayoffCase,
    PopulationWindow,
    payoff_matrices,
)
from botcolosseo.training.hierarchical_protocol import ControlCondition, GameIdentity


def test_version_upgrade_rejects_all_old_payoffs():
    game = GameIdentity("scenario", "rules", "executor1", ControlCondition(), ControlCondition())
    window = PopulationWindow(game, ("h",), ("o",))
    case = PayoffCase(game, "h", "o", 1, 1, 0, 150, 75, True)
    grid = frozenset({(1, 1, 0)})
    a, b = payoff_matrices(window, [case], required_cases=grid)
    assert a.item() == 1 and b.item() == 0.5
    with pytest.raises(ValueError, match="Stale"):
        payoff_matrices(window.next_executor("executor2"), [case], required_cases=grid)
    with pytest.raises(ValueError, match="Incomplete"):
        payoff_matrices(window.with_response("host", "new"), [case], required_cases=grid)
    with pytest.raises(ValueError, match="Duplicate"):
        payoff_matrices(window, [case, case], required_cases=grid)
    with pytest.raises(ValueError, match="Both players"):
        payoff_matrices(window, [replace(case, globally_settled=False)], required_cases=grid)
