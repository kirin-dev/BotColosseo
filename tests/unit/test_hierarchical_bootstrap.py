from botcolosseo.evaluation.hierarchical_bootstrap import bootstrap_meta_strategy
from botcolosseo.training.hierarchical_population import PayoffCase, PopulationWindow
from botcolosseo.training.hierarchical_protocol import ControlCondition, GameIdentity


def test_cluster_bootstrap_preserves_repeats_and_role_payoffs():
    game = GameIdentity("s", "r", "e", ControlCondition(), ControlCondition())
    window = PopulationWindow(game, ("h",), ("o",))
    grid = frozenset((seed, seed, repeat) for seed in (1, 2) for repeat in (0, 1))
    cases = [
        PayoffCase(game, "h", "o", seed, layout, repeat, 150, 75, True)
        for seed, layout, repeat in grid
    ]
    result = bootstrap_meta_strategy(window, cases, required_cases=grid, draws=3)
    assert result["clusters"] == 2
    assert result["host_weight_quantiles"] == [[1.0], [1.0], [1.0]]
    assert result["maximum_sample_regret"] == 0
