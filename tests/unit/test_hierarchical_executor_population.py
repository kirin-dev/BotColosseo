import pytest
import torch

from botcolosseo.agents.hierarchical_controller import HierarchicalController
from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.evaluation.hierarchical_executor import decide_executor_promotion
from botcolosseo.training.hierarchical_executor_population import (
    FrozenCommandSelector,
    upgrade_pair,
)
from botcolosseo.training.hierarchical_protocol import ControlCondition


def test_upgrade_commands_match_deployment_on_identical_fair_history():
    torch.manual_seed(19)
    low = CommandExecutor()
    high = StrategicActor().requires_grad_(False)
    deployed = HierarchicalController(low, high, seed=71)
    selector = FrozenCommandSelector(high, seed=71)
    for decision in range(25):
        scalars = torch.rand(1, 1, 9)
        event = decision in (1, 11)
        command = selector.select(deployed.low_hidden, scalars, public_event=event)
        result = deployed.step(
            torch.zeros(1, 1, 1, 84, 84, dtype=torch.uint8),
            scalars,
            torch.zeros(1, 1, dtype=torch.long),
            ControlCondition(),
            public_event=event,
        )
        assert command == result["command"]
        torch.testing.assert_close(selector.hidden, deployed.high_hidden)
        assert selector.elapsed == deployed.elapsed


def test_population_upgrade_sampling_balances_source_for_each_learner_role():
    solution = {"host": [0, 1, 0], "opponent": [1, 0]}
    counts = {(role, source): 0 for role in ("host", "opponent") for source in ("uniform", "meta")}
    for episode in range(40):
        source, pair = upgrade_pair(solution, episode)
        counts[("host" if episode % 2 == 0 else "opponent", source)] += 1
        assert 0 <= pair["host"] < 3 and 0 <= pair["opponent"] < 2
        if source == "meta":
            assert pair == {"host": 1, "opponent": 0}
        assert (source, pair) == upgrade_pair(solution, episode)
    assert set(counts.values()) == {10}
    with pytest.raises(ValueError):
        upgrade_pair({"host": [float("nan")], "opponent": [1]}, 0)


def comparison(**overrides):
    commands = {
        "SEARCH_NORTH": {"baseline_rate": .5, "candidate_rate": .6, "rate_delta": .1},
        "ENGAGE": {"baseline_rate": .5, "candidate_rate": .5, "rate_delta": 0},
    }
    value = {"commands": commands, "extraction_rate_delta": 0.0}
    value.update(overrides)
    return value


def test_promotion_requires_retention_and_named_weak_skill_gain():
    decision = decide_executor_promotion(comparison(), weak_commands=("SEARCH_NORTH",))
    assert decision["promote"]
    assert not decide_executor_promotion(
        comparison(extraction_rate_delta=-.051), weak_commands=("SEARCH_NORTH",)
    )["promote"]
    assert not decide_executor_promotion(comparison(), weak_commands=("ENGAGE",))["promote"]


def test_promotion_rejects_unmeasured_weak_command_and_regression():
    missing = decide_executor_promotion(comparison(), weak_commands=("DISENGAGE",))
    assert missing == {
        "promote": False,
        "reason": "missing_comparable_weak_command",
        "commands": ["DISENGAGE"],
    }
    bad = comparison(
        commands={"SEARCH_NORTH": {"baseline_rate": .8, "candidate_rate": .7, "rate_delta": -.1}}
    )
    assert (
        decide_executor_promotion(bad, weak_commands=("SEARCH_NORTH",))["reason"]
        == "command_regression"
    )
