import pytest

from botcolosseo.cli.evaluate_hierarchical_executor import evaluate_case, scheduled_command
from botcolosseo.training.hierarchical_protocol import Command


def test_extraction_oracle_rejects_mixed_action_provenance():
    with pytest.raises(ValueError, match="cannot mix"):
        evaluate_case(
            None, 96, "host", "cpu", expected_scenario="unused",
            privileged_oracle=True, oracle_extraction_only=True,
        )


def test_endpoint_intervention_preserves_entire_search_prefix():
    for t in range(240):
        assert scheduled_command(t, 97, extraction_endpoint="north") == scheduled_command(
            t, 97, extraction_endpoint="south"
        )
    assert scheduled_command(240, 97, extraction_endpoint="north") == Command.EXTRACT_NORTH
    assert scheduled_command(240, 96, extraction_endpoint="south") == Command.EXTRACT_SOUTH
    with pytest.raises(ValueError):
        scheduled_command(0, 96, extraction_endpoint="unknown")


def test_public_fixed_schedule_boundaries():
    assert scheduled_command(0, 96) == Command.SEARCH_NORTH
    assert scheduled_command(80, 96) == Command.SEARCH_CENTER
    assert scheduled_command(160, 96) == Command.SEARCH_SOUTH
    assert scheduled_command(240, 96) == Command.EXTRACT_NORTH
    assert scheduled_command(240, 97) == Command.EXTRACT_SOUTH
    assert scheduled_command(47, 96, "combat") == Command.ENGAGE
    assert scheduled_command(48, 96, "combat") == Command.DISENGAGE
    assert scheduled_command(128, 96, "combat") == Command.SEARCH_CENTER
    with pytest.raises(ValueError):
        scheduled_command(0, 96, "unknown")
