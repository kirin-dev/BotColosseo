import pytest

from botcolosseo.agents.hierarchical_seed import SEED_PROFILES, seed_command
from botcolosseo.training.hierarchical_protocol import Command


def test_seed_preferences_and_public_contact():
    public = [1.0, 0.75, 30 / 150, 2 / 3, 0, 0, 0, 0, 0.8]
    assert seed_command(public, profile="preserve", previous_health=1) == Command.EXTRACT_NORTH
    assert seed_command(public, profile="upgrade", previous_health=1) == Command.SEARCH_CENTER
    public[0] = 0.8
    assert seed_command(public, profile="selective_combat", previous_health=1) == Command.ENGAGE
    public[1] = 0.1
    assert seed_command(public, profile="selective_combat", previous_health=1) == Command.DISENGAGE


def test_all_seeds_can_extract_without_combat_and_keep_endpoint():
    public = [1, 0.75, 30 / 150, 2 / 3, 0, 0, 1, 0, 0.2]
    for profile in SEED_PROFILES:
        assert seed_command(
            public, profile=profile, previous_health=1, previous_command=Command.EXTRACT_NORTH
        ) == Command.EXTRACT_NORTH
    with pytest.raises(ValueError):
        seed_command(public, profile="unknown", previous_health=1)
