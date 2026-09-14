"""Public-state command teachers for interpretable population initialization."""

import math

from botcolosseo.training.hierarchical_protocol import Command

SEED_PROFILES = ("preserve", "upgrade", "selective_combat")


def seed_command(
    scalars, *, profile: str, previous_health: float, previous_command=Command.SEARCH_CENTER
) -> Command:
    """Labels for high-level distillation, not privileged deployment routing.

    Scalar order follows extraction_scalars. The time-based corridor rotation
    is a public initialization heuristic, not knowledge of remaining world loot.
    Both exits remain legal regardless of combat outcome.
    """
    if profile not in SEED_PROFILES or len(scalars) != 9:
        raise ValueError("Unknown seed profile or invalid public scalar shape")
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("Nonfinite public state")
    health, ammo, carried, free, minimum, _, opened, _, remaining = map(float, scalars)
    elapsed_seconds = (1 - remaining) * 75
    corridor = Command(int(max(0, elapsed_seconds) // 9) % 3)
    endpoint = (
        Command.EXTRACT_NORTH
        if int(max(0, elapsed_seconds) // 18) % 2 == 0 else Command.EXTRACT_SOUTH
    )
    if previous_command in (Command.EXTRACT_NORTH, Command.EXTRACT_SOUTH):
        endpoint = Command(previous_command)
    # Preserve an ongoing extraction whenever its public progress is positive.
    if scalars[7] > 0:
        return endpoint
    if carried > 0 and (health <= 0.4 or remaining <= 0.24):
        return endpoint
    if profile == "preserve" and carried >= 20 / 150:
        return endpoint
    if profile == "upgrade" and carried > 0 and free == 0 and minimum >= 0.6:
        return endpoint
    if profile == "selective_combat":
        under_attack = health < previous_health - 0.001
        if under_attack:
            return Command.ENGAGE if health >= 0.6 and ammo >= 0.25 else Command.DISENGAGE
        if carried >= 60 / 150 and opened:
            return endpoint
    if profile == "upgrade" and carried >= 90 / 150 and opened:
        return endpoint
    return corridor
