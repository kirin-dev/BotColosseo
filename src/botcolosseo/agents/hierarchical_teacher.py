"""Training-only command oracle. Never use its status as a deployed actor input."""

from __future__ import annotations

import math
from dataclasses import dataclass

from botcolosseo.agents.extraction_teachers import (
    opponent_health,
    opponent_position,
    player_health,
    player_pose,
    player_slots,
    steer_toward,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_layouts import randomized_loot_layout
from botcolosseo.envs.extraction_protocol import ExtractionProtocolSnapshot
from botcolosseo.envs.extraction_types import ExtractionPrivilegedState
from botcolosseo.training.hierarchical_protocol import Command, search_band


@dataclass(frozen=True)
class OracleDecision:
    action: MacroAction
    status: str  # executing, complete, unavailable, inactive
    target: tuple[float, float] | None = None


class PrivilegedCommandTeacher:
    """D0 upper-bound probe and D1 action labels, not a fair policy.

    Search commands seek one useful pickup in a horizontal map band. Completion
    requires increased carried value, not simply reaching a coordinate. Extraction
    completes only after banked value increases. All state is reset per episode.
    """

    def __init__(self, *, side: str, layout_variant: int) -> None:
        if side not in {"host", "opponent"}:
            raise ValueError("Invalid player side")
        self.side = side
        self.layout = randomized_loot_layout(layout_variant)
        self.reset()

    def reset(self) -> None:
        self.command: Command | None = None
        self.initial_carried = 0
        self.initial_banked = 0
        self.initial_enemy_alive = False
        self.search_completed = False

    def begin(self, command: Command, state: ExtractionPrivilegedState) -> None:
        self.command = Command(command)
        self.initial_carried = sum(player_slots(state, self.side))
        self.initial_banked = getattr(state, f"{self.side}_banked")
        self.initial_enemy_alive = opponent_health(state, self.side) > 0
        self.search_completed = False

    def observe_transition(
        self, before: ExtractionProtocolSnapshot, after: ExtractionProtocolSnapshot
    ) -> None:
        """Credit only an unambiguous own world pickup in the commanded band.

        The ledger retains only the last loot ID. Multiple pickups in one macro
        step are conservatively not attributed; counters alone cannot locate them.
        """
        if self.command not in (Command.SEARCH_NORTH, Command.SEARCH_CENTER, Command.SEARCH_SOUTH):
            return
        total = (
            after.host_loot_pickups
            + after.opponent_loot_pickups
            - before.host_loot_pickups
            - before.opponent_loot_pickups
        )
        own = getattr(after, f"{self.side}_loot_pickups") - getattr(
            before, f"{self.side}_loot_pickups"
        )
        index = after.last_loot_id - 1  # Engine IDs are one-based; zero denotes cache.
        if total != 1 or own != 1 or after.last_loot_side != (1 if self.side == "host" else 2):
            return
        if not 0 <= index < len(self.layout):
            return
        if not before.world_loot_mask & (1 << index) or after.world_loot_mask & (1 << index):
            return
        y = self.layout[index][2]
        band = search_band(y)
        self.search_completed |= band == self.command

    def act(self, state: ExtractionPrivilegedState) -> OracleDecision:
        command = self.command
        if command is None:
            raise ValueError("begin command before requesting labels")
        banked = getattr(state, f"{self.side}_banked")
        if command in (Command.EXTRACT_NORTH, Command.EXTRACT_SOUTH):
            if banked > self.initial_banked:
                x, y, _ = player_pose(state, self.side)
                signed_y = y if command == Command.EXTRACT_NORTH else -y
                correct_zone = -128 <= x <= 128 and 352 <= signed_y <= 480
                return OracleDecision(MacroAction.IDLE, "complete" if correct_zone else "inactive")
        if player_health(state, self.side) <= 0 or banked > 0:
            return OracleDecision(MacroAction.IDLE, "inactive")
        x, y, _ = player_pose(state, self.side)
        if command.value <= Command.SEARCH_SOUTH.value:
            if self.search_completed:
                return OracleDecision(MacroAction.IDLE, "complete")
            minimum = min(player_slots(state, self.side))
            targets = []
            for index, (value, tx, ty) in enumerate(self.layout):
                band = search_band(ty)
                if band == command and value > minimum and state.world_loot_mask & (1 << index):
                    targets.append((tx, ty))
            if not targets:
                return OracleDecision(MacroAction.IDLE, "unavailable")
            target = min(targets, key=lambda point: math.dist((x, y), point))
        elif command == Command.ENGAGE:
            if opponent_health(state, self.side) <= 0:
                return OracleDecision(
                    MacroAction.IDLE,
                    "complete" if self.initial_enemy_alive else "unavailable",
                )
            target = opponent_position(state, self.side)
            return OracleDecision(
                steer_toward(state, self.side, target, attack=math.dist((x, y), target) <= 512),
                "executing",
                target,
            )
        elif command == Command.DISENGAGE:
            enemy = opponent_position(state, self.side)
            if opponent_health(state, self.side) <= 0 or math.dist((x, y), enemy) >= 512:
                return OracleDecision(MacroAction.IDLE, "complete")
            # Safe arena anchors, not an unbounded vector pointing outside the map.
            anchors = ((-520.0, 288.0), (-520.0, -288.0), (520.0, 288.0), (520.0, -288.0))
            target = max(
                anchors, key=lambda point: math.dist(point, enemy) - 0.5 * math.dist(point, (x, y))
            )
            # Retreat without first rotating toward a point behind us. The
            # action space already supports backwards and lateral translation.
            _, _, angle = player_pose(state, self.side)
            desired = math.degrees(math.atan2(target[1] - y, target[0] - x))
            error = (desired - angle + 180) % 360 - 180
            if abs(error) >= 135:
                action = MacroAction.MOVE_BACKWARD
            elif error >= 45:
                action = MacroAction.STRAFE_LEFT
            elif error <= -45:
                action = MacroAction.STRAFE_RIGHT
            else:
                action = MacroAction.MOVE_FORWARD
            return OracleDecision(action, "executing", target)
        else:
            target = (0.0, 400.0 if command == Command.EXTRACT_NORTH else -400.0)
            if math.dist((x, y), target) <= 28:
                return OracleDecision(MacroAction.IDLE, "executing", target)
        return OracleDecision(steer_toward(state, self.side, target), "executing", target)


def counterfactual_command_labels(
    state: ExtractionPrivilegedState, *, side: str, layout_variant: int
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """One-step fresh-command oracle labels; hidden world state is supervision only.

    No artificial separation: identical valid actions across commands are allowed.
    Fresh inactive/completed/unavailable commands have no action-imitation target.
    """
    labels = []
    valid = []
    for command in Command:
        oracle = PrivilegedCommandTeacher(side=side, layout_variant=layout_variant)
        oracle.begin(command, state)
        decision = oracle.act(state)
        labels.append(int(decision.action))
        valid.append(decision.status == "executing")
    return tuple(labels), tuple(valid)
