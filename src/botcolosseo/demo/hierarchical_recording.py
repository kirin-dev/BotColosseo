"""Observation-only recording wrapper: never supplies telemetry to either actor."""

import cv2
import numpy as np

from botcolosseo.demo.extraction_showcase import compose_extraction_showcase_frame
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_protocol import ExtractionEventType
from botcolosseo.training.hierarchical_protocol import Command


def viewer_event_label(events, *, side):
    """Do not attribute opponent loot or ambiguous shared-ledger amounts to self."""
    labels = []
    own_labels = {
        ExtractionEventType.LOOT_PICKUP: "LOOT PICKED UP",
        ExtractionEventType.LOOT_DROP: "BACKPACK ITEM REPLACED",
        ExtractionEventType.CACHE_LOOTED: "CORPSE CACHE LOOTED",
        ExtractionEventType.EXTRACTION_STARTED: "EXTRACTION STARTED - HOLD 3s",
        ExtractionEventType.EXTRACTION_INTERRUPTED: "EXTRACTION INTERRUPTED",
        ExtractionEventType.EXTRACTED: "VALUE BANKED - EXTRACTION COMPLETE",
    }
    for event in events:
        if event.type == ExtractionEventType.VALID_HIT:
            labels.append("HIT CONFIRMED" if event.side == side else "TAKING FIRE")
        elif event.type == ExtractionEventType.DEATH:
            labels.append("ELIMINATED - LOOT LOST" if event.side == side else "ENEMY DOWN - CACHE")
        elif event.side == side and event.type in own_labels:
            labels.append(own_labels[event.type])
    return " | ".join(dict.fromkeys(labels))


class StrategicRecordingEnv:
    def __init__(self, env, controller, *, side, label):
        self.env, self.controller, self.side, self.label = env, controller, side, label
        self.frames, self.events, self.command_changes = [], [], []
        self.finished = False
        self.previous_command = None
        self.event_text = ""
        self.event_age = 100

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, host, opponent):
        result = self.env.step(host, opponent)
        if self.finished:
            return result  # Global settlement continues without idle video padding.
        command = self.controller.command
        time_seconds = (len(self.frames) + 1) * 4 / 35
        if command != self.previous_command:
            self.command_changes.append(
                {"seconds": len(self.frames) * 4 / 35, "command": command.name}
            )
            self.previous_command = command
        event = viewer_event_label(result.events, side=self.side)
        if event:
            self.event_text, self.event_age = event, 0
            self.events.append({"seconds": time_seconds, "text": event})
        else:
            self.event_age += 1
        observation = getattr(result, self.side)
        frame = compose_extraction_showcase_frame(
            observation,
            privileged=self.env.privileged_state(),
            protocol=self.env.protocol_snapshot(),
            learner_side=self.side,
            style=self.label,
            action=MacroAction(host if self.side == "host" else opponent),
            event_label="",  # Draw wrapped events below; legacy single-line banner overlaps.
        )
        # Reuse verified health/ammo/loot telemetry; add an explicit strategic layer.
        canvas = np.full((440, 640, 3), 14, dtype=np.uint8)
        canvas[:360] = frame
        canvas[326:366] = 14
        if self.event_age < 18:
            lines = [""]
            for word in self.event_text.split():
                trial = (lines[-1] + " " + word).strip()
                width = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
                if width > 620:
                    lines.append(word)
                else:
                    lines[-1] = trial
            for index, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line,
                    (10, 338 + 12 * index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (80, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
        canvas[:27, :490] = 14
        cv2.putText(
            canvas,
            "BotColosseo | SEARCH - FIGHT - EXTRACT",
            (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"HIGH-LEVEL COMMAND: {Command(command).name}",
            (10, 382),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "APPLIED STYLE A/D/E: "
            + "/".join(f"{value:.1f}" for value in self.controller.applied_condition.as_tuple()[:3])
            + f" | LOW DIFFICULTY: {self.controller.applied_low_difficulty:.2f}",
            (10, 403),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "REQUESTED A/D/E: "
            + "/".join(
                f"{value:.1f}" for value in self.controller.requested_condition.as_tuple()[:3]
            )
            + f" | DIFFICULTY: {self.controller.requested_condition.difficulty:.2f}",
            (10, 428),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        self.frames.append(canvas)
        self.finished = (
            observation.health <= 0
            or observation.banked_value > 0
            or result.terminated
            or result.truncated
        )
        return result
