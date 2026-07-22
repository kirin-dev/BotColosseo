from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from botcolosseo.envs.duel_types import DuelActorObservation


def compose_learner_frame(
    observation: DuelActorObservation,
    *,
    policy_label: str,
    event_label: str,
) -> NDArray[np.uint8]:
    if not policy_label or not event_label:
        raise ValueError("Showcase overlay labels must be non-empty")
    view = cv2.resize(observation.frame, (256, 252), interpolation=cv2.INTER_NEAREST)
    view = cv2.cvtColor(view, cv2.COLOR_GRAY2RGB)
    canvas = np.zeros((300, 256, 3), dtype=np.uint8)
    canvas[48:] = view
    cv2.putText(
        canvas,
        policy_label,
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    state = (
        f"score={observation.own_score}-{observation.opponent_score} "
        f"core={int(observation.has_core)} event={event_label}"
    )
    cv2.putText(
        canvas,
        state,
        (6, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def compose_showcase_comparison(
    streams: Sequence[tuple[str, Sequence[NDArray[np.uint8]]]],
    *,
    subtitle: str,
) -> tuple[NDArray[np.uint8], ...]:
    if len(streams) < 2 or any(not label or not frames for label, frames in streams):
        raise ValueError("Showcase comparison contains an empty stream")
    shape = np.asarray(streams[0][1][0]).shape
    if shape != (300, 256, 3) or any(
        np.asarray(frame).shape != shape
        for _, frames in streams
        for frame in frames
    ):
        raise ValueError("Showcase comparison streams have incompatible geometry")
    result = []
    for index in range(max(len(frames) for _, frames in streams)):
        panels = [
            np.array(frames[min(index, len(frames) - 1)], copy=True)
            for _, frames in streams
        ]
        comparison = np.concatenate(panels, axis=1)
        canvas = np.zeros((332, comparison.shape[1], 3), dtype=np.uint8)
        canvas[32:] = comparison
        cv2.putText(
            canvas,
            subtitle,
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        result.append(canvas)
    return tuple(result)
