"""Explicit soft-target initialization for continuous and switched style controls."""

import torch


def style_training_tracks(length, *, device, compositions=False):
    if length <= 0:
        raise ValueError("A style track needs at least one high decision")
    endpoints = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device=device
    )
    tracks = endpoints[:, None].expand(-1, length, -1).clone()
    if not compositions:
        return tracks
    pairs = torch.tensor([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.float32, device=device)[
        :, None
    ].expand(-1, length, -1)
    interpolation = endpoints[1:, None].expand(-1, length, -1) * 0.5
    # Three tracks cover every endpoint and both transition directions. Segments
    # vary from 4 to 8 high decisions, without resetting recurrent state.
    switches = []
    for offset in range(3):
        segments, index = [], offset
        while len(segments) < length:
            segments.extend([endpoints[index % 4]] * (4 + index % 5))
            index += -1 if offset == 1 else 1
        switches.append(torch.stack(segments[:length]))
    return torch.cat((tracks, pairs, interpolation, torch.stack(switches)))


def mix_style_teachers(teacher_probabilities, styles):
    """Convex teacher blending is a warm start, not a semantic control guarantee."""
    if teacher_probabilities.shape[0] != 4 or styles.shape[-1] != 3:
        raise ValueError("Need Neutral/A/D/E teachers and three control axes")
    if not bool(torch.isfinite(styles).all() and ((styles >= 0) & (styles <= 1)).all()):
        raise ValueError("Style conditions must be finite and within [0,1]")
    neutral = (1 - styles.sum(-1, keepdim=True)).clamp_min(0)
    weights = torch.cat((neutral, styles), -1)
    weights = weights / weights.sum(-1, keepdim=True)
    return torch.einsum("btk,kta->bta", weights, teacher_probabilities)
