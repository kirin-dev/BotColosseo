"""Recurrent snapshot distillation into the existing difficulty FiLM modules."""

import torch
from torch import nn


def freeze_for_difficulty(actor):
    for name, parameter in actor.named_parameters():
        parameter.requires_grad_(
            name.startswith(("difficulty_input.network.", "difficulty_output.network."))
        )
    return [p for p in actor.parameters() if p.requires_grad]


def distill_episode(actor, teachers, batch, optimizer=None, *, chunk_size=64):
    """Replay identical fair histories; each teacher retains its own hidden state.

    Three static conditions are an initial curriculum, not live-switch training.
    Teacher inference uses Hard, where each source checkpoint was evaluated.
    """
    if len(teachers) != 3 or chunk_size <= 0:
        raise ValueError("Expected three teachers and a positive chunk size")
    keys = ("frames", "scalars", "previous_actions", "masks", "commands", "difficulty")
    count = batch["commands"].shape[1]
    if count == 0:
        raise ValueError("Empty trajectory")
    student_hidden = [None] * 3
    teacher_hidden = [None] * 3
    losses, agreements = [0.0] * 3, [0] * 3
    actor.train(optimizer is not None)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    for start in range(0, count, chunk_size):
        part = {k: batch[k][:, start : start + chunk_size] for k in keys}
        for index, difficulty in enumerate((0.0, 0.5, 1.0)):
            with torch.no_grad():
                target = teachers[index](
                    **{**part, "difficulty": torch.ones_like(part["difficulty"])},
                    hidden=teacher_hidden[index],
                )
                teacher_hidden[index] = target.hidden
                probabilities = target.logits.softmax(-1)
            with torch.set_grad_enabled(optimizer is not None):
                output = actor(
                    **{**part, "difficulty": torch.full_like(part["difficulty"], difficulty)},
                    hidden=student_hidden[index],
                )
                # All source-policy decisions are valid soft labels, including waits.
                loss = nn.functional.kl_div(
                    output.logits.log_softmax(-1), probabilities, reduction="sum"
                ) / (count * batch["commands"].shape[0])
                if optimizer is not None:
                    (loss * (2.0 if index == 2 else 1.0) / 4).backward()
            losses[index] += float(loss.detach())
            agreements[index] += int((output.logits.argmax(-1) == target.logits.argmax(-1)).sum())
            student_hidden[index] = output.hidden.detach()
    if optimizer is not None:
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    return {
        "kl": losses,
        "agreement": [n / (count * batch["commands"].shape[0]) for n in agreements],
    }
