"""Offline soft-policy distillation into one style-conditioned strategic actor."""

import argparse
import json
from pathlib import Path

import torch

from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.data.hierarchical_demonstrations import episode_tensors, load_command_episode
from botcolosseo.training.hierarchical_style_targets import (
    mix_style_teachers,
    style_training_tracks,
)


def main():
    parser = argparse.ArgumentParser()
    for name in ("executor", "neutral", "aggressive", "defensive", "explorer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--compositions", action="store_true")
    parser.add_argument("--combat-opportunities", action="store_true")
    parser.add_argument("--proactive-combat", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.epochs <= 0:
        raise ValueError("Fresh output and positive epochs required")
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    payload = torch.load(args.executor, map_location="cpu", weights_only=False)
    executor = CommandExecutor().to(args.device).eval()
    executor.load_state_dict(payload["actor"])
    executor.requires_grad_(False)
    teachers = []
    names = ("neutral", "aggressive", "defensive", "explorer")
    for name in names:
        item = torch.load(getattr(args, name), map_location="cpu", weights_only=False)
        if item["identity"]["executor"] != digest(args.executor):
            raise ValueError("Teachers must share the frozen executor")
        actor = StrategicActor().to(args.device).eval()
        actor.load_state_dict(item["actor"])
        actor.requires_grad_(False)
        teachers.append(actor)
    student = StrategicActor().to(args.device)
    student.load_state_dict(teachers[0].state_dict())
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    episodes = []
    opportunity_count = 0
    for filename, expected in payload["identity"]["train"].items():
        path = Path(filename)
        if digest(path) != expected:
            raise ValueError("Recorded trajectory hash mismatch")
        raw = load_command_episode(path)
        batch = episode_tensors(raw, device=args.device)
        with torch.no_grad():
            output = executor(
                **{
                    key: batch[key]
                    for key in (
                        "frames",
                        "scalars",
                        "previous_actions",
                        "masks",
                        "commands",
                        "difficulty",
                    )
                }
            )
        features = torch.cat((torch.zeros_like(output.features[:, :1]), output.features[:, :-1]), 1)
        previous = torch.cat(
            (torch.ones_like(batch["commands"][:, :1]), batch["commands"][:, :-1]), 1
        )
        scalars = batch["scalars"][:, ::8]
        length = scalars.shape[1]
        masks = torch.ones(1, length, device=args.device)
        masks[:, 0] = 0
        fair = dict(
            features=features[:, ::8],
            scalars=scalars,
            previous_commands=previous[:, ::8],
            masks=masks,
            elapsed=(8 * masks)[..., None],
            style=torch.zeros(1, length, 3, device=args.device),
            difficulty=torch.ones(1, length, 1, device=args.device),
        )
        with torch.no_grad():
            targets = torch.cat([teacher(**fair).logits.softmax(-1) for teacher in teachers])
        opportunities = torch.zeros(length, dtype=torch.bool, device=args.device)
        if args.combat_opportunities and "counterfactual_actions" in raw:
            # Training-only firing labels imply enemy alive and distance <=512,
            # not visibility or line of sight. They never enter actor inputs.
            oracle_actions = torch.as_tensor(
                raw["counterfactual_actions"][::8, 3], device=args.device
            )
            oracle_valid = torch.as_tensor(raw["counterfactual_valid"][::8, 3], device=args.device)
            opportunities = (
                oracle_valid
                & (oracle_actions >= 9)
                & (scalars[0, :, 0] >= 0.6)
                & (scalars[0, :, 1] >= 0.25)
                & (scalars[0, :, 5] == 0)
                & (scalars[0, :, 7] == 0)
            )
        if args.proactive_combat:
            # Public-clock initiation prior: seek combat for the opening 16s
            # while healthy and armed, then retain task-oriented teacher targets.
            opportunities |= (
                (scalars[0, :, 8] >= 59 / 75)
                & (scalars[0, :, 0] >= 0.6)
                & (scalars[0, :, 1] >= 0.25)
                & (scalars[0, :, 5] == 0)
                & (scalars[0, :, 7] == 0)
            )
        if args.combat_opportunities or args.proactive_combat:
            targets[1, opportunities] *= 0.2
            targets[1, opportunities, 3] += 0.8
            opportunity_count += int(opportunities.sum())
        tracks = style_training_tracks(length, device=args.device, compositions=args.compositions)
        condition_count = len(tracks)
        inputs = {
            key: value.expand(condition_count, *value.shape[1:]).clone()
            for key, value in fair.items()
        }
        inputs["style"] = tracks
        targets = mix_style_teachers(targets, tracks)
        valid = (
            ((scalars[0, :, 0] > 0) & (scalars[0, :, 5] == 0))
            .float()[None]
            .expand(condition_count, -1)
        )
        if valid.sum() > 0:
            if args.combat_opportunities or args.proactive_combat:
                valid = valid * (1 + 2 * tracks[:, :, 0] * opportunities[None])
            episodes.append((inputs, targets, valid))
    if (args.combat_opportunities or args.proactive_combat) and not opportunity_count:
        raise ValueError("No eligible recorded combat opportunity supervision")
    if not episodes:
        raise ValueError("No active learner trajectories for distillation")
    args.output.mkdir(parents=True)
    for epoch in range(args.epochs):
        totals, hits, mass = (
            torch.zeros(condition_count, device=args.device),
            torch.zeros(condition_count, device=args.device),
            torch.zeros(condition_count, device=args.device),
        )
        for index in torch.randperm(len(episodes)).tolist():
            inputs, targets, valid = episodes[index]
            logits = student(**inputs).logits
            kl = torch.nn.functional.kl_div(logits.log_softmax(-1), targets, reduction="none").sum(
                -1
            )
            loss = (kl * valid).sum() / valid.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1)
            optimizer.step()
            totals += (kl.detach() * valid).sum(-1)
            hits += ((logits.argmax(-1) == targets.argmax(-1)) * valid).sum(-1)
            mass += valid.sum(-1)
        progress = {
            "epoch": epoch + 1,
            "training_kl": (totals / mass).tolist(),
            "training_agreement": (hits / mass).tolist(),
            "combat_opportunity_samples": opportunity_count,
        }
        torch.save(
            {
                "actor": student.state_dict(),
                "identity": {
                    "executor": digest(args.executor),
                    "teachers": {name: digest(getattr(args, name)) for name in names},
                    "data": payload["identity"]["train"],
                    "protocol": "neutral-plus-style-endpoint-soft-distillation-v1",
                    "compositions": args.compositions,
                    "combat_opportunities": args.combat_opportunities,
                    "proactive_combat": args.proactive_combat,
                    "proactive_scope": "public 16s opening teacher prior; no deployment override",
                    "opportunity_scope": "training-only geometric range; not visual visibility",
                    "target_source": digest(
                        Path(__file__).parents[1] / "training/hierarchical_style_targets.py"
                    ),
                    "source": digest(Path(__file__)),
                    "epoch": epoch + 1,
                },
                "scope": "offline endpoint distillation; control effectiveness unverified",
            },
            args.output / f"epoch-{epoch + 1}.pt",
        )
        print(json.dumps(progress), flush=True)


if __name__ == "__main__":
    main()
