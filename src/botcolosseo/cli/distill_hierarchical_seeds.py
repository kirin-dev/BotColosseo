"""Initialize strategic neural seeds from public-state command heuristics."""

import argparse
import json
from pathlib import Path

import torch

from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.agents.hierarchical_seed import SEED_PROFILES, seed_command
from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.data.hierarchical_demonstrations import episode_tensors, load_command_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.output.exists() or args.epochs <= 0:
        raise ValueError("Need fresh output and positive epochs")
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    payload = torch.load(args.executor, map_location="cpu", weights_only=False)
    executor = CommandExecutor().to(args.device).eval()
    executor.load_state_dict(payload["actor"])
    executor.requires_grad_(False)
    episodes = []
    for filename, expected_hash in payload["identity"]["train"].items():
        path = Path(filename)
        if digest(path) != expected_hash:
            raise ValueError("Executor training trajectory changed")
        raw = load_command_episode(path)
        batch = episode_tensors(raw, device=args.device)
        with torch.no_grad():
            out = executor(**{key: batch[key] for key in (
                "frames", "scalars", "previous_actions", "masks", "commands", "difficulty"
            )})
        # At a high boundary, controller sees features from the preceding low action.
        prior = torch.cat((torch.zeros_like(out.features[:, :1]), out.features[:, :-1]), 1)
        episodes.append((raw, prior[:, ::8].detach(), batch["scalars"][:, ::8]))
    args.output.mkdir(parents=True)
    for profile in SEED_PROFILES:
        actor = StrategicActor().to(args.device)
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        for epoch in range(args.epochs):
            correct = total = 0
            for index in torch.randperm(len(episodes)).tolist():
                raw, features, scalars = episodes[index]
                previous, labels = [], []
                command, health = 1, 1.0
                for state in raw["scalars"][::8]:
                    previous.append(command)
                    command = int(seed_command(
                        state, profile=profile, previous_health=health, previous_command=command
                    ))
                    health = float(state[0])
                    labels.append(command)
                length = len(labels)
                masks = torch.ones(1, length, device=args.device)
                masks[:, 0] = 0
                output = actor(
                    features=features, scalars=scalars,
                    previous_commands=torch.tensor([previous], device=args.device),
                    elapsed=(8 * masks)[..., None], masks=masks,
                    style=torch.zeros(1, length, 3, device=args.device),
                    difficulty=torch.ones(1, length, 1, device=args.device),
                )
                targets = torch.tensor(labels, device=args.device)
                loss = torch.nn.functional.cross_entropy(output.logits[0], targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
                optimizer.step()
                correct += int((output.logits[0].argmax(-1) == targets).sum())
                total += length
            print(json.dumps({"profile": profile, "epoch": epoch + 1,
                              "training_label_accuracy": correct / total}), flush=True)
        torch.save({
            "actor": actor.state_dict(),
            "identity": {"executor": digest(args.executor), "profile": profile,
                         "protocol": "public-heuristic-distilled-neutral-seed-v1",
                         "epochs": args.epochs, "data": payload["identity"]["train"],
                         "source": digest(Path(__file__)),
                         "teacher_source": digest(
                             Path(__file__).parents[1] / "agents/hierarchical_seed.py"
                         )},
            "scope": "population initialization, not learned response or validated style",
        }, args.output / f"{profile}.pt")


if __name__ == "__main__":
    main()
