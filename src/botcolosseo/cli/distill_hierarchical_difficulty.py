"""Short, resumable difficulty-FiLM distillation from command-trained snapshots."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from botcolosseo.agents.hierarchical_model import CommandExecutor
from botcolosseo.data.hierarchical_demonstrations import episode_tensors, load_command_episode
from botcolosseo.training.hierarchical_difficulty import distill_episode, freeze_for_difficulty


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("easy", "normal", "hard", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-episodes", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--anchor-hard", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if not 0 < args.learning_rate <= 1e-3:
        raise ValueError("Learning rate must be in (0, 1e-3]")
    if args.epochs <= 0 or args.max_train_episodes <= 0:
        raise ValueError("Positive training budget required")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Preserving existing run; use --resume")
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    paths = [args.easy, args.normal, args.hard]
    payloads = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    identity = copy.deepcopy(payloads[-1]["identity"])
    if len({p["identity"]["command_schema"] for p in payloads}) != 1:
        raise ValueError("Teacher command schemas differ")
    files = {split: sorted(identity[split]) for split in ("train", "validation")}
    # Deterministic spread through the existing corpus, not just its first directory.
    if len(files["train"]) > args.max_train_episodes:
        indices = torch.linspace(0, len(files["train"]) - 1, args.max_train_episodes).long()
        files["train"] = [files["train"][i] for i in indices.tolist()]
    data = {}
    for split, selected in files.items():
        data[split] = []
        for path in selected:
            if digest(path) != identity[split][path]:
                raise ValueError("Source trajectory hash mismatch")
            data[split].append(load_command_episode(Path(path)))
        identity[split] = {p: identity[split][p] for p in selected}
    if not all(data.values()):
        raise ValueError("Both data splits are required")
    layouts = {k: {int(d["seed"]) % 128 for d in rows} for k, rows in data.items()}
    if layouts["train"] & layouts["validation"]:
        raise ValueError("Training/validation layouts overlap")
    scenarios = {str(d["scenario_hash"]) for rows in data.values() for d in rows}
    for payload in payloads:
        path = Path(next(iter(payload["identity"]["validation"])))
        scenarios.add(str(load_command_episode(path)["scenario_hash"]))
    if len(scenarios) != 1:
        raise ValueError("Teacher/data scenario mismatch")
    identity["difficulty_distillation"] = {
        "anchor_hard": args.anchor_hard,
        "teachers": [digest(p) for p in paths],
        "conditions": [0.0, 0.5, 1.0],
        "loss_weights": [1, 1, 2],
        "learning_rate": args.learning_rate,
        "chunk_size": 64,
        "source": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path(__file__).parents[1] / "training/hierarchical_difficulty.py",
                Path(__file__).parents[1] / "agents/hierarchical_model.py",
            )
        },
        "scope": "provisional snapshot curriculum; closed-loop calibration required",
    }
    teachers = []
    for payload in payloads:
        teacher = CommandExecutor().to(args.device)
        teacher.load_state_dict(payload["actor"])
        teacher.eval().requires_grad_(False)
        teachers.append(teacher)
    actor = CommandExecutor().to(args.device)
    actor.load_state_dict(payloads[-1]["actor"])
    if args.anchor_hard:
        actor.difficulty_input.anchor_at_hard()
        actor.difficulty_output.anchor_at_hard()
    optimizer = torch.optim.Adam(freeze_for_difficulty(actor), lr=args.learning_rate)
    start = 0
    if args.resume:
        saved = torch.load(args.output / "last.pt", map_location=args.device, weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Resume identity mismatch")
        actor.load_state_dict(saved["actor"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["epoch"]
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, args.epochs):
        order = torch.randperm(
            len(data["train"]), generator=torch.Generator().manual_seed(1701 + epoch)
        )
        for index in order.tolist():
            distill_episode(
                actor,
                teachers,
                episode_tensors(data["train"][index], device=args.device),
                optimizer,
            )
        statistics = [
            distill_episode(actor, teachers, episode_tensors(d, device=args.device))
            for d in data["validation"]
        ]
        report = {
            "epoch": epoch + 1,
            "train_episodes": len(data["train"]),
            "validation_episode_mean": {
                k: torch.tensor([r[k] for r in statistics]).mean(0).tolist()
                for k in ("kl", "agreement")
            },
            "scope": "offline imitation, not closed-loop difficulty success",
        }
        saved = {
            "actor": actor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "identity": identity,
            "epoch": epoch + 1,
            "report": report,
        }
        for name in (f"epoch-{epoch + 1}.pt", "last.pt"):
            temporary = args.output / (name + ".tmp")
            torch.save(saved, temporary)
            temporary.replace(args.output / name)
        temporary = args.output / "progress.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(args.output / "progress.json")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
