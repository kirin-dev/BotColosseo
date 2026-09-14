"""Small-budget BC with layout-disjoint validation and resumable epochs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from botcolosseo.agents.hierarchical_model import CommandExecutor
from botcolosseo.data.hierarchical_demonstrations import episode_tensors, load_command_episode
from botcolosseo.training.extraction_checkpoint import load_extraction_strong_actor
from botcolosseo.training.hierarchical_bc import (
    command_weights,
    evaluate_episode,
    fit_counterfactual_episode,
    fit_episode,
)
from botcolosseo.training.hierarchical_protocol import COMMAND_SCHEMA


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("train", "validation", "output", "strong"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--additional-train", type=Path, nargs="+", default=[])
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--counterfactual", action="store_true")
    parser.add_argument("--counterfactual-extraction-only", action="store_true")
    parser.add_argument("--conditioning-only", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--reset-policy-head", action="store_true")
    args = parser.parse_args()
    if not 0 < args.learning_rate <= 1e-3:
        raise ValueError("Learning rate must be in (0, 1e-3]")
    if args.conditioning_only and args.warm_start is None:
        raise ValueError("Conditioning-only training requires a warm-start executor")
    if args.counterfactual_extraction_only and not args.counterfactual:
        raise ValueError("Extraction-only supervision requires --counterfactual")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.reset_policy_head and args.warm_start is not None:
        raise ValueError(
            "Head reset is a fresh Strong initialization, not a warm-start continuation"
        )
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    files = {split: sorted(getattr(args, split).glob("*.npz")) for split in ("train", "validation")}
    for directory in args.additional_train:
        additional = sorted(directory.glob("*.npz"))
        if not additional:
            raise ValueError("Additional training directory is empty")
        files["train"] += additional
    if len({p.resolve() for p in files["train"]}) != len(files["train"]):
        raise ValueError("Duplicate training files")
    if not all(files.values()):
        raise ValueError("Both splits need trajectory files")
    data = {split: [load_command_episode(p) for p in paths] for split, paths in files.items()}
    layouts = {split: {int(d["seed"]) % 128 for d in episodes} for split, episodes in data.items()}
    if layouts["train"] & layouts["validation"]:
        raise ValueError("Train/validation layouts overlap")
    scenarios = {str(d["scenario_hash"]) for episodes in data.values() for d in episodes}
    if len(scenarios) != 1:
        raise ValueError("Mixed scenarios")
    for split, episodes in data.items():
        coverage = torch.zeros(7)
        for episode in episodes:
            coverage += torch.bincount(
                torch.as_tensor(episode["commands"][episode["valid"]]), minlength=7
            )
        if not bool((coverage > 0).all()):
            raise ValueError(f"{split} must cover all seven commands before training")
    identity = {
        split: {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        for split, paths in files.items()
    }
    identity["strong"] = hashlib.sha256(args.strong.read_bytes()).hexdigest()
    identity["command_schema"] = COMMAND_SCHEMA
    identity["training_protocol"] = {
        "version": "global-command-balanced-tbptt-v3",
        "learning_rate": args.learning_rate,
        "chunk_size": 64,
        "gradient_clip": 1.0,
        "shuffle_seed": 1701,
        "skip_zero_label_episodes": True,
        "counterfactual": args.counterfactual,
        "reset_policy_head": args.reset_policy_head,
    }
    if args.counterfactual_extraction_only:
        identity["training_protocol"]["counterfactual_commands"] = [5, 6]
    if args.conditioning_only:
        identity["training_protocol"]["trainable_scope"] = "command-film-only"
    actor = CommandExecutor().to(args.device)
    strong, _ = load_extraction_strong_actor(
        args.strong, expected_scenario_hash=next(iter(scenarios))
    )
    actor.initialize_from(strong)
    if args.reset_policy_head:
        actor.policy.reset_parameters()
    if args.warm_start is not None:
        parent = torch.load(args.warm_start, map_location=args.device, weights_only=False)
        if (
            parent["identity"]["command_schema"] != COMMAND_SCHEMA
            or parent["identity"]["strong"] != identity["strong"]
        ):
            raise ValueError("Warm-start provenance mismatch")
        actor.load_state_dict(parent["actor"])
        identity["warm_start"] = hashlib.sha256(args.warm_start.read_bytes()).hexdigest()
    if args.conditioning_only:
        for name, parameter in actor.named_parameters():
            parameter.requires_grad_(name.startswith("command_"))
    optimizer = torch.optim.Adam(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    counts = torch.zeros(7)
    for d in data["train"]:
        counts += torch.bincount(torch.as_tensor(d["commands"][d["valid"]]), minlength=7)
    weights = command_weights(counts).to(args.device)
    valid_episodes = sum(bool(d["valid"].any()) for d in data["train"])
    normalization_mass = float((counts.to(args.device) * weights).sum()) / valid_episodes
    identity["training_protocol"]["normalization_mass"] = normalization_mass
    start, best = 0, float("inf")
    checkpoint = args.output / "last.pt"
    if args.resume:
        saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Resume data or protocol mismatch")
        actor.load_state_dict(saved["actor"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best = saved["epoch"], saved["best"]
    elif args.output.exists():
        raise FileExistsError("Preserving existing run; use --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, args.epochs):
        order = torch.randperm(
            len(data["train"]), generator=torch.Generator().manual_seed(1701 + epoch)
        )
        for index in order.tolist():
            d = data["train"][index]
            if d["valid"].any():
                fit_episode(
                    actor,
                    episode_tensors(d, device=args.device),
                    optimizer,
                    weights,
                    normalization_mass=normalization_mass,
                )
            if (
                args.counterfactual
                and "counterfactual_actions" in d
                and d["counterfactual_valid"][::16].any()
            ):
                alternative_valid = torch.as_tensor(
                    d["counterfactual_valid"], device=args.device
                )[None].clone()
                if args.counterfactual_extraction_only:
                    alternative_valid[..., :5] = False
                if not alternative_valid[:, ::16].any():
                    continue
                fit_counterfactual_episode(
                    actor,
                    episode_tensors(d, device=args.device),
                    torch.as_tensor(d["counterfactual_actions"], device=args.device)[None],
                    alternative_valid,
                    optimizer,
                )
        stats = [
            evaluate_episode(actor, episode_tensors(d, device=args.device))
            for d in data["validation"]
        ]
        totals = {key: sum(s[key] for s in stats) for key in stats[0]}
        if not bool((totals["counts"] > 0).all()):
            raise ValueError("Validation must cover all seven commands")
        loss = float((totals["loss_sum"] / totals["counts"]).mean())
        improved = loss < best
        best = min(loss, best)
        payload = {
            "actor": actor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best": best,
            "identity": identity,
        }
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint)
        if improved:
            torch.save(payload, args.output / "best.tmp")
            (args.output / "best.tmp").replace(args.output / "best.pt")
        report = {
            "epoch": epoch + 1,
            "validation_macro_loss": loss,
            "accuracy_by_command": (totals["correct"] / totals["counts"]).tolist(),
            "switch_counts": totals["switch_counts"].tolist(),
            "switch_accuracy_by_command": [
                float(correct / count) if count > 0 else None
                for correct, count in zip(
                    totals["switch_correct"], totals["switch_counts"], strict=True
                )
            ],
            "best": best,
            "identity": identity,
        }
        (args.output / "progress.tmp").write_text(json.dumps(report, indent=2))
        (args.output / "progress.tmp").replace(args.output / "progress.json")
        print(json.dumps({k: v for k, v in report.items() if k != "identity"}), flush=True)


if __name__ == "__main__":
    main()
