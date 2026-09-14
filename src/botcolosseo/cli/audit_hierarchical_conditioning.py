"""Same-history, one-step command interventions on recorded fair observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from botcolosseo.agents.hierarchical_model import CommandExecutor
from botcolosseo.data.hierarchical_demonstrations import episode_tensors, load_command_episode


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserving existing audit")
    torch.set_num_threads(2)
    model = CommandExecutor().eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["actor"])
    counts = torch.zeros(7, 7)
    flips = torch.zeros(7, 7)
    variation = torch.zeros(7, 7)
    samples = 0
    label_counts = torch.zeros(7)
    label_hits = torch.zeros(7)
    endpoint_distinct = 0
    endpoint_both_correct = 0
    paths = sorted(args.data.glob("*.npz"))
    for path in paths:
        episode = load_command_episode(path)
        data = episode_tensors(episode)
        hidden = None
        keys = ("frames", "scalars", "previous_actions", "masks", "commands", "difficulty")
        for t in range(data["valid"].shape[1]):
            inputs = {k: data[k][:, t : t + 1] for k in keys}
            if bool(data["valid"][0, t]) and t % 16 == 0:
                alternatives = []
                for command in range(7):
                    changed = dict(inputs, commands=torch.tensor([[command]]))
                    alternatives.append(model(**changed, hidden=hidden).logits[0, 0].softmax(-1))
                probabilities = torch.stack(alternatives)
                choices = probabilities.argmax(-1)
                if "counterfactual_actions" in episode:
                    labels = torch.as_tensor(episode["counterfactual_actions"][t])
                    applicable = torch.as_tensor(episode["counterfactual_valid"][t])
                    label_counts += applicable
                    label_hits += (choices == labels) & applicable
                    if bool(applicable[5] & applicable[6]) and labels[5] != labels[6]:
                        endpoint_distinct += 1
                        endpoint_both_correct += int(bool((choices[5:] == labels[5:]).all()))
                flips += (choices[:, None] != choices[None, :]).float()
                variation += 0.5 * (probabilities[:, None] - probabilities[None, :]).abs().sum(-1)
                counts += 1
                samples += 1
            hidden = model(**inputs, hidden=hidden).hidden
    if samples == 0:
        raise ValueError("No valid intervention samples")
    result = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "samples": samples,
        "protocol": "same-hidden-one-step-intervention-stride16",
        "data_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "argmax_disagreement": (flips / counts).tolist(),
        "total_variation": (variation / counts).tolist(),
        "oracle_label_counts": label_counts.tolist(),
        "oracle_action_accuracy": [
            float(hits / count) if count else None
            for hits, count in zip(label_hits, label_counts, strict=True)
        ],
        "distinct_endpoint_label_samples": endpoint_distinct,
        "distinct_endpoint_both_correct": endpoint_both_correct,
        "interpretation": "Responsiveness diagnostic, not correct command execution evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "data_hashes"}), flush=True)


if __name__ == "__main__":
    main()
