"""Pair unchanged style actors with a verified Hard-anchored executor candidate."""

import argparse
import json
from pathlib import Path

import torch

from botcolosseo.agents.hierarchical_model import CommandExecutor
from botcolosseo.cli.train_hierarchical_strategic import digest


def prepare(executor, reference, strategies, output):
    if output.exists():
        raise FileExistsError("Preserve existing control candidate")
    candidate = torch.load(executor, map_location="cpu", weights_only=False)
    original = torch.load(reference, map_location="cpu", weights_only=False)
    protocol = candidate["identity"].get("difficulty_distillation", {})
    reference_hash = digest(reference)
    if not protocol.get("anchor_hard") or protocol["teachers"][-1] != reference_hash:
        raise ValueError("Candidate is not anchored to this reference")
    model = CommandExecutor()
    model.load_state_dict(candidate["actor"])
    for name, value in original["actor"].items():
        if name.startswith(("difficulty_input.network.", "difficulty_output.network.")):
            anchored_name = name.replace(".network.", ".anchor_network.")
            if not torch.equal(value, candidate["actor"][anchored_name]):
                raise ValueError("Frozen Hard anchor differs from the reference")
        elif not torch.equal(value, candidate["actor"][name]):
            raise ValueError("Non-difficulty executor weights changed")
    items = [torch.load(p, map_location="cpu", weights_only=False) for p in strategies]
    if not items or any(p["identity"]["executor"] != reference_hash for p in items):
        raise ValueError("High actors must belong to the reference executor")
    output.mkdir(parents=True)
    manifest = {
        "executor": digest(executor),
        "reference_executor": reference_hash,
        "status": "control_candidate_requires_rollout_validation",
        "strategies": [],
    }
    for index, (path, item) in enumerate(zip(strategies, items, strict=True)):
        target = output / f"strategy-{index}.pt"
        torch.save(
            {
                "actor": item["actor"],
                "identity": {
                    "executor": manifest["executor"],
                    "original_executor": reference_hash,
                    "original_strategy": digest(path),
                    "scope": "unchanged high actor with Hard-anchored low; candidate, not promoted",
                },
            },
            target,
        )
        manifest["strategies"].append(
            {
                "source": str(path),
                "source_sha256": digest(path),
                "candidate": str(target),
                "candidate_sha256": digest(target),
            }
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    for name in ("executor", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--strategies", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.executor, args.reference, args.strategies, args.output)))


if __name__ == "__main__":
    main()
