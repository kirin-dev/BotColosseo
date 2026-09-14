"""Explicit scripted ENGAGE high-level opponent for pressure diagnostics only."""

import argparse
from pathlib import Path

import torch

from botcolosseo.agents.hierarchical_model import StrategicActor
from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.training.hierarchical_protocol import Command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing diagnostic opponent")
    actor = StrategicActor()
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.policy.bias[int(Command.ENGAGE)] = 100
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "identity": {
                "executor": digest(args.executor),
                "source": digest(Path(__file__)),
                "protocol": "scripted-always-engage-high-fair-visual-low-v1",
            },
            "scope": "scripted pressure opponent, not a learned strategic response or style",
        },
        args.output,
    )


if __name__ == "__main__":
    main()
