from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.training.league_checkpoint import LeagueCheckpointState


def freeze_candidate(source: Path, output: Path) -> dict[str, object]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source == output:
        raise ValueError("Candidate source and output must differ")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported M3 candidate checkpoint schema")
    try:
        state = LeagueCheckpointState(**payload["state"])
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid M3 candidate checkpoint state") from error
    if state.episodes % 2:
        raise ValueError("Candidate freeze requires a paired episode boundary")
    source_hash = sha256_file(source)
    if output.exists():
        if sha256_file(output) != source_hash:
            raise FileExistsError("Candidate output exists with a different hash")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary, source.open("rb") as reader:
                temporary_name = temporary.name
                shutil.copyfileobj(reader, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, output)
        except BaseException:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
            raise
    return {
        "checkpoint": str(output),
        "checkpoint_sha256": source_hash,
        "environment_steps": state.environment_steps,
        "episodes": state.episodes,
        "paired_boundary": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze an immutable M3 validation candidate"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(freeze_candidate(args.source, args.output), indent=2, sort_keys=True))
    return 0
