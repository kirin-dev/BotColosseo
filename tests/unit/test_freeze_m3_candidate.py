from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.cli.freeze_m3_candidate import freeze_candidate
from botcolosseo.training.league_checkpoint import LeagueCheckpointState


def _checkpoint(path: Path, *, episodes: int) -> None:
    torch.save(
        {
            "schema_version": 1,
            "state": asdict(
                LeagueCheckpointState(200_000, 10, episodes, episodes // 2)
            ),
        },
        path,
    )


def test_freeze_candidate_requires_pair_boundary_and_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "latest.pt"
    output = tmp_path / "candidate.pt"
    _checkpoint(source, episodes=2)

    result = freeze_candidate(source, output)
    repeated = freeze_candidate(source, output)

    assert result == repeated
    assert result["environment_steps"] == 200_000
    assert result["checkpoint_sha256"] == sha256_file(output)

    odd = tmp_path / "odd.pt"
    _checkpoint(odd, episodes=3)
    with pytest.raises(ValueError, match="paired episode boundary"):
        freeze_candidate(odd, tmp_path / "rejected.pt")
