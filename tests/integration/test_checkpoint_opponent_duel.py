import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest
import torch

from botcolosseo.agents.league_opponents import (
    CheckpointOpponentPolicy,
    OpponentSpec,
    sha256_file,
)
from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.cli.evaluate_crossplay import run_case_with_retries
from botcolosseo.evaluation.crossplay import run_crossplay_episode
from botcolosseo.scenarios.league_splits import load_league_cases
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.league_rollout import CheckpointDuelOpponentController


class TrackingController:
    def __init__(self, controller: CheckpointDuelOpponentController) -> None:
        self.controller = controller
        self.reset_count = 0
        self.frame_shapes: list[tuple[int, ...]] = []

    def reset(self, *, seed: int) -> None:
        self.reset_count += 1
        self.controller.reset(seed=seed)

    def act(self, observation, privileged_state):
        self.frame_shapes.append(observation.frame.shape)
        return self.controller.act(observation, privileged_state)


def _checkpoint(path: Path, *, scenario_hash: str) -> OpponentSpec:
    model = AsymmetricActorCritic()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.actor.policy.bias[0] = 1.0
    torch.save(
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "metadata": {
                "config_hash": "integration",
                "scenario_hash": scenario_hash,
                "counters": {"environment_steps": 0},
            },
        },
        path,
    )
    return OpponentSpec(
        opponent_id="integration-policy",
        kind="checkpoint",
        checkpoint=str(path),
        checkpoint_sha256=sha256_file(path),
        scenario_hash=scenario_hash,
        selection_evidence="integration:runtime-generated",
    )


@pytest.mark.skipif(
    os.environ.get("BOTCOLOSSEO_RUN_M3_RUNTIME_SMOKE") != "1",
    reason="set BOTCOLOSSEO_RUN_M3_RUNTIME_SMOKE=1 for the real checkpoint duel gate",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.timeout(60)
def test_runtime_generated_checkpoint_plays_both_sides_and_cleans_workers(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    scenario_hash = json.loads(
        (root / "assets/scenarios/crystal_run/manifest.json").read_text(
            encoding="utf-8"
        )
    )["wad_sha256"]
    spec = _checkpoint(tmp_path / "policy.pt", scenario_hash=scenario_hash)
    device = torch.device(os.environ.get("BOTCOLOSSEO_M3_RUNTIME_DEVICE", "cuda:1"))
    template = CheckpointOpponentPolicy.load(spec, device=device)
    cases = load_league_cases(
        root / "configs/m3/validation.json",
        expected_split="validation",
        expected_pairs=50,
    )[:2]
    graph = RegionGraph.from_yaml(
        root / "assets/scenarios/crystal_run/src/regions.yaml"
    )
    before = {process.pid for process in mp.active_children()}
    started = time.monotonic()
    rows = []
    controllers: list[TrackingController] = []
    for case in cases:
        left = TrackingController(
            CheckpointDuelOpponentController(template.fork())
        )
        right = TrackingController(
            CheckpointDuelOpponentController(template.fork())
        )
        controllers.extend((left, right))
        rows.append(
            run_case_with_retries(
                lambda case=case, left=left, right=right: run_crossplay_episode(
                    case,
                    left_spec=spec,
                    right_spec=spec,
                    left_controller=left,
                    right_controller=right,
                    graph=graph,
                    config_path=root
                    / "assets/scenarios/crystal_run/crystal_run.cfg",
                    max_decisions=525,
                ),
                max_attempts=2,
            )
        )

    assert time.monotonic() - started < 45.0
    assert [row.left_side for row in rows] == ["host", "opponent"]
    assert all(row.peer_tic_lag_max == 0 for row in rows)
    assert all(not row.protocol_inconsistent for row in rows)
    assert all(not row.action_tic_inconsistent for row in rows)
    assert all(not row.score_event_inconsistent for row in rows)
    assert all(controller.reset_count == 1 for controller in controllers)
    assert all(
        controller.frame_shapes
        and set(controller.frame_shapes) == {(84, 84)}
        for controller in controllers
    )
    assert {process.pid for process in mp.active_children()} <= before
