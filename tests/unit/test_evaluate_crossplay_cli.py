import dataclasses
from pathlib import Path

import pytest
import torch

from botcolosseo.agents.league_opponents import OpponentSpec
from botcolosseo.cli.evaluate_crossplay import (
    CrossplayControllerFactory,
    build_parser,
    ensure_evidence_targets_absent,
    run_case_with_retries,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.evaluation.crossplay import CrossplayRow
from botcolosseo.scenarios.regions import RegionGraph


def _checkpoint_spec() -> OpponentSpec:
    return OpponentSpec(
        opponent_id="policy-a",
        kind="checkpoint",
        checkpoint="policy.pt",
        checkpoint_sha256="a" * 64,
        scenario_hash="scenario",
        selection_evidence="reports/validation.json",
    )


def _script_spec() -> OpponentSpec:
    return OpponentSpec(
        opponent_id="objective_first",
        kind="script",
        checkpoint=None,
        checkpoint_sha256=None,
        scenario_hash="scenario",
        selection_evidence="builtin:objective_first",
    )


class FakeTemplate:
    def __init__(self, marker: object | None = None) -> None:
        self.marker = marker or object()

    def fork(self):
        return FakeTemplate(self.marker)

    def reset(self) -> None:
        pass

    def act(self, observation):
        del observation
        return 0


def _row(attempts: int = 1) -> CrossplayRow:
    return CrossplayRow(
        left_policy="policy-a",
        right_policy="policy-b",
        split="validation",
        pair_index=1,
        seed=7,
        left_side="host",
        outcome="draw",
        left_objective_completed=False,
        right_objective_completed=False,
        left_score=0,
        right_score=0,
        decisions=10,
        terminated=False,
        truncated=True,
        peer_tic_lag_max=0,
        protocol_inconsistent=False,
        action_tic_inconsistent=False,
        score_event_inconsistent=False,
        scenario_hash="scenario",
        environment_attempts=attempts,
    )


def test_controller_factory_loads_checkpoint_once_and_forks_sessions() -> None:
    loads: list[str] = []

    def loader(spec, *, device):
        del device
        loads.append(spec.opponent_id)
        return FakeTemplate()

    factory = CrossplayControllerFactory(
        graph=RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml")),
        device=torch.device("cpu"),
        checkpoint_loader=loader,
    )

    first = factory.create(_checkpoint_spec(), side="host")
    second = factory.create(_checkpoint_spec(), side="opponent")

    assert loads == ["policy-a"]
    assert first._policy is not second._policy
    assert first._policy.marker is second._policy.marker


def test_controller_factory_isolates_privileged_state_to_script_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    privileged = object()

    class FakeTeacher:
        def reset(self, *, seed: int) -> None:
            assert seed == 17

        def act(self, state: object) -> MacroAction:
            assert state is privileged
            return MacroAction.IDLE

    monkeypatch.setattr(
        "botcolosseo.cli.evaluate_crossplay.create_duel_teacher",
        lambda policy_id, graph, *, side: FakeTeacher(),
    )
    factory = CrossplayControllerFactory(
        graph=RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml")),
        device=torch.device("cpu"),
        checkpoint_loader=lambda *args, **kwargs: pytest.fail(
            "script policies must not load checkpoints"
        ),
    )
    controller = factory.create(_script_spec(), side="host")

    controller.reset(seed=17)
    assert controller.act(object(), lambda: privileged) is MacroAction.IDLE


def test_crossplay_retry_is_bounded_and_records_attempt_count() -> None:
    attempts = 0

    def runner() -> CrossplayRow:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Duel respawn did not complete within the warm-up limit")
        return _row()

    result = run_case_with_retries(runner, max_attempts=2)

    assert result.environment_attempts == 2
    assert dataclasses.replace(result, environment_attempts=1) == _row()

    with pytest.raises(RuntimeError, match="unrelated"):
        run_case_with_retries(
            lambda: (_ for _ in ()).throw(RuntimeError("unrelated")), max_attempts=2
        )


def test_crossplay_refuses_to_overwrite_any_evidence_target(tmp_path: Path) -> None:
    ensure_evidence_targets_absent(tmp_path)
    (tmp_path / "crossplay.csv").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="crossplay.csv"):
        ensure_evidence_targets_absent(tmp_path)


def test_candidate_roster_arguments_are_explicit_and_paired() -> None:
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "--pool",
            "pool.json",
            "--output-dir",
            "out",
            "--candidate-checkpoint",
            "candidate.pt",
            "--candidate-id",
            "candidate-0200000",
            "--include-scripts",
        ]
    )

    assert parsed.candidate_checkpoint == Path("candidate.pt")
    assert parsed.candidate_id == "candidate-0200000"
    assert parsed.include_scripts is True
