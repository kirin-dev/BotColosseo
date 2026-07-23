import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from botcolosseo.agents.league_opponents import sha256_file
from botcolosseo.agents.model import AsymmetricActorCritic
from botcolosseo.agents.style_model import StyledActorCritic
from botcolosseo.cli import train_league
from botcolosseo.cli.train_league import (
    _candidate_checkpoint_step,
    _collection_pending,
    _load_payoff_report,
    _load_style_warm_start,
    _run_config_hash,
    _status_json,
    _style_reward_shaper,
    _validate_base_authorization,
    _validate_config,
    build_parser,
    main,
)
from botcolosseo.scenarios.league_splits import (
    generate_league_splits,
    write_league_manifests,
)
from botcolosseo.training.defensive_reward import DefensiveRewardLedger
from botcolosseo.training.historical_pool import (
    HistoricalPoolManifest,
    PoolEntry,
    write_pool_atomic,
)
from botcolosseo.training.league_rollout import LeagueEpisodeResult
from botcolosseo.training.ppo import PPOUpdateMetrics


def _entry(index: int, *, anchor: bool) -> PoolEntry:
    return PoolEntry(
        policy_id=f"policy-{index}",
        checkpoint=f"runs/policy-{index}.pt",
        checkpoint_sha256=f"{index + 1:064x}",
        scenario_hash="6" * 64,
        config_hash="2" * 64,
        source_git_commit="a" * 40,
        parent_checkpoint_sha256="1" * 64,
        environment_steps=index * 200_000,
        admitted_at_utc=f"2026-07-21T0{index}:00:00Z",
        validation_report=f"reports/validation-{index}.json",
        validation_report_sha256=f"{index + 10:064x}",
        script_average_win_rate=0.75,
        script_worst_case_win_rate=0.60,
        objective_rate=0.90,
        payoff_by_policy={"axis": 0.5},
        anchor=anchor,
        admission_reason="anchor" if anchor else "diversity",
    )


def _pool() -> HistoricalPoolManifest:
    return HistoricalPoolManifest(
        1,
        0,
        None,
        "2026-07-21T00:00:00Z",
        (_entry(0, anchor=True), _entry(1, anchor=False)),
    )


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "seed": 20260721,
        "train_cases": "configs/m3/train.json",
        "environment_steps": 2_000_000,
        "rollout_steps": 256,
        "candidate_interval_steps": 200_000,
        "max_episode_decisions": 525,
        "sequence_length": 16,
        "burn_in": 8,
        "minibatch_sequences": 16,
        "update_epochs": 4,
        "learning_rate": 0.00005,
        "weight_decay": 0.00001,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "policy_clip": 0.2,
        "value_clip": 0.2,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.01,
        "gradient_clip": 0.5,
        "max_kl": 0.03,
        "shaping_decay_steps": 600_000,
    }


def _write_cli_project(root: Path) -> tuple[Path, Path, Path, Path]:
    (root / "assets/scenarios/crystal_run").mkdir(parents=True)
    (root / "assets/scenarios/crystal_run/src").mkdir()
    (root / "assets/scenarios/crystal_run/src/regions.yaml").write_text(
        Path("assets/scenarios/crystal_run/src/regions.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    (root / "runs").mkdir()
    write_league_manifests(generate_league_splits(), root / "configs/m3")
    (root / "assets/scenarios/crystal_run/manifest.json").write_text(
        json.dumps({"wad_sha256": "6" * 64}), encoding="utf-8"
    )
    base = root / "runs/base.pt"
    model = AsymmetricActorCritic()
    torch.save(
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "metadata": {
                "config_hash": "m2",
                "scenario_hash": "6" * 64,
                "counters": {"environment_steps": 800_000},
            },
        },
        base,
    )
    report = root / "reports/validation-anchor.json"
    report.parent.mkdir()
    report.write_text("validation", encoding="utf-8")
    anchor = replace(
        _entry(0, anchor=True),
        checkpoint="runs/base.pt",
        checkpoint_sha256=sha256_file(base),
        validation_report="reports/validation-anchor.json",
        validation_report_sha256=sha256_file(report),
        parent_checkpoint_sha256=sha256_file(base),
    )
    pool = HistoricalPoolManifest(
        1, 0, None, "2026-07-21T00:00:00Z", (anchor,)
    )
    pool_path = root / "reports/pool.json"
    write_pool_atomic(pool, pool_path)
    payoffs = root / "reports/payoffs.json"
    payoffs.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "validation",
                "pool_manifest_sha256": pool.manifest_sha256,
                "win_rates": {"policy-0": 0.5},
            }
        ),
        encoding="utf-8",
    )
    config = root / "configs/m3/league.yaml"
    config.write_text(yaml.safe_dump(_config(), sort_keys=True), encoding="utf-8")
    return config, base, pool_path, payoffs


def test_parser_requires_base_pool_payoffs_and_run_dir() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    parsed = parser.parse_args(
        [
            "--base-checkpoint",
            "base.pt",
            "--pool",
            "pool.json",
            "--payoffs",
            "payoffs.json",
            "--run-dir",
            "run",
        ]
    )
    assert parsed.device == "cuda:0"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--base-checkpoint",
                "base.pt",
                "--pool",
                "pool.json",
                "--payoffs",
                "payoffs.json",
                "--run-dir",
                "run",
                "--resume",
                "latest.pt",
                "--transition-from",
                "candidate.pt",
            ]
        )


def test_config_is_frozen_and_cannot_reference_test_manifests() -> None:
    _validate_config(_config())
    with pytest.raises(ValueError, match="train manifest"):
        _validate_config({**_config(), "train_cases": "configs/m3/test.json"})
    with pytest.raises(ValueError, match="schema"):
        _validate_config({**_config(), "schema_version": 2})
    with pytest.raises(ValueError, match="candidate interval"):
        _validate_config({**_config(), "candidate_interval_steps": 100_000})


def test_defensive_style_config_and_reward_factory_are_supported() -> None:
    config = yaml.safe_load(
        Path("configs/m5/defensive_ppo.yaml").read_text(encoding="utf-8")
    )

    _validate_config(config, style="defensive")
    reward = _style_reward_shaper(
        "defensive",
        config["style"],
        learner_side="host",
    )

    assert isinstance(reward, DefensiveRewardLedger)


def test_defensive_interpolation_is_a_hash_bound_style_warm_start(
    tmp_path: Path,
) -> None:
    base_hash = "a" * 64
    scenario_hash = "b" * 64
    model = StyledActorCritic.from_base(AsymmetricActorCritic(), bottleneck=32)
    path = tmp_path / "defensive-alpha-025.pt"
    torch.save(
        {
            "schema_version": 1,
            "kind": "style_interpolation",
            "style": "defensive",
            "alpha": 0.25,
            "base_checkpoint_sha256": base_hash,
            "scenario_hash": scenario_hash,
            "distilled_checkpoint_sha256": "c" * 64,
            "neutral_checkpoint_sha256": "d" * 64,
            "interpolation_sha256": "e" * 64,
            "model": model.state_dict(),
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "style": "defensive",
                "base_checkpoint_sha256": base_hash,
                "scenario_hash": scenario_hash,
                "checkpoint_sha256": sha256_file(path),
                "test_cases_accessed": False,
            }
        ),
        encoding="utf-8",
    )

    metadata = _load_style_warm_start(
        path,
        model=model,
        expected_base_checkpoint_sha256=base_hash,
        expected_scenario_hash=scenario_hash,
        expected_style="defensive",
    )

    assert metadata["alpha"] == 0.25
    with pytest.raises(ValueError, match="identity"):
        _load_style_warm_start(
            path,
            model=model,
            expected_base_checkpoint_sha256=base_hash,
            expected_scenario_hash=scenario_hash,
            expected_style="aggressive",
        )


def test_base_authorization_distinguishes_promoted_and_explicit_provisional() -> None:
    base_hash = "1" * 64
    assert _validate_base_authorization(base_hash, None, allow_provisional=True) is False
    with pytest.raises(ValueError, match="provisional"):
        _validate_base_authorization(base_hash, None, allow_provisional=False)
    assert (
        _validate_base_authorization(
            base_hash,
            {"passed": True, "checkpoint_sha256": {"ppo": base_hash}},
            allow_provisional=False,
        )
        is True
    )
    with pytest.raises(ValueError, match="capability"):
        _validate_base_authorization(
            base_hash,
            {"passed": False, "checkpoint_sha256": {"ppo": base_hash}},
            allow_provisional=True,
        )
    with pytest.raises(ValueError, match="checkpoint"):
        _validate_base_authorization(
            base_hash,
            {"passed": True, "checkpoint_sha256": {"ppo": "0" * 64}},
            allow_provisional=True,
        )
    assert (
        _validate_base_authorization(
            base_hash,
            {
                "passed": False,
                "integrity_passed": True,
                "checkpoint_sha256": {"ppo": base_hash},
            },
            allow_provisional=False,
            allow_integrity_qualified=True,
        )
        is False
    )
    with pytest.raises(ValueError, match="capability"):
        _validate_base_authorization(
            base_hash,
            {
                "passed": False,
                "integrity_passed": True,
                "checkpoint_sha256": {"ppo": base_hash},
            },
            allow_provisional=False,
        )


def test_payoff_report_is_validation_complete_and_pool_bound(tmp_path: Path) -> None:
    pool = _pool()
    path = tmp_path / "payoffs.json"
    payload = {
        "schema_version": 1,
        "split": "validation",
        "pool_manifest_sha256": pool.manifest_sha256,
        "win_rates": {"policy-0": 0.4, "policy-1": 0.6},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _load_payoff_report(path, pool) == payload["win_rates"]

    path.write_text(json.dumps({**payload, "split": "test"}), encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        _load_payoff_report(path, pool)
    path.write_text(
        json.dumps({**payload, "win_rates": {"policy-0": 0.4}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="policy set"):
        _load_payoff_report(path, pool)


def test_candidate_cadence_and_status_output_are_deterministic() -> None:
    assert (
        _candidate_checkpoint_step(
            199_936, 200_192, interval=200_000, target=2_000_000
        )
        == 200_192
    )
    assert (
        _candidate_checkpoint_step(
            200_192, 200_448, interval=200_000, target=2_000_000
        )
        is None
    )
    assert _status_json({"z": 1, "a": 2}) == '{\n  "a": 2,\n  "z": 1\n}'


def test_paired_boundary_collection_stays_in_one_process_until_even_episode() -> None:
    assert _collection_pending(
        environment_steps=200_000,
        stop_after=200_000,
        episode_index=3,
        finish_paired_boundary=True,
    )
    assert not _collection_pending(
        environment_steps=200_256,
        stop_after=200_000,
        episode_index=4,
        finish_paired_boundary=True,
    )
    assert not _collection_pending(
        environment_steps=200_000,
        stop_after=200_000,
        episode_index=3,
        finish_paired_boundary=False,
    )


def test_run_config_hash_binds_actual_target_and_rollout() -> None:
    first = _run_config_hash("a" * 64, target_steps=2_000_000, rollout_steps=256)

    assert first == _run_config_hash(
        "a" * 64, target_steps=2_000_000, rollout_steps=256
    )
    assert first != _run_config_hash(
        "a" * 64, target_steps=1_000_000, rollout_steps=256
    )
    assert first != _run_config_hash(
        "a" * 64, target_steps=2_000_000, rollout_steps=128
    )


def test_preflight_binds_all_training_inputs_without_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "repo"
    (root / "assets/scenarios/crystal_run").mkdir(parents=True)
    (root / "runs").mkdir()
    write_league_manifests(generate_league_splits(), root / "configs/m3")
    (root / "assets/scenarios/crystal_run/manifest.json").write_text(
        json.dumps({"wad_sha256": "6" * 64}), encoding="utf-8"
    )
    base = root / "runs/base.pt"
    model = AsymmetricActorCritic()
    torch.save(
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "metadata": {
                "config_hash": "m2",
                "scenario_hash": "6" * 64,
                "counters": {"environment_steps": 800_000},
            },
        },
        base,
    )
    report = root / "reports/validation-anchor.json"
    report.parent.mkdir()
    report.write_text("validation", encoding="utf-8")
    anchor = replace(
        _entry(0, anchor=True),
        checkpoint="runs/base.pt",
        checkpoint_sha256=sha256_file(base),
        validation_report="reports/validation-anchor.json",
        validation_report_sha256=sha256_file(report),
        parent_checkpoint_sha256=sha256_file(base),
    )
    pool = HistoricalPoolManifest(
        1, 0, None, "2026-07-21T00:00:00Z", (anchor,)
    )
    pool_path = root / "reports/pool.json"
    write_pool_atomic(pool, pool_path)
    payoffs = root / "reports/payoffs.json"
    payoffs.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "validation",
                "pool_manifest_sha256": pool.manifest_sha256,
                "win_rates": {"policy-0": 0.5},
            }
        ),
        encoding="utf-8",
    )
    config = root / "configs/m3/league.yaml"
    config.write_text(yaml.safe_dump(_config(), sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(train_league, "_project_root", lambda: root)

    result = main(
        [
            "--config",
            str(config),
            "--base-checkpoint",
            str(base),
            "--pool",
            str(pool_path),
            "--payoffs",
            str(payoffs),
            "--run-dir",
            str(root / "runs/m3/preflight"),
            "--allow-provisional-base",
            "--preflight",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["preflight_passed"] is True
    assert output["base_promoted"] is False
    assert output["test_cases_accessed"] is False
    assert output["train_case_count"] == 500


def test_fake_training_runtime_writes_checkpoint_candidate_metrics_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    config, base, pool_path, payoffs = _write_cli_project(root)

    class FakeRollout:
        def sequence_minibatches(self, **kwargs):
            del kwargs
            yield object()

    class FakeCollector:
        def __init__(self, model, *, episode_index, **kwargs):
            del model, kwargs
            self.episode_index = episode_index

        def collect(self, *, steps, start_environment_step):
            del start_environment_step
            episode = LeagueEpisodeResult(
                episode_index=self.episode_index,
                seed=7,
                opponent="objective_first",
                opponent_kind="script",
                source="script",
                pair_slot=self.episode_index // 2,
                sampling_probability=1.0,
                learner_side="host",
                decisions=steps,
                reward=1.0,
                objective_completed=True,
                terminated=True,
                truncated=False,
            )
            self.episode_index += 1
            return type(
                "Collection",
                (),
                {
                    "rollout": FakeRollout(),
                    "environment_steps": steps,
                    "episodes": (episode,),
                    "event_counts": {"learner:score": 1},
                },
            )()

        def close(self) -> None:
            pass

    class FakeTrainer:
        def __init__(self, model):
            self.model = model
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1
            )
            self.updates = 0

        @classmethod
        def create(cls, model, **kwargs):
            del kwargs
            return cls(model)

        def train_step(self, batch):
            del batch
            self.updates += 1
            return PPOUpdateMetrics(
                total_loss=0.1,
                policy_loss=0.05,
                value_loss=0.1,
                entropy=1.0,
                approximate_kl=0.0,
                clip_fraction=0.0,
                valid_count=1,
                pre_clip_grad_norm=0.1,
                post_clip_grad_norm=0.1,
                learning_rate=1e-4,
                update=self.updates,
            )

    monkeypatch.setattr(train_league, "_project_root", lambda: root)
    monkeypatch.setattr(
        train_league, "LeagueRolloutCollector", FakeCollector, raising=False
    )
    monkeypatch.setattr(train_league, "PPOTrainer", FakeTrainer, raising=False)
    run_dir = root / "runs/m3/fake"

    assert (
        main(
            [
                "--config",
                str(config),
                "--base-checkpoint",
                str(base),
                "--pool",
                str(pool_path),
                "--payoffs",
                str(payoffs),
                "--run-dir",
                str(run_dir),
                "--allow-provisional-base",
                "--device",
                "cpu",
                "--environment-steps",
                "2",
                "--stop-after-steps",
                "1",
                "--rollout-steps",
                "1",
            ]
        )
        == 0
    )
    first = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert first["completed"] is False
    assert (
        main(
            [
                "--config",
                str(config),
                "--base-checkpoint",
                str(base),
                "--pool",
                str(pool_path),
                "--payoffs",
                str(payoffs),
                "--run-dir",
                str(run_dir),
                "--allow-provisional-base",
                "--device",
                "cpu",
                "--environment-steps",
                "2",
                "--rollout-steps",
                "1",
                "--resume",
                str(run_dir / "latest.pt"),
            ]
        )
        == 0
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed"] is True
    assert summary["environment_steps"] == 2
    assert summary["episode_count"] == 2
    assert summary["opponent_source_counts"] == {"script": 2}
    assert summary["pfsp_probabilities"] == {}
    assert summary["test_cases_accessed"] is False
    assert (run_dir / "latest.pt").is_file()
    assert (run_dir / "candidate-0000002.pt").is_file()
    assert (run_dir / "metrics.jsonl").is_file()

    previous_pool = train_league.load_pool(pool_path, artifact_root=root)
    transition_source = run_dir / "latest.pt"
    validation_report = root / "reports/validation-candidate.json"
    validation_report.write_text("validation", encoding="utf-8")
    candidate = replace(
        _entry(1, anchor=False),
        checkpoint="runs/m3/fake/latest.pt",
        checkpoint_sha256=sha256_file(transition_source),
        parent_checkpoint_sha256=sha256_file(base),
        validation_report="reports/validation-candidate.json",
        validation_report_sha256=sha256_file(validation_report),
    )
    next_pool = HistoricalPoolManifest(
        1,
        1,
        previous_pool.manifest_sha256,
        "2026-07-21T01:00:00Z",
        (*previous_pool.entries, candidate),
    )
    next_pool_path = root / "reports/pool-v1.json"
    write_pool_atomic(next_pool, next_pool_path)
    next_payoffs = root / "reports/payoffs-v1.json"
    next_payoffs.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "validation",
                "pool_manifest_sha256": next_pool.manifest_sha256,
                "win_rates": {"policy-0": 0.5, "policy-1": 0.5},
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--config",
                str(config),
                "--base-checkpoint",
                str(base),
                "--pool",
                str(next_pool_path),
                "--payoffs",
                str(next_payoffs),
                "--run-dir",
                str(run_dir),
                "--allow-provisional-base",
                "--device",
                "cpu",
                "--environment-steps",
                "2",
                "--rollout-steps",
                "1",
                "--transition-from",
                str(transition_source),
            ]
        )
        == 0
    )
    transition = json.loads(
        (run_dir / "transitions/transition-0000002.json").read_text(
            encoding="utf-8"
        )
    )
    assert transition["previous_identity"]["pool_manifest_hash"] == (
        previous_pool.manifest_sha256
    )
    assert transition["next_identity"]["pool_manifest_hash"] == (
        next_pool.manifest_sha256
    )
    assert sha256_file(Path(transition["archive_checkpoint"])) == (
        transition["source_checkpoint_sha256"]
    )
