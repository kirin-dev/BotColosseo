from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import botcolosseo.cli.render_showcase as showcase_cli
from botcolosseo.demo.showcase import RecordedShowcaseEpisode
from botcolosseo.evaluation.showcase import (
    ShowcasePolicySpec,
    load_showcase_config,
)
from botcolosseo.scenarios.duel_splits import DuelCase


def test_cli_exposes_only_config_checkpoint_root_and_device() -> None:
    args = showcase_cli.build_parser().parse_args(
        [
            "--config",
            "configs/showcase/development.yaml",
            "--checkpoint-root",
            "artifacts",
            "--device",
            "cpu",
        ]
    )

    assert args.config == Path("configs/showcase/development.yaml")
    assert args.checkpoint_root == Path("artifacts")
    assert args.device == "cpu"


def test_unknown_development_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="development policy"):
        showcase_cli.load_showcase_policy(
            ShowcasePolicySpec("random", "Random", Path("x.pt"), "1" * 64),
            publication=False,
            checkpoint_root=Path.cwd(),
            scenario_hash="a" * 64,
            device="cpu",
        )


def test_checkpoint_hash_mismatch_fails_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "runs/m2/ppo-full/selected.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"wrong")
    called = False

    def fail_if_called(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(showcase_cli, "load_actor_policy", fail_if_called)

    with pytest.raises(ValueError, match="hash"):
        showcase_cli.load_showcase_policy(
            ShowcasePolicySpec("ppo", "PPO", checkpoint, "1" * 64),
            publication=False,
            checkpoint_root=tmp_path,
            scenario_hash="a" * 64,
            device="cpu",
        )

    assert called is False


def test_m4_target_names_match_the_public_contract(tmp_path: Path) -> None:
    root = Path.cwd()
    development = load_showcase_config(
        Path("configs/showcase/development.yaml"), root=root
    )
    config = replace(
        development,
        stage="m4",
        publication=True,
        policies=(
            ShowcasePolicySpec(
                "strong_base", "Strong Base", root / "runs/m3/selected.pt", "1" * 64
            ),
            ShowcasePolicySpec(
                "aggressive", "Aggressive", root / "runs/m4/selected.pt", "2" * 64
            ),
        ),
        output_dir=tmp_path / "media",
        evidence_dir=tmp_path / "evidence",
    )

    targets = showcase_cli._target_paths(config)

    assert targets["comparison_gif"].name == "m4-base-vs-aggressive.gif"
    assert targets["strong_base_mp4"].name == "m4-strong-base.mp4"
    assert targets["aggressive_mp4"].name == "m4-aggressive.mp4"
    assert targets["metrics_png"].name == "m4-metrics.png"


def test_development_renderer_stays_non_public_and_aligns_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path.cwd()
    loaded = load_showcase_config(
        Path("configs/showcase/development.yaml"), root=root
    )
    config = replace(
        loaded,
        output_dir=tmp_path / "media",
        evidence_dir=tmp_path / "evidence",
    )
    case = DuelCase(
        "validation", 250, 656489971, "random_legal", "host", 0, "direct_lower"
    )
    videos: dict[Path, tuple[np.ndarray, ...]] = {}
    gif_frames: list[np.ndarray] = []
    decisions = {"ppo": 2, "bc": 1}

    monkeypatch.setattr(showcase_cli, "load_showcase_config", lambda path, root: config)
    monkeypatch.setattr(
        showcase_cli,
        "load_showcase_cases",
        lambda path, root, expected_count: (case,),
    )
    monkeypatch.setattr(
        showcase_cli, "load_showcase_policy", lambda *args, **kwargs: object()
    )

    def fake_record(*args: object, **kwargs: object) -> RecordedShowcaseEpisode:
        policy_id = str(kwargs["policy_id"])
        count = decisions[policy_id]
        frames = tuple(
            np.full((300, 256, 3), index + 1, dtype=np.uint8)
            for index in range(count)
        )
        return RecordedShowcaseEpisode(
            policy_id=policy_id,
            case=case,
            frames=frames,
            events=(),
            decisions=count,
            learner_score=1,
            opponent_score=0,
            objective_completed=True,
            terminated=True,
            truncated=False,
            peer_tic_lag_max=0,
            protocol_inconsistent=False,
            action_tic_inconsistent=False,
            score_event_inconsistent=False,
            environment_attempts=1,
            scenario_hash="91569d20cd52844cfa31284fe8df2886b3d8f2860bacfb6070c5d828511a7cb8",
        )

    def fake_write_mp4(frames: object, output: Path, fps: int) -> Path:
        del fps
        videos[output] = tuple(frames)  # type: ignore[arg-type]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"MP4")
        return output

    def fake_write_gif(
        frames: object, output: Path, *, fps: int, max_bytes: int
    ) -> Path:
        del fps, max_bytes
        captured = tuple(frames)  # type: ignore[arg-type]
        gif_frames.extend(captured)
        videos[output] = captured
        output.write_bytes(b"GIF89a")
        return output

    monkeypatch.setattr(showcase_cli, "record_showcase_episode", fake_record)
    monkeypatch.setattr(showcase_cli, "write_mp4", fake_write_mp4)
    monkeypatch.setattr(showcase_cli, "read_video_frames", lambda path: videos[path])
    monkeypatch.setattr(showcase_cli, "write_gif", fake_write_gif)

    manifest = showcase_cli.render_showcase(
        root=root,
        config_path=config.config_path,
        checkpoint_root=tmp_path,
        device="cpu",
    )

    assert manifest["publication"] is False
    assert manifest["official_test_result"] is False
    assert (tmp_path / "media/development-ppo.mp4").is_file()
    assert (tmp_path / "media/development-bc.mp4").is_file()
    assert (tmp_path / "media/development-comparison.gif").is_file()
    assert len(gif_frames) == 2
    assert "docs/assets/showcase" not in json.dumps(manifest)
