from pathlib import Path

from botcolosseo.cli.evaluate_native_style_difficulty import build_parser


def test_native_style_difficulty_cli_freezes_style_and_inputs() -> None:
    args = build_parser().parse_args(
        [
            "--style",
            "explorer",
            "--base-checkpoint",
            "base.pt",
            "--style-checkpoint",
            "explorer.pt",
            "--output-dir",
            "reports/explorer",
        ]
    )

    assert args.style == "explorer"
    assert args.base_checkpoint == Path("base.pt")
    assert args.style_checkpoint == Path("explorer.pt")
    assert args.pairs_per_opponent == 10
    assert args.bootstrap_samples == 10_000
