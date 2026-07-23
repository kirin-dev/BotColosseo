from pathlib import Path

from botcolosseo.cli.evaluate_defensive import build_parser


def test_defensive_cli_freezes_formal_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--base-checkpoint",
            "base.pt",
            "--defensive-checkpoint",
            "defensive.pt",
            "--output-dir",
            "reports/m5/defensive/formal",
        ]
    )

    assert args.base_checkpoint == Path("base.pt")
    assert args.defensive_checkpoint == Path("defensive.pt")
    assert args.cases == Path("configs/m3/validation.json")
    assert args.pairs_per_opponent == 10
    assert args.max_attempts == 2
