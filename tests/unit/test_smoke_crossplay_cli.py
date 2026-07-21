import pytest

from botcolosseo.cli.smoke_crossplay import build_parser


def test_smoke_crossplay_parser_requires_checkpoint_and_positive_pairs() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    parsed = parser.parse_args(["--checkpoint", "policy.pt"])
    assert parsed.pairs == 1
    assert parsed.max_decisions == 525
