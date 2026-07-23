from __future__ import annotations

import json

from botcolosseo.cli.evaluate_hybrid_difficulty import main


def test_hybrid_difficulty_preflight_freezes_200_validation_episodes(
    capsys,
) -> None:
    assert (
        main(
            [
                "--style",
                "defensive",
                "--output-dir",
                "unused",
                "--preflight",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == "m5-hybrid-defensive-difficulty-extension"
    assert payload["difficulties"] == ["easy", "normal"]
    assert payload["expected_episodes"] == 200
    assert payload["test_cases_accessed"] is False
    assert payload["preflight_passed"] is True
