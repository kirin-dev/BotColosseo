import hashlib
import json
import sys

import pytest

from botcolosseo.cli.decide_hierarchical_executor_promotion import main
from botcolosseo.training.hierarchical_protocol import Command


def write_report(path, *, improved=False):
    report = {
        "complete": True, "profile": "test", "protocol": "test",
        "command_schema": "test", "scoring_version": "test", "extraction_endpoint": None,
        "cases": [{
            "seed": seed, "side": "host", "scenario_hash": "test", "banked": 50,
            "commands": [{
                "command": command.name, "start": int(command) * 20,
                "completed_at": int(command) * 20 + 10
                if seed == 0 or (improved and command == Command.SEARCH_NORTH) else None,
                "applicable": True,
            } for command in Command],
        } for seed in (0, 1)],
    }
    path.write_text(json.dumps(report))


def test_cli_binds_inputs_and_cannot_authorize_from_small_screen(tmp_path, monkeypatch):
    baseline, candidate, output = [tmp_path / n for n in ("old.json", "new.json", "out.json")]
    write_report(baseline)
    write_report(candidate, improved=True)
    monkeypatch.setattr(sys, "argv", ["promotion", "--baseline", str(baseline),
        "--candidate", str(candidate), "--weak-command", "SEARCH_NORTH", "--output", str(output)])
    main()
    result = json.loads(output.read_text())
    assert result["decision"]["screening_passed"]
    assert result["decision"]["promote"] is False
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert result["source_sha256"]["candidate"] == candidate_hash
    previous = output.read_bytes()
    with pytest.raises(FileExistsError):
        main()
    assert output.read_bytes() == previous


def test_cli_incomplete_report_cannot_write_decision(tmp_path, monkeypatch):
    baseline, candidate, output = [tmp_path / n for n in ("old.json", "new.json", "out.json")]
    write_report(baseline)
    candidate.write_text(json.dumps({"complete": False, "cases": []}))
    monkeypatch.setattr(sys, "argv", ["promotion", "--baseline", str(baseline),
        "--candidate", str(candidate), "--weak-command", "SEARCH_NORTH", "--output", str(output)])
    with pytest.raises(ValueError):
        main()
    assert not output.exists()
