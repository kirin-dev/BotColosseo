from __future__ import annotations

import runpy

import pytest


@pytest.mark.parametrize(
    "module",
    (
        "botcolosseo.cli.generate_extraction_demonstrations",
        "botcolosseo.cli.train_extraction_bc",
        "botcolosseo.cli.train_extraction_strong",
        "botcolosseo.cli.train_extraction_style",
        "botcolosseo.cli.evaluate_extraction_v3",
        "botcolosseo.cli.admit_extraction_strong_showcase",
        "botcolosseo.cli.resolve_extraction_strong_artifact",
        "botcolosseo.cli.admit_extraction_showcase",
        "botcolosseo.cli.create_extraction_showcase_demonstration",
        "botcolosseo.cli.resolve_extraction_showcase_artifact",
        "botcolosseo.cli.audit_extraction_showcase",
        "botcolosseo.cli.adopt_extraction_showcase_capture",
        "botcolosseo.cli.summarize_extraction_ablations",
        "botcolosseo.cli.select_randomized_strong_1m",
    ),
)
def test_v3_cli_modules_expose_executable_help(module: str) -> None:
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module(module, run_name="__main__")
    assert exit_info.value.code == 2
