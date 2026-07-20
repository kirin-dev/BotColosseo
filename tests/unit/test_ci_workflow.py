from pathlib import Path


def test_github_ci_runs_portable_quality_gate() -> None:
    workflow = Path(".github/workflows/quality.yml").read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert 'python-version: "3.10"' in workflow
    assert "https://download.pytorch.org/whl/cpu" in workflow
    assert 'pip install -e ".[training,dev]"' in workflow
    assert "python -m ruff check src scripts tests" in workflow
    assert "python -m pytest -q tests/unit" in workflow
    assert "tests/integration" not in workflow
