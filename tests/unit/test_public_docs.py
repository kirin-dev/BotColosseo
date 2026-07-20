from pathlib import Path

PUBLIC_DOCS = (
    Path("README.md"),
    Path("docs/milestones/m0.md"),
    Path("docs/milestones/m1.md"),
    Path("docs/milestones/m2.md"),
    Path("assets/scenarios/crystal_run/README.md"),
)


def test_public_documentation_has_no_machine_specific_home_paths() -> None:
    violations = {
        str(path): line
        for path in PUBLIC_DOCS
        for line in path.read_text(encoding="utf-8").splitlines()
        if "/home/" in line or "wencong@" in line
    }

    assert violations == {}
