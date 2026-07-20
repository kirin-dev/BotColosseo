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


def test_freedoom_rendered_assets_ship_the_required_bsd_notice() -> None:
    notice = Path("licenses/FREEDOOM-BSD-3-CLAUSE.txt").read_text(encoding="utf-8")
    third_party = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert "Copyright © 2001-2024 Contributors to the Freedoom project" in notice
    assert "Redistribution and use in source and binary forms" in notice
    assert "Neither the name of the Freedoom project" in notice
    assert "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS" in notice
    assert "licenses/FREEDOOM-BSD-3-CLAUSE.txt" in third_party
