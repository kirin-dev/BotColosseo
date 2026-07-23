import re
from pathlib import Path

PUBLIC_DOCS = (
    Path("README.md"),
    Path("README_CN.md"),
    Path("docs/milestones/m0.md"),
    Path("docs/milestones/m1.md"),
    Path("docs/milestones/m2.md"),
    Path("assets/scenarios/crystal_run/README.md"),
    Path("THIRD_PARTY_NOTICES.md"),
)

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


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


def test_public_documentation_local_links_resolve() -> None:
    broken = []
    for document in PUBLIC_DOCS:
        for target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            relative = target.split("#", 1)[0]
            if not relative or "://" in relative or relative.startswith("mailto:"):
                continue
            resolved = (document.parent / relative).resolve()
            if not resolved.exists():
                broken.append((str(document), target))

    assert broken == []


def test_readme_publishes_m4_media_with_validation_boundary() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "m4-base-vs-aggressive.gif" in readme
    assert "reports/showcase/m4/manifest.json" in readme
    assert re.search(r"not an\s+official test result", readme)
    assert "M4 passed" not in readme


def test_chinese_readme_preserves_failed_gate_and_validation_boundaries() -> None:
    readme = Path("README_CN.md").read_text(encoding="utf-8")

    assert "m4-base-vs-aggressive.gif" in readme
    assert "不是 official test 结果" in readme
    assert "M2 真实同步 1v1 与初始 PPO | FAIL" in readme
    assert "M3 historical/PFSP Strong Base | 未通过全部能力门" in readme
    assert "M5 Defensive / Explorer / Difficulty | 进行中" in readme
