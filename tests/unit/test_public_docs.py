import re
from pathlib import Path

PUBLIC_DOCS = (
    Path("README.md"),
    Path("README_CN.md"),
    Path("Plan.md"),
    Path("script.md"),
    Path("assets/scenarios/crystal_run_extraction/README.md"),
    Path("docs/review/2026-07-26-heldout-layout-approval.md"),
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


def test_readme_publishes_clean_v3_product_and_evidence_boundary() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "# BotColosseo" in readme
    assert "Controllable Game Bots for Search-Fight-Extract" in readme
    assert "100 HP" in readme
    assert "20 damage" in readme
    assert "30 rounds" in readme
    assert "three slots" in readme
    assert "no benchmark-success claim" in readme
    assert "Previous Crystal Run" not in readme
    assert "Extraction v2" not in readme


def test_readme_preserves_fair_actor_and_learned_style_boundary() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "The Actor never receives opponent HP" in readme
    assert "During training only" in readme
    assert "same frozen Strong Actor hash" in readme
    assert "no runtime behavior governors" in readme
    assert "one frozen 400-episode official test per policy" in readme


def test_chinese_readme_preserves_pending_and_test_boundaries() -> None:
    readme = Path("README_CN.md").read_text(encoding="utf-8")

    assert "在冻结门通过前，不声称取得了" in readme
    assert "候选选择阶段禁止访问 test" in readme
    assert "同一个冻结 Strong Actor 哈希" in readme
    assert "不包含规则式风格" in readme
    assert "official test 总计 1,600 局" in readme
