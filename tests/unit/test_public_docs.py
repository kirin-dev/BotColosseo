import re
from pathlib import Path

PUBLIC_DOCS = (
    Path("README.md"),
    Path("README_CN.md"),
    Path("Plan.md"),
    Path("script.md"),
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


def test_readme_publishes_clean_product_and_evidence_boundary() -> None:
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
    assert "v3" not in readme.lower()
    assert "## Technical evolution" in readme
    assert "Fixed loot → randomized loot" in readme
    assert "Global style rewards → opportunity-conditioned shaping" in readme


def test_readme_preserves_fair_actor_and_learned_style_boundary() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "The Actor never receives opponent HP" in readme
    assert "asymmetric training Critic and reward shaping" in readme
    assert "offline" in readme
    assert "evaluation and viewer telemetry" in readme
    assert "none of it enters the deployed Actor" in readme
    assert "same frozen Strong" in readme
    assert "Training-only opportunity detectors" in readme
    assert "deployed" in readme
    assert "makes no PFSP-training or" in readme
    assert "one frozen 400-episode official test per policy" in readme


def test_chinese_readme_preserves_pending_and_test_boundaries() -> None:
    readme = Path("README_CN.md").read_text(encoding="utf-8")

    assert "当前公开结论限定为产品 Showcase" in readme
    assert "候选选择阶段禁止访问 test" in readme
    assert "同一个冻结 Strong Actor 哈希" in readme
    assert "仅训练期使用机会检测器" in readme
    assert "部署策略仍只" in readme
    assert "不声称其因果增益" in readme
    assert "## 技术路线演进" in readme
    assert "固定物资 → 随机物资" in readme
    assert "全局风格奖励 → 机会条件化塑形" in readme
    assert "不会进入" in readme
    assert "部署 Actor" in readme
    assert "official test 总计 1,600 局" in readme


def test_pages_scopes_metrics_and_matches_showcase_evidence() -> None:
    page = Path("docs/index.html").read_text(encoding="utf-8")

    assert "four scripted opponent styles × paired learner sides" in page
    assert "240 validation episodes and 120 heldout episodes" in page
    assert "same finite randomized-layout family" in page
    assert "corpse cache → 100 value" in page
    assert "backpack upgrade → 70 value" in page
    assert "corpse cache → 85 value" not in page
