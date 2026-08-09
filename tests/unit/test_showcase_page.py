from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path


class _ShowcaseParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.sources: list[str] = []
        self.images: list[str] = []
        self.links: list[str] = []
        self.video_count = 0
        self.posters: list[str] = []
        self.script_count = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if identifier := attributes.get("id"):
            self.ids.add(identifier)
        if tag == "video":
            self.video_count += 1
            if poster := attributes.get("poster"):
                self.posters.append(poster)
        elif tag == "source" and (source := attributes.get("src")):
            self.sources.append(source)
        elif tag == "img" and (source := attributes.get("src")):
            self.images.append(source)
        elif tag == "a" and (target := attributes.get("href")):
            self.links.append(target)
        elif tag == "script":
            self.script_count += 1


def _parse_showcase() -> tuple[str, _ShowcaseParser]:
    source = Path("docs/index.html").read_text(encoding="utf-8")
    parser = _ShowcaseParser()
    parser.feed(source)
    return source, parser


def test_showcase_has_four_playable_style_videos_and_matching_emotes() -> None:
    source, parser = _parse_showcase()

    assert parser.video_count == 4
    assert parser.sources == [
        "assets/extraction/strong.mp4",
        "assets/extraction/aggressive.mp4",
        "assets/extraction/defensive.mp4",
        "assets/extraction/explorer.mp4",
    ]
    assert parser.images[:4] == [
        "assets/extraction/emotes/Strong.jpg",
        "assets/extraction/emotes/Aggressive.jpg",
        "assets/extraction/emotes/Defensive.jpg",
        "assets/extraction/emotes/Explorer.jpg",
    ]
    assert parser.posters == [
        "assets/extraction/strong-poster.jpg",
        "assets/extraction/aggressive-poster.jpg",
        "assets/extraction/defensive-poster.jpg",
        "assets/extraction/explorer-poster.jpg",
    ]
    for asset in parser.sources + parser.images + parser.posters:
        assert (Path("docs") / asset.split("?", 1)[0]).is_file()
    assert source.count('width="80"') == 4
    assert source.count('height="80"') == 4


def test_showcase_opens_with_compact_botcolosseo_identity_and_video_grid() -> None:
    source, _ = _parse_showcase()

    assert "<title>BotColosseo · Controllable Game Bots for SFE</title>" in source
    assert 'href="showcase.css?v=randomized-1"' in source
    assert "<h1>BotColosseo Controllable Game Bots for SFE</h1>" in source
    assert 'class="github-link"' in source
    assert '<header class="site-header">' in source
    assert '<nav aria-label="Primary navigation">' in source
    assert all(
        f'href="#{target}"' in source
        for target in ("scenario", "method", "styles", "results")
    )
    assert "VIZDOOM · REINFORCEMENT LEARNING" not in source
    assert "Search · Fight · Extract" not in source
    assert "One capable visual Bot" not in source
    assert "Evidence ↗" not in source
    assert "Crystal Run" not in source

    hero = source.index('<section class="hero shell">')
    bot_grid = source.index('<section class="bot-showcase shell" id="bots"')
    scenario = source.index('<section class="section shell" id="scenario">')
    assert hero < bot_grid < scenario


def test_showcase_sections_use_numbered_labels_without_secondary_titles() -> None:
    source, _ = _parse_showcase()
    stylesheet = Path("docs/showcase.css").read_text(encoding="utf-8")

    assert "01 · SCENARIO" in source
    assert "02 · METHOD" in source
    assert "03 · STYLES" in source
    assert "04 · RESULTS" in source
    assert "05 ·" not in source
    assert "Search, fight, extract" not in source
    assert "One base, three learned adapters" not in source
    assert "Different priorities, visible decisions" not in source
    assert "Capability first, style second" not in source
    assert ".section-heading h2" not in stylesheet
    assert "scroll-margin-top: 52px" in stylesheet


def test_showcase_footer_keeps_only_back_to_top_link() -> None:
    source, _ = _parse_showcase()
    footer = source[source.index('<footer class="shell">') :]

    assert footer.count("<a ") == 1
    assert '<a href="#top">Back to top ↑</a>' in footer
    assert "Repository" not in footer
    assert "中文说明" not in footer


def test_showcase_uses_fluid_viewport_gutters_for_chrome_and_content() -> None:
    stylesheet = Path("docs/showcase.css").read_text(encoding="utf-8")

    assert "--page-gutter: clamp(12px, 2.5vw, 48px)" in stylesheet
    assert "margin-inline: var(--page-gutter)" in stylesheet
    assert "padding-inline: var(--page-gutter)" in stylesheet
    assert "width: min(1180px" not in stylesheet
    assert "footer {\n    flex-direction: column" not in stylesheet


def test_mobile_navigation_keeps_full_section_names() -> None:
    source, _ = _parse_showcase()
    stylesheet = Path("docs/showcase.css").read_text(encoding="utf-8")

    assert all(
        label in source
        for label in ("01 Scenario", "02 Method", "03 Styles", "04 Results")
    )
    assert "overflow-x: auto" in stylesheet
    assert "01</a>" not in source


def test_showcase_title_has_responsive_vertical_space() -> None:
    stylesheet = Path("docs/showcase.css").read_text(encoding="utf-8")

    assert "clamp(84px, 8vw, 88px)" in stylesheet
    assert "clamp(42px, 5vw, 48px)" in stylesheet


def test_showcase_declares_resolvable_favicon_assets() -> None:
    source, _ = _parse_showcase()
    favicon_assets = {
        "assets/favicon-16x16.png": b"\x89PNG\r\n\x1a\n",
        "assets/favicon-32x32.png": b"\x89PNG\r\n\x1a\n",
        "assets/apple-touch-icon.png": b"\x89PNG\r\n\x1a\n",
    }

    assert 'rel="apple-touch-icon"' in source
    for asset, signature in favicon_assets.items():
        path = Path("docs") / asset
        assert f'href="{asset}"' in source
        assert path.read_bytes().startswith(signature)


def test_showcase_covers_scenario_method_styles_and_results() -> None:
    source, parser = _parse_showcase()

    assert {"bots", "scenario", "method", "styles", "results"} <= parser.ids
    assert "assets/extraction/map.svg" in parser.images
    assert "assets/extraction/method.svg?v=1200x600-1" in parser.images
    assert all(
        text in source
        for text in (
            "100 HP",
            "20 damage",
            "3 slots",
            "Scripted Teacher",
            "Behavioral Cloning",
            "Evidence tier",
            "Selected causal chain",
            "official-test results",
        )
    )

    map_svg = Path("docs/assets/extraction/map.svg").read_text(encoding="utf-8")
    method_svg = Path("docs/assets/extraction/method.svg").read_text(encoding="utf-8")
    assert "16 SAFE LOOT ANCHORS" in map_svg
    assert "7 ITEMS SAMPLED PER RAID" in map_svg
    assert "opportunity-conditioned PBRS" in method_svg
    assert "partitioned KL" in method_svg
    assert "training-only" in method_svg
    assert 'width="1200" height="600"' in method_svg
    assert 'd="M857 198 V246 H606 V269"' in method_svg
    assert 'width="1400" height="520"' not in method_svg
    public_copy = (source + map_svg + method_svg).lower()
    assert "crystal run" not in public_copy
    assert "base layout" not in public_copy
    assert re.search(r"\bv3\b", public_copy) is None


def test_styles_and_results_describe_the_current_randomized_pipeline() -> None:
    source, _ = _parse_showcase()
    training_curve = Path("docs/assets/extraction/training-curve.svg").read_text(
        encoding="utf-8"
    )

    assert "Three bounded adapters share the same frozen Strong Actor" in source
    assert "Training-only opportunity-conditioned PBRS" in source
    assert "Randomized-loot capability and representative style evidence" in source
    assert "Candidate selection never accesses the test split" in source
    assert "the style adapters' partitioned KL is not plotted here" in source
    assert "B · Strong PPO diagnostics" in training_curve
    assert "KL to frozen BC reference" in training_curve
    assert "randomized-lineage" not in source
    assert "Skill-retention diagnostics" not in training_curve
    assert "frozen-reference KL" not in training_curve


def test_showcase_is_dependency_free_and_all_local_assets_resolve() -> None:
    _, parser = _parse_showcase()

    assert parser.script_count == 0
    local_targets = [
        target.split("?", 1)[0]
        for target in parser.images + parser.sources + parser.posters
        if "://" not in target
    ]
    assert all((Path("docs") / target).is_file() for target in local_targets)

    stylesheet = Path("docs/showcase.css").read_text(encoding="utf-8")
    assert "color-scheme: light" in stylesheet
    assert "--background: #ffffff" in stylesheet
    assert "width: 80px" in stylesheet
    assert ".site-header" in stylesheet
    assert "position: sticky" in stylesheet
    assert "@media (max-width: 860px)" in stylesheet
    assert "@media (max-width: 620px)" in stylesheet
    assert "@media (prefers-reduced-motion: reduce)" in stylesheet
