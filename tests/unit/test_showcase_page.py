from __future__ import annotations

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
    for asset in parser.sources + parser.images:
        assert (Path("docs") / asset).is_file()
    assert source.count('width="80"') == 4
    assert source.count('height="80"') == 4


def test_showcase_opens_with_compact_botcolosseo_identity_and_video_grid() -> None:
    source, _ = _parse_showcase()

    assert "<title>BotColosseo · Controllable Game Bots for SFE</title>" in source
    assert "<h1>BotColosseo Controllable Game Bots for SFE</h1>" in source
    assert 'class="github-link"' in source
    assert "Search · Fight · Extract" not in source
    assert "One capable visual Bot" not in source
    assert "Evidence ↗" not in source
    assert "<nav" not in source
    assert "site-header" not in source
    assert "Crystal Run" not in source

    hero = source.index('<section class="hero shell">')
    bot_grid = source.index('<section class="bot-showcase shell" id="bots"')
    scenario = source.index('<section class="section shell" id="scenario">')
    assert hero < bot_grid < scenario


def test_showcase_sections_are_sequential_and_headings_do_not_wrap() -> None:
    source, _ = _parse_showcase()
    stylesheet = Path("docs/showcase.css").read_text(encoding="utf-8")

    assert "01 · SCENARIO" in source
    assert "02 · METHOD" in source
    assert "03 · HOW TO TELL" in source
    assert "04 · RESULTS" in source
    assert "05 ·" not in source
    assert ".section-heading h2" in stylesheet
    assert "white-space: nowrap" in stylesheet


def test_showcase_covers_scenario_method_styles_and_results() -> None:
    source, parser = _parse_showcase()

    assert {"bots", "scenario", "method", "styles", "results"} <= parser.ids
    assert "assets/extraction/map.svg" in parser.images
    assert "assets/extraction/method.svg" in parser.images
    assert all(
        text in source
        for text in (
            "100 HP",
            "20 damage",
            "3 slots",
            "Scripted Teacher",
            "Behavioral Cloning",
            "Paired style shift",
            "Task retention",
            "not official-test results",
        )
    )


def test_showcase_is_dependency_free_and_all_local_assets_resolve() -> None:
    _, parser = _parse_showcase()

    assert parser.script_count == 0
    local_targets = [
        target
        for target in parser.images + parser.sources
        if "://" not in target
    ]
    assert all((Path("docs") / target).is_file() for target in local_targets)

    stylesheet = Path("docs/showcase.css").read_text(encoding="utf-8")
    assert "color-scheme: light" in stylesheet
    assert "--background: #ffffff" in stylesheet
    assert "width: 80px" in stylesheet
    assert ".site-header" not in stylesheet
    assert "@media (max-width: 860px)" in stylesheet
    assert "@media (max-width: 620px)" in stylesheet
    assert "@media (prefers-reduced-motion: reduce)" in stylesheet
