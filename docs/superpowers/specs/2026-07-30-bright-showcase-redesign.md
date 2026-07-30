# BotColosseo Bright Showcase Redesign

## Goal

Redesign the GitHub Pages landing page as a concise, bright research-project
showcase inspired by the information hierarchy of NVIDIA GEAR project pages.
The first viewport must establish the BotColosseo identity and immediately make
the four learned Bot styles visible.

This is a presentation change only. It must not alter videos, reported metrics,
evidence boundaries, game rules, or the static deployment model.

## Identity

The page uses one centered, non-wrapping title:

```text
BotColosseo Controllable Game Bots for SFE
```

It uses the former `Controllable Game Bots for SFE` subtitle size and weight,
not the original oversized `BotColosseo` hero treatment. Responsive typography
keeps the title on one line at desktop and mobile widths.

The hero contains no expanded SFE label and no descriptive sentence. The full
original research phrasing may appear later in supporting text, but `Crystal
Run` is not used as the project name, page title, brand, or footer identity.

## First Viewport

Use a vertically symmetric research-page layout rather than a left/right hero.

1. A compact centered introduction contains only the small ViZDoom eyebrow and
   the single-line project title.
2. The full-width 2×2 style-video grid follows immediately. No navigation bar,
   oversized hero copy, section heading, or explanatory block may intervene.
3. At a common desktop viewport, the title and first video row must be
   completely visible, with the second row visibly continuing the same grid.

A single lightweight `GitHub ↗` text link sits in the upper-right corner.
There is no Evidence button or second hero action.

The four cards remain Strong, Aggressive, Defensive, and Explorer. Each card
retains its first-person video and concise causal chain. The style emote grows
from the current small avatar to roughly 80 px and is visually prominent beside
the style name.

Videos remain user-controlled, muted, and non-autoplaying.

## Visual System

- White page background with near-black primary text.
- Soft gray secondary text, borders, and section dividers.
- Flat cards with subtle shadows or tonal separation; no dark gradients or
  glass navigation.
- Limited accent color: one restrained site accent plus the existing per-style
  colors.
- Centered content column with balanced whitespace, closer to a research page
  than a product-marketing landing page.
- Section order after the style grid:
  Scenario, Method, How to Tell, Results.
- Those sections are numbered `01` through `04`. Their headings use smaller
  responsive type and remain on one line at all supported widths.

## Responsive Behavior

- Desktop and tablet: 2×2 video grid.
- Narrow mobile: single-column video cards after the compact centered title.
- Emotes remain prominent on mobile instead of shrinking back to icon size.
- The single-line project and section headings shrink responsively rather than
  wrapping or overflowing.
- Tables keep horizontal overflow behavior and all local media remain playable.

## Evidence and Technical Boundaries

- Keep the page dependency-free and JavaScript-free.
- Preserve all existing local video, image, report, and repository links.
- Preserve fair-observation language and failed-gate disclosures.
- Add the matched style-ablation table only after the running experiments
  produce audited values; the redesign must leave a natural Results location
  for it without placeholders or invented data.

## Verification

- Update page tests to reject the old Crystal Run identity and top navigation.
- Verify the exact single-line title and all four video/emote assets.
- Verify the absence of the former SFE expansion, descriptive sentence, and
  Evidence action.
- Verify the upper-right GitHub link and sequential `01`–`04` section labels.
- Render desktop and mobile screenshots locally and inspect the first viewport,
  card hierarchy, overflow, and readability.
- Run public-document link checks, unit tests, Ruff, and `git diff --check`.
- After publication, verify the deployed GitHub Pages HTML, stylesheet, and
  representative media links.

## Completion

The redesign is complete when the live page is bright, vertically symmetric,
uses the BotColosseo identity, has no top navigation, exposes the 2×2 video grid
immediately below the compact single-line title, shows larger emotes, presents
only GitHub in the upper-right, keeps all section headings unwrapped, preserves
all evidence claims, and passes desktop/mobile and repository verification.
