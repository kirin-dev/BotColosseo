# Clean Showcase Page Design

**Date:** 2026-07-29
**Status:** approved direction, pending written-spec review

## Goal

Replace the dense repository presentation with a clean, product-facing
Showcase that lets a visitor answer five questions quickly:

1. What do the four Bots look like in play?
2. What is the extraction task?
3. How are the Bots trained and derived?
4. How can the four styles be distinguished?
5. What evidence supports capability and style?

The page is a static GitHub Pages site. The repository README becomes a concise
entry point rather than duplicating the full presentation.

## Page structure

### 1. Hero

Use the project name, one sentence describing the product story, and two small
links: repository and evidence audit. Avoid badges, decorative statistics, and
long research framing above the videos.

### 2. Four playable Bot videos

Place four video cards in a two-column grid and collapse to one column on
mobile. Each card contains:

- a small matching image from `img/`;
- the Bot name and one short style label;
- the actual MP4 with native controls, muted playback, and a poster frame;
- one causal-chain sentence explaining why the behavior matters.

The videos remain the primary visual element. The images are identifiers, not
substitutes for video.

The four causal chains are:

- Strong: search -> collect valuable loot -> extract -> bank value;
- Aggressive: hit -> kill -> loot the corpse cache -> extract;
- Defensive: low HP while carrying loot -> stop pursuing -> disengage ->
  preserve value through extraction;
- Explorer: visit multiple loot regions -> upgrade a full backpack -> extract.

### 3. Scenario model

Show one simplified top-down map for the base layout beside a compact rules
list. The map includes both extraction zones, both spawn regions, low/medium/
high-value loot, and the contested central region.

Only the rules needed to interpret behavior are shown:

- 1v1, 75 seconds, two neutral extraction zones;
- extraction opens at 30 seconds and requires a 3-second hold;
- 100 HP, 20 damage per valid hit, 30 rounds, no respawn;
- three-slot backpack with deterministic low-value replacement;
- death drops unbanked loot into a collectible corpse cache;
- only extracted value counts.

### 4. Method

Use a small horizontal flow:

`Scripted Teacher -> BC -> Recurrent PPO -> frozen Strong Actor`

The frozen Actor then branches to three bounded residual style adapters:
Aggressive, Defensive, and Explorer. A short note preserves the fairness
boundary: the public Actor sees first-person pixels and its own public state;
privileged Critic and reward ledgers are training-only.

### 5. Distinguishing the styles

Use a four-column comparison with three compact fields:

- priority;
- characteristic decision;
- what to watch in the video.

Strong is the balanced task-capable baseline. Aggressive seeks useful combat
conversion. Defensive protects carried value under risk. Explorer prioritizes
useful route and loot diversity rather than movement for its own sake.

### 6. Results

Avoid a large benchmark table. Show only evidence that directly supports the
product story:

- Strong capability: solo extraction, scripted-opponent win rate, validation
  extraction, and heldout-layout extraction;
- style differentiation: paired validation style shift;
- capability preservation: paired task retention.

Evidence tiers and the most important failed generalization checks remain
visible in a short disclosure below the table. The page must not imply that
all styles passed a strict research gate or official test.

## Visual system

- Dark neutral background with one restrained accent per style.
- Maximum content width around 1180 px.
- Rounded cards, subtle borders, generous spacing, and no ornamental
  animation.
- System font stack; no external font or JavaScript dependency.
- Responsive CSS only.
- Accessible labels, visible video controls, keyboard-friendly links, and
  reduced-motion handling.

## Repository layout

- `docs/index.html`: GitHub Pages entry page.
- `docs/showcase.css`: page styles.
- `docs/assets/extraction/`: existing videos and generated diagrams.
- `img/`: user-provided style identifiers.
- `README.md` and `README_CN.md`: concise overview, Showcase link, reproduction
  entry points, and evidence boundary.

No build framework is introduced. All links are relative so the page works
both through GitHub Pages and a local static server.

## Verification

Completion requires:

- all four MP4 elements load from repository assets;
- every style card uses the matching image and causal-chain description;
- scenario, method, style comparison, and results sections are present;
- desktop and mobile layouts are visually inspected;
- no horizontal overflow, missing asset, or broken internal link;
- public documentation tests and the full unit suite pass;
- the evidence numbers match the frozen Showcase reports;
- the page and README preserve split isolation and official-test boundaries.
