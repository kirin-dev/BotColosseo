# GitHub Pages Training Curves Design

**Date:** 2026-08-08
**Status:** Approved design; implementation deferred until the randomized-loot
Strong and three style policies share one validated release lineage.

## Objective

Add one compact, evidence-backed training-and-selection figure to the Results
section of GitHub Pages. The figure should show why the 950k Strong checkpoint
was selected and how Conservative PPO preserved the BC skill prior. It must not
displace the four style videos from the first viewport or imply that on-policy
training statistics are held-out evaluation results.

## Release gate

Do not publish the figure independently. Update Pages only after Aggressive,
Defensive, and Explorer have been trained and evaluated from the selected
randomized-loot Strong checkpoint. The refreshed page must bind the Strong
metrics, style metrics, videos, reports, and checkpoint hashes to that same
release lineage.

If the new style policies are rejected, keep the current public Showcase and do
not mix the randomized Strong curve with videos derived from the older Strong
checkpoint.

## Figure

Place a single two-panel figure inside the existing `04 Results` section, below
the capability summary and above detailed evidence notes.

### Panel A: capability selection

- X-axis: environment steps, 50k through 1M.
- Primary line: extraction rate from the frozen 32-episode randomized screening
  protocol for all 20 checkpoints.
- Optional secondary line: screening win rate, visually lighter than extraction.
- Highlight the selected 950k checkpoint.
- Add the selected checkpoint's 240-episode randomized result as a distinct
  confirmation marker, not as another point on the 32-episode line.
- Caption the two sample budgets explicitly: `32-episode screening` and
  `240-episode confirmation`.

The plot should make the non-monotonic selection result readable: 950k was
selected by validation, while the terminal 1M checkpoint was not assumed best.

### Panel B: skill retention

- X-axis: environment steps in the same 50k bins.
- Show mean offline BC replay agreement as a percentage.
- Show mean frozen-reference KL on a separate axis or separate aligned subplot;
  never place two differently scaled values on one unlabeled axis.
- Annotate that both are training diagnostics, not task success metrics.

Do not publish total loss, policy loss, entropy, gradient norm, or raw Teacher
loss on the product-facing page. They remain available in machine-readable logs
but add clutter without improving the core project story.

## On-policy training statistics

The 2,585 on-policy training episodes may be rendered as a muted contextual line
or omitted. If included, aggregate them into fixed 50k-step windows and label
them `on-policy training opponent pool`. Do not call this line validation,
generalization, or a convergence guarantee. The page's capability claim must be
anchored to frozen screening and confirmation protocols.

## Data provenance

Generate the chart deterministically from:

- `runs/extraction-randomized/strong-ppo-conservative-v2/metrics.jsonl`;
- `reports/extraction/conservative-strong-1m-selection/screening-32/*.json`;
- `reports/extraction/conservative-strong-1m-selection.json`.

The generated chart data must record source SHA-256 values, the selected
checkpoint SHA-256, aggregation window, metric definitions, and
`test_cases_accessed=false`. The public repository should contain the compact
derived chart data and SVG, not private checkpoints or full raw training logs.

## Page hierarchy and responsive behavior

- Preserve the current white background, horizontal title, and four-video grid
  as the first visual focus.
- Keep the figure within the existing responsive content width.
- Desktop: two panels side by side.
- Narrow screens: stack the panels without shrinking labels below readable size.
- Use existing typography, borders, and accent colors; do not add a new visual
  system or charting runtime dependency.
- Prefer a deterministic static SVG plus concise accessible alt text.

## Copy boundary

Acceptable public claim:

> Frozen randomized validation selected the 950k checkpoint rather than the
> terminal 1M checkpoint; Conservative PPO retained high BC replay agreement
> while improving closed-loop extraction over the earlier randomized baseline.

Do not claim monotonic convergence, continuous-placement generalization,
official-test success, or causal superiority of every Conservative PPO
component. The 32-episode line is a screening curve; the 240-episode result is
the stronger confirmation estimate.

## Verification

Before publishing:

1. Recompute every plotted point from the frozen source reports.
2. Verify selected checkpoint and source hashes.
3. Confirm no official test data was accessed.
4. Check that new Strong and style artifacts share the intended base lineage.
5. Render at desktop, tablet, and mobile widths.
6. Confirm the four-video grid remains visible before the Results section.
7. Run existing unit checks and the Pages asset/link audit.
