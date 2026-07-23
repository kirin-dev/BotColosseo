# M6 Readable User-Study Clips Design

## Status

Approved by the project owner on 2026-07-23. This design replaces the
one-clip-per-style study package. It does not replace the existing GitHub
showcase or any formal M4/M5 evaluation.

## Problem diagnosis

The existing blind clips are 14.4, 26.8, and 22.5 seconds. Their shared
`defensive_script:458:host` replay is a poor perception-study case:

- the Aggressive replay contains three attacks but zero valid hits;
- none of the four policy replays records an opponent death;
- the recorded `DROP` events do not prove a hit-to-death-to-drop sequence;
- opponent health and other neutral causal feedback are not visible.

This is primarily a selection and presentation failure, not evidence that the
Aggressive policy never attacks. In its 100 formal validation episodes,
Aggressive attacks in 42 episodes and lands valid hits in 25; it produces 59
valid hits from 221 attacks, with a maximum of eight in one episode. Strong
Base records zero attacks and zero valid hits on the same cases.

The formal hybrid ledgers also contain usable non-combat mechanisms:

- Defensive: 23 episodes with a successful escape and six with recovery;
- Explorer: 76 episodes with completed routes, 14 with multiple route types,
  and 52 covering all six regions.

Defensive records no carrier denial in its formal ledger. The study therefore
must not imply that a Defensive clip demonstrates a kill-induced core drop.

## Study unit

The revised package contains six opaque clips:

- two Aggressive;
- two Defensive;
- two Explorer.

Each respondent watches all six in a deterministic counterbalanced order.
The answer key remains private until collection closes. Each clip is a real
validation replay from the exact published policy artifact and frozen scenario.

The target duration is 25--40 seconds. A shorter clip is accepted only if the
episode terminates naturally after the required visible mechanism. Longer
episodes are cropped around the representative mechanism. The final clips are
explicitly described as curated validation showcases, not random performance
samples.

## Frozen candidate selection

Candidate ranking uses committed formal ledgers to produce a small high-signal
shortlist. The shortlist is then replayed and visually reviewed. The final two
clips per style are the clearest representative demonstrations, not merely the
first rows in a numeric ordering. Selection records retain every shortlisted
case, its formal metrics, replay events, rejection reason, and the final
curation decision.

### Aggressive

Rank eligible objective-complete episodes by:

1. valid hits, descending;
2. engagement initiations, descending;
3. attack decisions, descending;
4. stable case ID.

The two selected clips must jointly provide:

- at least three valid hits in each replay;
- at least one replay with six or more valid hits;
- at least one visible opponent death if the replay reproduces one.

A replay with a visible opponent death is preferred over a numerically higher
hit count with poor visibility. A kill-induced core drop is preferred but not
required. If no shortlisted replay produces the causal sequence, the
documentation states that limitation rather than claiming it.

### Defensive

Rank eligible objective-complete episodes by:

1. low-health opportunities;
2. recovery present;
3. successful escapes;
4. risk decisions;
5. stable case ID.

This ordering reflects a replay audit performed during implementation:
`successful_escapes` records a triggered escape process, but does not guarantee
that the episode later remains death-free. Final visual review therefore checks
the replay's actual health and death events rather than treating that formal
counter as a survival label.

The pair must contain a visually legible recovery episode and an escape-heavy
episode where possible. Carrier denial is not a selection requirement because
the formal ledger contains none.

### Explorer

Rank eligible objective-complete episodes by:

1. number of completed route types;
2. route entropy;
3. completed routes;
4. unique regions;
5. stable case ID.

The pair must jointly show at least Upper and Lower route behavior, make the
route contrast visible from the first-person view, and use two different
opponents or learner sides where available.

## Observer-only HUD

Every frame receives the same neutral overlay:

- `SELF HP`;
- `OPP HP`;
- `CORE: SELF / OPP / FREE`;
- `SCORE`;
- the learner macro action;
- neutral public events: `HIT`, `PICKUP`, `DROP`, `DEATH`, `RESPAWN`, `SCORE`.

The overlay never displays policy/style labels, governor states, route mode,
reward, privileged coordinates, regions, or a suggested interpretation.
Opponent health is viewer-only telemetry obtained after the policy action. It
is not passed to the Actor. Public documentation must say:

> Observer-only telemetry; not available to the policy.

The manifest records this boundary and hashes every source artifact, selected
case, replay trace, and output clip.

## Mobile study instructions

Participants are told before viewing:

- Crystal Run is a 1v1 capture-and-return objective;
- the video is the evaluated Bot's first-person view;
- the visible character is the opponent;
- movement, camera turns, route choice, and firing are the evaluated Bot's
  actions;
- repeated valid hits reduce opponent HP, death causes respawn, and a carrier
  drops the core;
- style, clarity, and perceived difficulty refer to the first-person Bot.

Participants may choose the same style more than once or `unsure`.

## Acceptance gates

The revised package passes only if:

- it contains exactly six distinct clips and two clips per true style;
- all source artifacts and validation cases are hash-bound;
- the manifest distinguishes curated showcase selection from random sampling;
- every replay is protocol-clean and has no test access;
- clips satisfy the frozen per-style mechanism rules;
- duration is 25--40 seconds unless naturally terminated earlier;
- observer HUD fields are complete and explicitly excluded from policy input;
- assignments contain each clip exactly once per respondent;
- analysis still reports per-style and aggregate recognition with Wilson
  intervals;
- automated tests reject missing clips, answer leakage, incomplete responses,
  HUD provenance drift, and altered media.

## Non-goals

- No policy retraining or reward change.
- No hand-authored combat animation or fabricated event.
- No fixed number-of-shots-to-kill claim.
- No style labels in blind media.
- No relabeling of Defensive or Explorer as learned policies.
