# Crystal Run: Extraction v2

## Outcome

Extraction v2 is a separate, playable ViZDoom scenario built to make the Bot
decision loop legible in one first-person replay:

```text
search -> fight or evade -> manage scarce loot -> extract -> bank value
```

The public release contains four real validation replays. Every clip is
30--32 seconds and uses a viewer-only telemetry overlay; none uses test cases
or a scripted animation.

| Bot | Selected replay | Visible outcome |
|---|---|---|
| Strong | seed 34001 | survives combat pressure and extracts 85 |
| Aggressive | seed 35005 | five hits, kill, corpse-cache pickup, extracts 30 |
| Defensive | seed 34005 | zero attack decisions, extracts 45 at full health |
| Explorer | seed 34001 | 30 route cells, real 10-to-50 backpack upgrade, extracts 85 |

The [release manifest](../../reports/extraction-v2/showcase/manifest.json)
binds 21 scenario, selection, evidence, media, and specification artifacts by
SHA-256. Run `PYTHONPATH=src python scripts/audit_extraction_release.py` to
verify it.

## Scenario rules

- Two neutral extraction zones open at 30 seconds and require a three-second
  uninterrupted hold.
- Both players start at 100 HP with 30 rounds. Each valid hit deals exactly
  20 damage; there is no reload or respawn.
- Seven world items have values 10, 25, or 50. A three-slot backpack
  deterministically replaces its oldest minimum-value item when a better item
  is collected.
- Death creates a corpse cache containing the victim's individual backpack
  slots. Kills score nothing by themselves.
- Only value banked through a successful extraction determines the result.

The X0 integration gate proves the full 100→80→60→40→20→0 health trace,
corpse-cache value conservation, two extraction zones, backpack replacement,
and a kill→loot→extract replay in real ViZDoom. Its scenario WAD hash is
`bc76e895035cc9fa76e86e2f3b22e605254f6a70e7f33b37876228f9c6c2c42e`.

## Learning and observation boundary

Each policy is a recurrent visual Actor trained from 60,000 style-specific
training transitions with a separate 12,000-transition validation split.
The Actor sees only first-person grayscale pixels plus its own HP, ammunition,
backpack summary, extraction status, banked value, remaining time, and
previous action. Opponent HP, coordinates, loot coordinates, region IDs, and
viewer telemetry never enter the Actor.

Strong, Defensive, and Explorer are learned checkpoints. Aggressive uses a
public-observation capability governor: the learned Aggressive checkpoint
handles encounter behavior, then a separately trained extraction finisher
takes over after meaningful carried value is observed. Both recurrent models
receive exactly the same legal public observation stream.

## Honest evidence boundary

This milestone demonstrates a complete engineering and product showcase, not
a statistically conclusive benchmark claim. Selection used a frozen
validation candidate manifest. The Aggressive search stopped at the first
replay satisfying the predefined five-hit→kill→cache→extraction predicate and
records all five cases evaluated.

The initial monolithic Aggressive checkpoint, a bounded DAgger correction,
and an earlier governor did not reliably convert post-combat loot into
extraction. Their reports remain committed. The finisher plus frozen candidate
search produced the accepted replay, but no official held-out test result is
claimed.
