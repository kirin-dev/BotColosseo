# Crystal Run Extraction v2 Proposal

## Status

Future scenario proposal only. The current Crystal Run release remains frozen
and complete under its own evidence boundary. Implementing this proposal
requires a separate design review, new scenario hashes, isolated splits, and
new experiments; no current metric transfers as a v2 result.

## Why change the scenario

The current capture-and-return arena is fast to train and easy to audit, but
its product story is weaker than its engineering:

- scoring often dominates combat;
- a kill, core drop, respawn, and later score can happen too quickly for a
  viewer to understand the causal value transfer;
- there is little accumulated value at risk, so Defensive and Explorer have
  limited room to express meaningful choices;
- a short shared objective produces frequent route overlap and makes styles
  harder to distinguish from first-person video.

The proposed v2 keeps the small, reproducible ViZDoom scope while changing the
player-facing loop to:

```text
INFILTRATE → SEARCH → FIGHT / EVADE → LOOT → EXTRACT
```

## Minimal raid rules

1. Each 60–90 second raid begins with a fixed basic weapon and empty carried
   inventory. There is no persistent economy in the first prototype.
2. Three auditable loot tiers are placed across low-risk, contested, and
   high-risk regions. A small subset is randomized per split.
3. Loot has visible carried value and a bounded slot count. High-value loot
   creates an explicit reason to fight or disengage.
4. A killed player drops all carried loot as one corpse cache. Death ends that
   player's raid; there is no in-raid respawn.
5. The survivor receives a short window to loot the cache and reach one of two
   extraction zones. Extraction requires a visible uninterrupted hold.
6. A raid score is the value successfully extracted. Kills matter only through
   survival, access, and transferred loot; kill farming alone cannot win.

This is intentionally smaller than a full extraction-shooter economy. Stashes,
traders, weapon rarity, insurance, squads, and multi-raid progression are
excluded from the first v2 gate.

## Visible consequence chain

The showcase must make one causal sequence readable without narration:

```text
opponent carries value
  → repeated valid hits
  → opponent death
  → corpse cache appears with dropped value
  → Bot picks up transferred loot
  → carried value increases
  → Bot extracts
  → extracted value is banked
```

The first-person HUD shows own HP, ammo/attack readiness, carried value, free
slots, extraction availability, extraction progress, and public score. A
viewer-only evaluation overlay may show opponent HP and event labels, but
those fields remain excluded from the Actor exactly as in the current release.

## Style semantics

| Style | Extraction-v2 behavior |
|---|---|
| Strong Base | maximizes expected extracted value across search, combat, and survival |
| Aggressive | contests high-value areas, initiates favorable fights, converts kills into looted value |
| Defensive | protects carried value, exits losing fights, chooses safer extraction timing |
| Explorer | searches more distinct loot regions, uses alternate approaches, discovers underused value |

This loop gives every style a decision with an understandable opportunity
cost. Aggressive is not rewarded for shooting walls; Defensive is not rewarded
for camping with no value; Explorer is not rewarded for wandering after the
inventory is full.

## Fair observation and events

The Actor may receive:

- `84×84` grayscale first-person frame;
- own health, ammo/attack readiness, carried value and remaining slots;
- extraction open/closed state and public countdown;
- own extracted score, remaining time, previous action.

The Actor may not receive opponent coordinates, opponent inventory, hidden
loot coordinates, region IDs, automap, depth, labels, or viewer telemetry.
Privileged state remains limited to Teachers, asymmetric Critic, reward,
event generation, and offline evaluation.

New auditable events:

- `LOOT_SPAWN`, `LOOT_PICKUP`, `LOOT_DROP`;
- `CACHE_CREATED`, `CACHE_LOOTED`;
- `EXTRACTION_STARTED`, `EXTRACTION_INTERRUPTED`, `EXTRACTED`;
- existing `VALID_HIT`, `DEATH`, and terminal outcome events.

## Primary metrics

- raid extraction rate;
- mean and worst-case extracted value;
- survival rate and carried value lost on death;
- kill-to-cache-loot conversion;
- cache-loot-to-extraction conversion;
- time to extraction and extraction interruption rate;
- encounter initiation, disengage success, and valid-hit rate;
- unique loot regions, route entropy, and value per travelled decision.

All headline results use held-out raids and paired cases. Kill count is a
diagnostic, not the objective.

## Implementation sequence and gates

### X0 — mechanics prototype

Implement loot, corpse cache, terminal death, extraction hold, and the event
protocol in a separate scenario directory. Gate with deterministic integration
tests for the entire consequence chain and no stale ViZDoom processes.

### X1 — Teacher capability

Add Search/Extract, Aggressive Contest, Value-Preserving Defensive, and
Alternate-Route Teachers. Freeze at least 100 validation cases per capability
before learned training.

### X2 — Strong Base transfer

Reuse the environment, synchronous duel, CNN-GRU architecture, PPO code,
evidence manifests, and audit tooling. Treat the current checkpoint only as an
initialization candidate; BC/PPO capability and all v2 thresholds must be
re-evaluated from new isolated data.

### X3 — style and difficulty

Train or govern styles against extracted-value objectives, then repeat paired
retention, anti-hacking, difficulty, and fair-observation audits. Do not reuse
current Crystal Run success rates as v2 baselines.

### X4 — consequence-first showcase

Select real validation replays that visibly complete kill → cache → loot →
extract or low-health value-preserving escape chains. Run a new blind human or
explicitly synthetic perception study under a separate label.

## Decision gate before implementation

Before X0 begins, the project owner must approve:

- terminal death with no in-raid respawn;
- no persistent stash/economy in the first prototype;
- extracted value, rather than kills, as the primary objective;
- two extraction zones and a bounded extraction hold;
- a full new experimental evidence namespace rather than retrofitting current
  M1–M6 claims.

Only after that review should long-run commands be added to `script.md`.
