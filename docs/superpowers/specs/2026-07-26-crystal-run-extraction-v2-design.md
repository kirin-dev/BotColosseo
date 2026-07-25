# Crystal Run Extraction v2 Design

## Status and product objective

This specification freezes the first implementation scope for a separate
Crystal Run Extraction v2 scenario. It does not modify or reinterpret the
released Crystal Run Arena evidence.

The product objective is not to reproduce a full commercial extraction
shooter. A viewer unfamiliar with the project must be able to understand this
causal loop from a 30--45 second first-person clip:

```text
search for loot
  -> choose to fight or disengage
  -> death transfers carried loot
  -> keep or replace scarce backpack contents
  -> extract successfully
  -> bank value and determine the result
```

Aggressive, Defensive, and Explorer Bots must make visibly different choices
inside the same rules while retaining useful extraction performance.

## Scope boundary

The first version includes:

- synchronous fair-start 1v1 raids;
- two neutral extraction zones;
- three loot values and a three-slot backpack;
- automatic pickup and deterministic replacement of the lowest-value item;
- finite total ammunition without magazines or manual reload;
- fixed-damage combat, terminal death, and a lootable corpse cache;
- extracted value as the only score that determines the result;
- public-observation Bot inputs and a separate viewer-only overlay.

The first version excludes armor, healing, multiple weapons, manual inventory
selection, manual item dropping, weapon rarity, shops, insurance, persistent
stashes, squads, and cross-raid progression.

## Raid geometry and timing

The existing compact arena geometry is reused. The first version changes
landmarks and scripted regions rather than building a new map from scratch.

- The host and opponent spawn at the current west and east positions.
- The north extraction zone is centered near `(0, 400)`.
- The south extraction zone is centered near `(0, -400)`.
- Both extraction zones are neutral and equally available to both players.
- Spawn-to-extraction distances are symmetric under side swapping.
- Extraction zones receive distinct floor coloring, lighting, and visible
  markers so they can be identified in first-person video.
- A raid lasts 75 seconds.
- Both extraction zones open at 30 seconds.
- Extraction requires remaining inside one zone for 3 uninterrupted seconds.
- Leaving the zone, taking damage, or issuing an attack action resets progress
  to zero.

When a player extracts, that player becomes inactive and keeps the banked
value. The other player may continue until they extract, die, or the raid
times out.

## Combat

Both players start with:

- 100 HP;
- the same basic hitscan rifle;
- 30 rounds of total ammunition;
- an empty backpack.

The release version of the rifle deals exactly 20 HP per valid hit, so a
full-health player dies after five valid hits. It has no headshots, armor,
damage falloff, splash damage, or health regeneration. The weapon reuses
existing pistol-compatible presentation assets but uses a small custom weapon
definition to make damage deterministic.

There is no magazine state and no reload action. Ammunition decreases directly
from the total count. Two side-symmetric ammunition pickups provide 10 rounds
each, with a hard cap of 40 rounds. Ammunition is a consumable and does not
occupy a backpack slot.

No new Bot action is introduced for combat or inventory management. The
existing movement, turning, attack, and combined movement-attack actions are
retained.

## Loot and backpack

Each raid uses a side-symmetric loot template:

- four low-risk items worth 10, with two accessible from each spawn side;
- two contested-route items worth 25;
- one exposed central item worth 50.

The exact candidate positions vary by seeded split, but each paired side-swap
case uses the same loot template and seed.

Each item consumes one of three backpack slots. Walking over an item applies
these deterministic rules:

1. If a slot is free, pick up the item.
2. If the backpack is full and the new value is greater than the current
   minimum value, replace the oldest item among the minimum-value items.
3. Spawn the replaced item at the player's position so it remains available
   to the opponent.
4. If the new value is less than or equal to the current minimum value, leave
   it in the world.

This preserves scarcity and replacement decisions without adding inventory
menus or `DROP_SLOT` actions. The Bot retains strategic control because it
chooses which visible items to approach and whether further search is worth
the risk.

## Death, corpse cache, extraction, and outcome

Death is terminal for the raid. There is no in-raid respawn.

On death:

1. The dead player's current backpack contents are removed.
2. One visible corpse cache is created at the death position.
3. The cache records the three individual slot values, not only their sum.
4. A surviving player who touches the cache processes its items from highest
   to lowest value through the normal backpack replacement rules.
5. Values that cannot improve the survivor's backpack remain in the cache.
6. The cache disappears only when it is empty or the raid times out.

A kill gives no direct score. Carried value also gives no score. Only value
successfully banked through extraction counts.

- Higher extracted value wins.
- Equal extracted value is a draw.
- Kills do not break ties.
- A player who kills the opponent but fails to extract can still finish on
  zero.
- Unextracted carried value is lost at timeout.

For same-tic conflicts, the engine applies this fixed order:

```text
damage and death
  -> corpse-cache creation
  -> world and cache pickup
  -> extraction progress or completion
  -> timeout
```

Therefore a lethal hit prevents an extraction that would otherwise complete
on the same tic.

## Observation boundary

The Actor may receive:

- `84x84` grayscale first-person pixels;
- own HP and total ammunition;
- own carried total value, free-slot count, and minimum carried slot value;
- whether extraction is open;
- own extraction progress;
- own banked value, remaining time, and previous action.

The Actor may not receive:

- opponent HP, coordinates, angle, ammunition, backpack, or extraction progress;
- hidden loot coordinates or loot-template identity;
- region IDs, automap, depth, labels, or viewer telemetry.

Opponent HP, opponent carried value, hit damage, and event labels may appear in
the recorded viewer overlay. They remain excluded from Actor observations.
Teachers, the asymmetric Critic, reward computation, deterministic event
generation, and offline evaluation may use privileged state.

## Auditable state and events

The scenario protocol exposes enough state to validate the causal chain:

- player life state, own carried slots, and banked value;
- loot identity, value, world state, and position;
- cache identity, remaining slot values, and position;
- extraction open state, active zone, and progress;
- fixed damage and total-ammunition state.

Required events are:

- `VALID_HIT` with damage;
- `DEATH`;
- `LOOT_SPAWN`, `LOOT_PICKUP`, and `LOOT_DROP`;
- `CACHE_CREATED` and `CACHE_LOOTED`;
- `EXTRACTION_STARTED`, `EXTRACTION_INTERRUPTED`, and `EXTRACTED`;
- `TIMEOUT`.

Every transition is monotonic and emitted once. Event ledgers reject duplicate
pickup, value creation, value loss without a corresponding drop or timeout,
extraction after death, and banked value without an `EXTRACTED` event.

## Style semantics

### Strong Base

Maximizes expected extracted value by balancing search value, combat
probability, health, ammunition, carried value, extraction distance, and
remaining time.

### Aggressive

- enters contested and high-value areas earlier;
- initiates more favorable encounters;
- pursues damaged opponents when ammunition and health permit;
- converts kills into cache loot and then extraction;
- is not rewarded for shots without valid hits or kills without extraction.

### Defensive

- becomes more risk-averse as carried value rises;
- disengages at low health or ammunition;
- uses safer extraction timing and alternate extraction zones;
- is not rewarded for camping while carrying no meaningful value.

### Explorer

- visits more distinct loot regions and alternate approaches;
- searches for upgrades when the backpack contains low-value items;
- stops wandering when full of high-value items or when extraction urgency is
  high;
- is not rewarded for distance without loot discovery or extraction value.

## Reward and evaluation principle

The shared capability objective is expected extracted value, not kill count.
Style shaping may influence encounter choice, survival behavior, and route
coverage, but it may not award terminal success independently of extraction.

Headline held-out metrics are:

- extraction rate and mean extracted value;
- survival rate and value lost on death;
- kill-to-cache-loot and cache-loot-to-extraction conversion;
- replacement of low-value items by higher-value items;
- encounter initiation, valid-hit rate, and disengage success;
- unique loot regions, route entropy, and value per travelled decision.

Style success is evaluated against a frozen Strong Base on paired validation
cases. Test cases remain inaccessible until selection is complete.

## Showcase acceptance

The public showcase must contain real validation replays, not scripted
animations, with a viewer overlay that displays:

- both players' HP bars and each valid `-20` hit;
- own ammunition;
- backpack slots and replacement animation;
- carried and banked value;
- corpse-cache creation and transferred value;
- extraction availability and progress;
- a final extracted-value result.

At least one selected Aggressive clip must show:

```text
five-hit kill -> corpse cache -> cache pickup -> successful extraction
```

At least one Defensive clip must show protection of meaningful carried value
through disengagement or a safer extraction choice. At least one Explorer clip
must show an alternate route and a genuine backpack-value upgrade. Each style
clip should be 30--45 seconds unless a longer clip is required to preserve the
complete causal outcome.

## Technical impact and isolation

The work remains a bounded extension rather than a rewrite:

- reuse the UDMF arena geometry and add only extraction markers and loot
  landmarks;
- add a separate ACS protocol and WAD namespace for v2;
- extend the WAD builder with the minimal custom weapon/item definitions;
- reuse the synchronous duel process, action set, CNN-GRU, PPO, evidence
  manifests, paired evaluation, and showcase renderer;
- define new observation and checkpoint schema versions because the public
  scalar inputs change;
- store all v2 splits, checkpoints, reports, and media in a separate namespace.

Current Crystal Run checkpoints may be evaluated as initialization candidates,
but no current M1--M6 metric or pass claim transfers to v2.

## Mechanics acceptance gate

Before any long training run, deterministic CPU and real-ViZDoom integration
checks must demonstrate:

1. two symmetric extraction zones with correct interruption behavior;
2. 100 HP, deterministic 20-damage hits, 30 starting rounds, and no reload;
3. three-slot pickup, rejection, deterministic replacement, and world drop;
4. terminal death and exact corpse-cache value conservation;
5. kill, cache loot, extraction, and banked-value outcome in one replay;
6. no respawn, duplicate event, hidden Actor input, or stale ViZDoom process;
7. side-swapped seeded cases preserve loot and extraction fairness.

Only after this gate passes may teacher-data generation or GPU training begin.
