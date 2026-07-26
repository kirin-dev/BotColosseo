from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

PLAYER_MAX_HEALTH = 100
STARTING_AMMO = 30
MAX_AMMO = 40
FIXED_HIT_DAMAGE = 20
BACKPACK_SLOTS = 3
LOOT_VALUES = frozenset((10, 25, 50))
EXTRACTION_OPEN_TIC = 30 * 35
EXTRACTION_REQUIRED_TICS = 3 * 35
RAID_TIMEOUT_TIC = 75 * 35


class LifeState(IntEnum):
    ACTIVE = 1
    DEAD = 2
    EXTRACTED = 3


@dataclass(frozen=True)
class Backpack:
    items: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.items) > BACKPACK_SLOTS:
            raise ValueError("Backpack exceeds three slots")
        if any(value not in LOOT_VALUES for value in self.items):
            raise ValueError(f"Unsupported loot value: {self.items}")

    @property
    def total(self) -> int:
        return sum(self.items)

    @property
    def free_slots(self) -> int:
        return BACKPACK_SLOTS - len(self.items)

    @property
    def minimum_value(self) -> int:
        return min(self.items, default=0)

    def pickup(self, value: int) -> BackpackPickup:
        if value not in LOOT_VALUES:
            raise ValueError(f"Unsupported loot value: {value}")
        if self.free_slots:
            return BackpackPickup(
                backpack=Backpack((*self.items, value)),
                accepted=True,
                dropped_value=None,
            )
        minimum = self.minimum_value
        if value <= minimum:
            return BackpackPickup(
                backpack=self,
                accepted=False,
                dropped_value=None,
            )
        replace_index = self.items.index(minimum)
        updated = list(self.items)
        updated[replace_index] = value
        return BackpackPickup(
            backpack=Backpack(tuple(updated)),
            accepted=True,
            dropped_value=minimum,
        )


@dataclass(frozen=True)
class BackpackPickup:
    backpack: Backpack
    accepted: bool
    dropped_value: int | None


@dataclass(frozen=True)
class PlayerRaidState:
    health: int = PLAYER_MAX_HEALTH
    ammo: int = STARTING_AMMO
    backpack: Backpack = Backpack()
    banked_value: int = 0
    life_state: LifeState = LifeState.ACTIVE

    def __post_init__(self) -> None:
        if not 0 <= self.health <= PLAYER_MAX_HEALTH:
            raise ValueError(f"Invalid extraction health: {self.health}")
        if not 0 <= self.ammo <= MAX_AMMO:
            raise ValueError(f"Invalid extraction ammo: {self.ammo}")
        if self.banked_value < 0:
            raise ValueError("Banked value must be nonnegative")
        if self.life_state is LifeState.ACTIVE and self.health == 0:
            raise ValueError("Active player must have positive health")
        if self.life_state is not LifeState.ACTIVE and self.health != 0:
            raise ValueError("Inactive player must have zero health")

    def take_hit(self, damage: int = FIXED_HIT_DAMAGE) -> DamageResult:
        if self.life_state is not LifeState.ACTIVE:
            raise ValueError("Inactive player cannot take damage")
        if damage <= 0:
            raise ValueError("Damage must be positive")
        health = max(0, self.health - damage)
        if health:
            return DamageResult(
                player=PlayerRaidState(
                    health=health,
                    ammo=self.ammo,
                    backpack=self.backpack,
                    banked_value=self.banked_value,
                ),
                cache=None,
            )
        return DamageResult(
            player=PlayerRaidState(
                health=0,
                ammo=self.ammo,
                backpack=Backpack(),
                banked_value=self.banked_value,
                life_state=LifeState.DEAD,
            ),
            cache=CorpseCache(self.backpack.items),
        )

    def extract(self) -> PlayerRaidState:
        if self.life_state is not LifeState.ACTIVE:
            raise ValueError("Only an active player can extract")
        return PlayerRaidState(
            health=0,
            ammo=self.ammo,
            backpack=Backpack(),
            banked_value=self.banked_value + self.backpack.total,
            life_state=LifeState.EXTRACTED,
        )


@dataclass(frozen=True)
class CorpseCache:
    items: tuple[int, ...]

    def __post_init__(self) -> None:
        Backpack(self.items)

    @property
    def total(self) -> int:
        return sum(self.items)

    def loot(self, backpack: Backpack) -> CacheLootResult:
        current = backpack
        remaining: list[int] = []
        world_drops: list[int] = []
        for value in sorted(self.items, reverse=True):
            result = current.pickup(value)
            if not result.accepted:
                remaining.append(value)
                continue
            current = result.backpack
            if result.dropped_value is not None:
                world_drops.append(result.dropped_value)
        return CacheLootResult(
            backpack=current,
            cache=CorpseCache(tuple(remaining)),
            world_drops=tuple(world_drops),
        )


@dataclass(frozen=True)
class DamageResult:
    player: PlayerRaidState
    cache: CorpseCache | None


@dataclass(frozen=True)
class CacheLootResult:
    backpack: Backpack
    cache: CorpseCache
    world_drops: tuple[int, ...]


@dataclass(frozen=True)
class ExtractionProgress:
    tics: int
    completed: bool
    interrupted: bool


def advance_extraction(
    previous_tics: int,
    *,
    elapsed_tics: int,
    extraction_open: bool,
    inside_zone: bool,
    took_damage: bool = False,
    attacked: bool = False,
) -> ExtractionProgress:
    if previous_tics < 0 or elapsed_tics <= 0:
        raise ValueError("Extraction timing values are invalid")
    interrupted = previous_tics > 0 and (
        not extraction_open or not inside_zone or took_damage or attacked
    )
    if not extraction_open or not inside_zone or took_damage or attacked:
        return ExtractionProgress(tics=0, completed=False, interrupted=interrupted)
    current = min(previous_tics + elapsed_tics, EXTRACTION_REQUIRED_TICS)
    return ExtractionProgress(
        tics=current,
        completed=current == EXTRACTION_REQUIRED_TICS,
        interrupted=False,
    )


def winner(host_banked: int, opponent_banked: int) -> int:
    if host_banked < 0 or opponent_banked < 0:
        raise ValueError("Banked values must be nonnegative")
    if host_banked > opponent_banked:
        return 1
    if opponent_banked > host_banked:
        return 2
    return 3
