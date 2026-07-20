from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        if self.min_x >= self.max_x or self.min_y >= self.max_y:
            raise ValueError(f"Invalid bounds: {self}")

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x < self.max_x and self.min_y <= y < self.max_y

    def overlaps_interior(self, other: Bounds) -> bool:
        return (
            self.min_x < other.max_x
            and other.min_x < self.max_x
            and self.min_y < other.max_y
            and other.min_y < self.max_y
        )


@dataclass(frozen=True)
class Region:
    id: int
    name: str
    bounds: Bounds
    neighbors: tuple[str, ...]


@dataclass(frozen=True)
class Route:
    name: str
    regions: tuple[str, ...]
    waypoints: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RegionGraph:
    arena: Bounds
    regions: tuple[Region, ...]
    routes: tuple[Route, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> RegionGraph:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        arena = Bounds(**payload["arena"])
        regions = tuple(
            Region(
                id=int(item["id"]),
                name=str(item["name"]),
                bounds=Bounds(**item["bounds"]),
                neighbors=tuple(item["neighbors"]),
            )
            for item in payload["regions"]
        )
        routes = tuple(
            Route(
                name=name,
                regions=tuple(item["regions"]),
                waypoints=tuple((float(x), float(y)) for x, y in item["waypoints"]),
            )
            for name, item in payload["routes"].items()
        )
        graph = cls(arena=arena, regions=regions, routes=routes)
        graph._validate()
        return graph

    def _validate(self) -> None:
        names = [region.name for region in self.regions]
        ids = [region.id for region in self.regions]
        if len(names) != len(set(names)) or len(ids) != len(set(ids)):
            raise ValueError("Region names and IDs must be unique")
        if any(region.id <= 0 for region in self.regions):
            raise ValueError("Region IDs must be positive")
        for index, region in enumerate(self.regions):
            for other in self.regions[index + 1 :]:
                if region.bounds.overlaps_interior(other.bounds):
                    raise ValueError(f"Regions overlap: {region.name}, {other.name}")
        lookup = {region.name: region for region in self.regions}
        for region in self.regions:
            for neighbor in region.neighbors:
                if neighbor not in lookup:
                    raise ValueError(f"Unknown neighbor {neighbor!r} for {region.name!r}")
                if region.name not in lookup[neighbor].neighbors:
                    raise ValueError(f"Asymmetric neighbors: {region.name!r}, {neighbor!r}")
        route_names = [route.name for route in self.routes]
        if len(route_names) != len(set(route_names)):
            raise ValueError("Route names must be unique")
        for route in self.routes:
            if not route.regions or not route.waypoints:
                raise ValueError(f"Route must not be empty: {route.name}")
            unknown = set(route.regions).difference(lookup)
            if unknown:
                raise ValueError(f"Unknown regions in route {route.name}: {sorted(unknown)}")
            for x, y in route.waypoints:
                if not self.arena.contains(x, y):
                    raise ValueError(f"Waypoint outside arena in route {route.name}: {(x, y)}")

    def region_at(self, x: float, y: float) -> Region | None:
        matches = [region for region in self.regions if region.bounds.contains(x, y)]
        if len(matches) > 1:
            raise RuntimeError(f"Point belongs to multiple regions: {(x, y)}")
        return matches[0] if matches else None

    def region(self, name: str) -> Region:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"Unknown region: {name}")

    def route(self, name: str) -> Route:
        for route in self.routes:
            if route.name == name:
                return route
        raise KeyError(f"Unknown route: {name}")

    def shortest_path(self, start: str, goal: str) -> tuple[str, ...]:
        self.region(start)
        self.region(goal)
        queue = deque([(start, (start,))])
        visited = {start}
        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path
            neighbors = sorted(
                self.region(current).neighbors,
                key=lambda name: self.region(name).id,
            )
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, (*path, neighbor)))
        raise ValueError(f"No path from {start!r} to {goal!r}")
