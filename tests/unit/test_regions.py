from pathlib import Path

import pytest

from botcolosseo.scenarios.regions import RegionGraph

REGIONS_PATH = Path("assets/scenarios/crystal_run/src/regions.yaml")


def test_tracked_region_graph_is_valid_and_connected() -> None:
    graph = RegionGraph.from_yaml(REGIONS_PATH)

    assert len({region.id for region in graph.regions}) == len(graph.regions)
    assert all(region.id > 0 for region in graph.regions)
    assert graph.shortest_path("home", "center") == ("home", "center")
    assert graph.route("flank").regions != graph.shortest_path("home", "center")


def test_region_boundaries_have_deterministic_ownership() -> None:
    graph = RegionGraph.from_yaml(REGIONS_PATH)

    assert graph.region_at(-512.0, 0.0).name == "lower_route"
    assert graph.region_at(-768.0, 0.0).name == "home"
    assert graph.region_at(768.0, 0.0) is None


def test_unknown_region_path_is_rejected() -> None:
    graph = RegionGraph.from_yaml(REGIONS_PATH)

    with pytest.raises(KeyError, match="unknown"):
        graph.shortest_path("home", "unknown")
