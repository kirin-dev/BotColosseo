from pathlib import Path

from botcolosseo.envs.events import EpisodeEvent, EventType
from botcolosseo.envs.rewards import RewardLedger
from botcolosseo.scenarios.regions import RegionGraph


def event(event_type: EventType, *, region_from=None, region_to=None) -> EpisodeEvent:
    return EpisodeEvent(
        episode_id=0,
        engine_tic=10,
        decision_index=1,
        type=event_type,
        region_from=region_from,
        region_to=region_to,
    )


def test_reward_caps_prevent_repeated_event_farming() -> None:
    graph = RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml"))
    ledger = RewardLedger(graph)

    pickup_reward = sum(
        ledger.apply((event(EventType.PICKUP),), target_region="center") for _ in range(3)
    )
    hit_reward = sum(
        ledger.apply((event(EventType.VALID_HIT),), target_region="center") for _ in range(8)
    )

    assert pickup_reward == 0.25
    assert hit_reward == 0.25


def test_progress_rewards_only_reduce_graph_distance() -> None:
    graph = RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml"))
    ledger = RewardLedger(graph)

    closer = ledger.apply(
        (event(EventType.REGION_TRANSITION, region_from="home", region_to="center"),),
        target_region="center",
    )
    sideways = ledger.apply(
        (event(EventType.REGION_TRANSITION, region_from="center", region_to="upper_route"),),
        target_region="away",
    )

    assert closer == 0.01
    assert sideways == 0.0


def test_score_reward_and_reset() -> None:
    graph = RegionGraph.from_yaml(Path("assets/scenarios/crystal_run/src/regions.yaml"))
    ledger = RewardLedger(graph)

    assert ledger.apply((event(EventType.SCORE),), target_region="home") == 1.0
    ledger.reset()
    assert ledger.apply((event(EventType.PICKUP),), target_region="center") == 0.25
