from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from botcolosseo.agents.extraction_model import load_extraction_policy
from botcolosseo.agents.extraction_teachers import StyledExtractionTeacher
from botcolosseo.data.extraction_demonstrations import ExtractionCase
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_protocol import ExtractionEventType
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.extraction_bc import extraction_observation_tensors

ATTACK_ACTIONS = frozenset(
    (
        MacroAction.ATTACK,
        MacroAction.FORWARD_ATTACK,
        MacroAction.TURN_LEFT_ATTACK,
        MacroAction.TURN_RIGHT_ATTACK,
    )
)


@dataclass(frozen=True)
class ExtractionEpisodeMetrics:
    seed: int
    learner_side: str
    opponent_style: str
    decisions: int
    extracted_value: int
    extracted: bool
    died: bool
    valid_hits: int
    kills: int
    cache_looted: int
    attack_decisions: int
    unique_route_cells: int
    terminated: bool
    truncated: bool


def is_aggressive_showcase_chain(episode: ExtractionEpisodeMetrics) -> bool:
    """Return whether one replay proves the complete Aggressive product story."""
    return (
        episode.valid_hits >= 5
        and episode.kills >= 1
        and episode.cache_looted >= 1
        and episode.extracted
        and episode.extracted_value > 0
    )


def _learner_event_count(
    events: Counter[tuple[str, ExtractionEventType]],
    *,
    side: str,
    event_type: ExtractionEventType,
) -> int:
    return events[(side, event_type)]


@torch.no_grad()
def evaluate_extraction_episode(
    *,
    root: Path,
    checkpoint: Path,
    style: str,
    case: ExtractionCase,
    device: torch.device,
    max_decisions: int = 700,
    policy_model: torch.nn.Module | None = None,
) -> ExtractionEpisodeMetrics:
    scenario_hash = json.loads(
        (
            root / "assets/scenarios/crystal_run_extraction/manifest.json"
        ).read_text(encoding="utf-8")
    )["wad_sha256"]
    if policy_model is None:
        model, _ = load_extraction_policy(
            checkpoint,
            style=style,
            scenario_hash=scenario_hash,
            device=device,
        )
    else:
        model = policy_model
    env = SynchronousExtractionEnv(
        config_path=root
        / "assets/scenarios/crystal_run_extraction/crystal_run_extraction.cfg",
        seed=case.seed,
        max_decisions=max_decisions,
    )
    opponent_side = "opponent" if case.learner_side == "host" else "host"
    opponent = StyledExtractionTeacher(
        side=opponent_side,
        style=case.opponent_style,
    )
    events: Counter[tuple[str, ExtractionEventType]] = Counter()
    route_cells: set[tuple[int, int]] = set()
    attack_decisions = 0
    hidden = model.initial_state(1, device=device)
    try:
        observations, _ = env.reset()
        opponent.reset()
        episode_start = True
        terminated = False
        truncated = False
        decisions = 0
        for _ in range(max_decisions):
            decisions += 1
            observation = (
                observations.host
                if case.learner_side == "host"
                else observations.opponent
            )
            output = model(
                *extraction_observation_tensors(
                    observation,
                    episode_start=episode_start,
                    device=device,
                ),
                hidden,
            )
            learner_action = MacroAction(int(output.logits[0, 0].argmax()))
            hidden = output.hidden
            if learner_action in ATTACK_ACTIONS:
                attack_decisions += 1
            state = env.privileged_state()
            x, y = (
                (state.host_x, state.host_y)
                if case.learner_side == "host"
                else (state.opponent_x, state.opponent_y)
            )
            route_cells.add((math.floor(x / 160.0), math.floor(y / 160.0)))
            opponent_action = opponent.act(state)
            host_action, away_action = (
                (learner_action, opponent_action)
                if case.learner_side == "host"
                else (opponent_action, learner_action)
            )
            step = env.step(host_action, away_action)
            observations = type(observations)(step.host, step.opponent)
            for event in step.events:
                events[(event.side, event.type)] += event.count
            episode_start = False
            terminated, truncated = step.terminated, step.truncated
            learner = step.host if case.learner_side == "host" else step.opponent
            life = env.protocol_snapshot().public_state(case.learner_side).life_state
            if life != 1 or terminated or truncated:
                break
        learner = (
            observations.host
            if case.learner_side == "host"
            else observations.opponent
        )
        life = env.protocol_snapshot().public_state(case.learner_side).life_state
        return ExtractionEpisodeMetrics(
            seed=case.seed,
            learner_side=case.learner_side,
            opponent_style=case.opponent_style,
            decisions=decisions,
            extracted_value=learner.banked_value,
            extracted=life == 3,
            died=life == 2,
            valid_hits=_learner_event_count(
                events,
                side=case.learner_side,
                event_type=ExtractionEventType.VALID_HIT,
            ),
            kills=_learner_event_count(
                events,
                side=opponent_side,
                event_type=ExtractionEventType.DEATH,
            ),
            cache_looted=_learner_event_count(
                events,
                side=case.learner_side,
                event_type=ExtractionEventType.CACHE_LOOTED,
            ),
            attack_decisions=attack_decisions,
            unique_route_cells=len(route_cells),
            terminated=terminated,
            truncated=truncated,
        )
    finally:
        env.close()


def summarize_extraction_episodes(
    episodes: tuple[ExtractionEpisodeMetrics, ...],
) -> dict[str, object]:
    if not episodes:
        raise ValueError("Extraction evaluation requires episodes")
    count = len(episodes)
    return {
        "attack_decisions_mean": sum(item.attack_decisions for item in episodes)
        / count,
        "cache_looted_total": sum(item.cache_looted for item in episodes),
        "death_rate": sum(item.died for item in episodes) / count,
        "episodes": [asdict(item) for item in episodes],
        "extraction_rate": sum(item.extracted for item in episodes) / count,
        "kill_total": sum(item.kills for item in episodes),
        "mean_extracted_value": sum(item.extracted_value for item in episodes)
        / count,
        "route_cells_mean": sum(item.unique_route_cells for item in episodes)
        / count,
        "valid_hits_total": sum(item.valid_hits for item in episodes),
    }
