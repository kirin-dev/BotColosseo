from __future__ import annotations

from dataclasses import fields

import numpy as np
import torch

from botcolosseo.agents.extraction_model import create_extraction_actor
from botcolosseo.cli.render_extraction_v3 import (
    _representative_claims,
    build_parser,
)
from botcolosseo.demo.extraction_showcase import (
    _warm_extraction_policy,
    compose_extraction_showcase_frame,
)
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.extraction_protocol import ExtractionProtocolSnapshot
from botcolosseo.envs.extraction_types import (
    ExtractionActorObservation,
    ExtractionPrivilegedState,
)


def test_extraction_showcase_warms_policy_before_starting_live_game() -> None:
    model = create_extraction_actor()
    calls: list[torch.Size] = []
    model.register_forward_hook(
        lambda _module, _inputs, output: calls.append(output.logits.shape)
    )

    _warm_extraction_policy(model, device=torch.device("cpu"))

    assert calls == [torch.Size((1, 1, 13))]


def test_extraction_showcase_defaults_to_randomized_scenario() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "candidate.pt",
            "--policy",
            "strong",
            "--case-index",
            "0",
            "--output",
            "showcase.mp4",
            "--evidence",
            "showcase.json",
        ]
    )

    assert args.scenario_directory == "crystal_run_extraction_randomized"


def test_extraction_showcase_frame_has_viewer_overlay_geometry() -> None:
    observation = ExtractionActorObservation(
        frame=np.zeros((84, 84), dtype=np.uint8),
        health=80,
        ammo=20,
        carried_value=85,
        free_slots=0,
        minimum_slot_value=10,
        banked_value=0,
        extraction_open=True,
        extraction_progress=0.5,
        remaining_time=20,
        previous_action=1,
    )
    defaults = {field.name: 0 for field in fields(ExtractionProtocolSnapshot)}
    defaults.update(
        {
            "protocol_version": 3,
            "round_state": 1,
            "extraction_open": 1,
            "host_life_state": 1,
            "opponent_life_state": 1,
            "host_slot_0": 10,
            "host_slot_1": 25,
            "host_slot_2": 50,
        }
    )
    protocol = ExtractionProtocolSnapshot(**defaults)
    privileged = ExtractionPrivilegedState(
        host_x=0,
        host_y=0,
        host_angle=0,
        opponent_x=100,
        opponent_y=0,
        opponent_angle=180,
        host_health=80,
        opponent_health=40,
        host_slots=(10, 25, 50),
        opponent_slots=(10, 0, 0),
        host_banked=0,
        opponent_banked=0,
        cache_owner=0,
        cache_slots=(0, 0, 0),
        cache_x=0,
        cache_y=0,
        world_loot_mask=1,
        round_state=1,
        winner=0,
        engine_tic=1200,
    )

    frame = compose_extraction_showcase_frame(
        observation,
        privileged=privileged,
        protocol=protocol,
        learner_side="host",
        style="aggressive",
        action=MacroAction.FORWARD_ATTACK,
        event_label="HIT CONFIRMED -20 HP",
    )

    assert frame.shape == (360, 640, 3)
    assert frame.dtype == np.uint8
    assert np.count_nonzero(frame) > 10_000


def test_showcase_story_checks_are_fail_closed_and_style_specific() -> None:
    base = {
        "decisions": 300,
        "died": False,
        "extracted": True,
        "extracted_value": 85,
        "valid_hits": 0,
        "kills": 0,
        "cache_looted": 0,
        "aggressive_chains": 0,
        "successful_disengagements": 0,
        "meaningful_extractions": 1,
        "meaningful_loot_regions": 4,
        "backpack_upgrades": 1,
        "upgrade_to_extraction_conversions": 1,
    }
    explorer_ok, explorer_failures = _representative_claims("explorer", base)
    aggressive_ok, aggressive_failures = _representative_claims(
        "aggressive", base
    )

    assert explorer_ok
    assert explorer_failures == []
    assert not aggressive_ok
    assert "missing_aggressive_chains" in aggressive_failures


def test_showcase_story_rejects_short_or_combat_noisy_explorer() -> None:
    claims = {
        "decisions": 90,
        "died": False,
        "extracted": True,
        "extracted_value": 85,
        "valid_hits": 1,
        "kills": 0,
        "cache_looted": 0,
        "aggressive_chains": 0,
        "successful_disengagements": 0,
        "meaningful_extractions": 1,
        "meaningful_loot_regions": 4,
        "backpack_upgrades": 1,
        "upgrade_to_extraction_conversions": 1,
    }

    accepted, failures = _representative_claims("explorer", claims)

    assert not accepted
    assert failures == ["too_short", "explorer_combat"]
