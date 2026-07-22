from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from botcolosseo.agents.checkpoint import CheckpointMetadata
from botcolosseo.agents.duel_teachers import create_duel_teacher
from botcolosseo.agents.model import AsymmetricActorCritic, RecurrentActor
from botcolosseo.agents.style_model import StyledActorCritic
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.training.duel_rollout import actor_observation_tensors

_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_LEAGUE_IDENTITY_FIELDS = {
    "base_checkpoint_sha256",
    "config_hash",
    "train_manifest_hash",
    "pool_manifest_hash",
    "payoff_report_hash",
    "scenario_hash",
}
_LEAGUE_STATE_FIELDS = {
    "environment_steps",
    "updates",
    "episodes",
    "next_pair_slot",
}


@dataclass(frozen=True)
class OpponentSpec:
    opponent_id: str
    kind: Literal["script", "checkpoint"]
    checkpoint: str | None
    checkpoint_sha256: str | None
    scenario_hash: str
    selection_evidence: str

    def __post_init__(self) -> None:
        if _ID_PATTERN.fullmatch(self.opponent_id) is None:
            raise ValueError("Invalid opponent_id")
        if self.kind not in ("script", "checkpoint"):
            raise ValueError("Invalid opponent kind")
        if not self.scenario_hash or not self.selection_evidence:
            raise ValueError("Opponent provenance must be non-empty")
        if self.kind == "script":
            if self.checkpoint is not None or self.checkpoint_sha256 is not None:
                raise ValueError("Script opponents cannot reference a checkpoint")
            return
        if not self.checkpoint:
            raise ValueError("Checkpoint opponents require a checkpoint path")
        if (
            self.checkpoint_sha256 is None
            or _SHA256_PATTERN.fullmatch(self.checkpoint_sha256) is None
        ):
            raise ValueError("Checkpoint opponents require a lowercase SHA-256")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_scenario_hash(payload: dict[str, object]) -> str:
    if "metadata" in payload:
        try:
            return CheckpointMetadata(**payload["metadata"]).scenario_hash  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("Invalid opponent checkpoint metadata") from error
    if payload.get("kind") == "style_distillation":
        required = (
            "base_checkpoint_sha256",
            "config_hash",
            "data_manifest_sha256",
        )
        hash_values = tuple(payload.get(field) for field in required)
        updates = payload.get("updates")
        if (
            payload.get("style") != "aggressive"
            or any(
                not isinstance(value, str)
                or _SHA256_PATTERN.fullmatch(value) is None
                for value in hash_values
            )
            or type(updates) is not int
            or updates <= 0
        ):
            raise ValueError("Invalid style-distillation checkpoint metadata")
        scenario_hash = payload.get("scenario_hash")
        if not isinstance(scenario_hash, str) or not scenario_hash:
            raise ValueError("Invalid style-distillation scenario hash")
        return scenario_hash
    identity = payload.get("identity")
    state = payload.get("state")
    if (
        not isinstance(identity, dict)
        or set(identity) != _LEAGUE_IDENTITY_FIELDS
        or not isinstance(state, dict)
        or set(state) != _LEAGUE_STATE_FIELDS
    ):
        raise ValueError("Invalid opponent checkpoint metadata")
    hash_fields = _LEAGUE_IDENTITY_FIELDS - {"scenario_hash"}
    if any(
        not isinstance(identity[field], str)
        or _SHA256_PATTERN.fullmatch(identity[field]) is None
        for field in hash_fields
    ):
        raise ValueError("Invalid opponent league-checkpoint identity")
    scenario_hash = identity["scenario_hash"]
    if not isinstance(scenario_hash, str) or not scenario_hash:
        raise ValueError("Invalid opponent league-checkpoint scenario hash")
    if any(type(state[field]) is not int or state[field] < 0 for field in state):
        raise ValueError("Invalid opponent league-checkpoint counters")
    if state["next_pair_slot"] != state["episodes"] // 2:
        raise ValueError("Invalid opponent league-checkpoint pair slot")
    return scenario_hash


class CheckpointOpponentPolicy:
    def __init__(
        self, spec: OpponentSpec, actor: RecurrentActor, *, device: torch.device
    ) -> None:
        self.spec = spec
        self._actor = actor.to(device).eval()
        self._actor.requires_grad_(False)
        self._device = device
        self._hidden: torch.Tensor | None = None
        self._episode_start = True

    @classmethod
    def load(
        cls, spec: OpponentSpec, *, device: torch.device
    ) -> CheckpointOpponentPolicy:
        if spec.kind != "checkpoint" or spec.checkpoint is None:
            raise ValueError("CheckpointOpponentPolicy requires a checkpoint spec")
        path = Path(spec.checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != spec.checkpoint_sha256:
            raise ValueError("Opponent checkpoint hash does not match")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported opponent checkpoint schema version")
        scenario_hash = _checkpoint_scenario_hash(payload)
        if scenario_hash != spec.scenario_hash:
            raise ValueError("Opponent checkpoint scenario hash does not match")
        state_dict = payload.get("model")
        if not isinstance(state_dict, dict):
            raise ValueError("Opponent checkpoint is missing model weights")
        try:
            if any(name.startswith("adapter.") for name in state_dict):
                bottleneck = int(state_dict["adapter.layers.0.weight"].shape[0])
                styled = StyledActorCritic.from_base(
                    AsymmetricActorCritic(), bottleneck=bottleneck
                )
                styled.load_state_dict(state_dict, strict=True)
                actor = styled.public_actor()
            else:
                model = AsymmetricActorCritic()
                model.load_state_dict(state_dict, strict=True)
                actor = model.actor
        except (KeyError, RuntimeError) as error:
            raise ValueError("Opponent checkpoint model dimensions do not match") from error
        return cls(spec, actor, device=device)

    def reset(self) -> None:
        self._hidden = self._actor.initial_state(1, device=self._device)
        self._episode_start = True

    def fork(self) -> CheckpointOpponentPolicy:
        return CheckpointOpponentPolicy(self.spec, self._actor, device=self._device)

    @torch.inference_mode()
    def act(self, observation: DuelActorObservation) -> MacroAction:
        if self._hidden is None:
            raise RuntimeError("Checkpoint opponent must be reset before act")
        inputs = actor_observation_tensors(
            observation, episode_start=self._episode_start, device=self._device
        )
        output = self._actor(*inputs, self._hidden)
        self._hidden = output.hidden
        self._episode_start = False
        return MacroAction(int(output.logits[0, 0].argmax()))


class ScriptOpponentPolicy:
    def __init__(self, name: str, graph: RegionGraph, *, side: str) -> None:
        self.spec = OpponentSpec(
            opponent_id=name,
            kind="script",
            checkpoint=None,
            checkpoint_sha256=None,
            scenario_hash="builtin",
            selection_evidence=f"builtin:{name}",
        )
        self._teacher = create_duel_teacher(name, graph, side=side)

    def reset(self, *, seed: int) -> None:
        self._teacher.reset(seed=seed)

    def act(self, state: DuelPrivilegedState) -> MacroAction:
        return MacroAction(self._teacher.act(state))
