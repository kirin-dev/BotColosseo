from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from botcolosseo.agents.difficulty import DifficultyPolicy, DifficultyProfile
from botcolosseo.agents.hybrid_policy import HybridStylePolicy
from botcolosseo.agents.style_governor import GovernorTelemetry
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.duel_types import DuelActorObservation, DuelPrivilegedState


@dataclass(frozen=True)
class HybridExecutionTrace:
    decision_index: int
    policy_updated: bool
    proposed_action: MacroAction
    emitted_action: MacroAction
    source_decision_index: int | None
    base_action: MacroAction | None
    state: str
    trigger: str
    reason: str
    intervened: bool
    used_override: bool
    fallback_condition: str
    route_mode: str | None
    warmup: bool


@dataclass(frozen=True)
class _Provenance:
    action: MacroAction
    telemetry: GovernorTelemetry | None


class TracedHybridDifficultyPolicy:
    def __init__(
        self,
        name: str,
        policy: HybridStylePolicy,
        profile: DifficultyProfile,
    ) -> None:
        if name not in ("defensive", "explorer"):
            raise ValueError("Traced hybrid difficulty style is invalid")
        self.name = name
        self.profile = profile
        self._hybrid = policy
        self._difficulty = DifficultyPolicy(policy, profile)
        self._decision = 0
        self._held: _Provenance | None = None
        self._delay: deque[_Provenance] = deque()
        self._trace: list[HybridExecutionTrace] = []
        self._ready = False

    @property
    def trace(self) -> tuple[HybridExecutionTrace, ...]:
        return tuple(self._trace)

    @property
    def governor_telemetry(self) -> tuple[GovernorTelemetry, ...]:
        return self._hybrid.telemetry

    def reset(self, *, seed: int) -> None:
        del seed
        self._difficulty.reset()
        warmup = _Provenance(MacroAction.IDLE, None)
        self._delay = deque(
            [warmup] * self.profile.reaction_delay,
            maxlen=self.profile.reaction_delay + 1,
        )
        self._decision = 0
        self._held = None
        self._trace.clear()
        self._ready = True

    def act(
        self,
        observation: DuelActorObservation,
        state: DuelPrivilegedState,
    ) -> MacroAction:
        del state
        if not self._ready:
            raise RuntimeError("Traced hybrid difficulty policy must be reset")
        updated = self._decision % self.profile.policy_update_interval == 0
        before = len(self._hybrid.telemetry)
        emitted = self._difficulty.act(observation)
        after = len(self._hybrid.telemetry)
        if updated:
            if after != before + 1:
                raise RuntimeError("Hybrid difficulty governor update accounting drifted")
            telemetry = self._hybrid.telemetry[-1]
            self._held = _Provenance(telemetry.final_action, telemetry)
        elif after != before or self._held is None:
            raise RuntimeError("Hybrid difficulty held-action accounting drifted")
        self._delay.append(self._held)
        source = self._delay.popleft()
        if source.action != emitted:
            raise RuntimeError("Hybrid difficulty action provenance drifted")
        telemetry = source.telemetry
        self._trace.append(
            HybridExecutionTrace(
                decision_index=self._decision,
                policy_updated=updated,
                proposed_action=self._held.action,
                emitted_action=emitted,
                source_decision_index=(
                    telemetry.decision_index if telemetry is not None else None
                ),
                base_action=telemetry.base_action if telemetry is not None else None,
                state=telemetry.state if telemetry is not None else "warmup",
                trigger=telemetry.trigger if telemetry is not None else "warmup",
                reason=telemetry.reason if telemetry is not None else "fifo_warmup",
                intervened=telemetry.intervened if telemetry is not None else False,
                used_override=telemetry.used_override if telemetry is not None else False,
                fallback_condition=(
                    telemetry.fallback_condition if telemetry is not None else "warmup"
                ),
                route_mode=telemetry.route_mode if telemetry is not None else None,
                warmup=telemetry is None,
            )
        )
        self._decision += 1
        return emitted

    def drain_evidence(
        self,
    ) -> tuple[tuple[GovernorTelemetry, ...], tuple[HybridExecutionTrace, ...]]:
        telemetry = self._hybrid.drain_telemetry()
        trace = self.trace
        self._trace.clear()
        return telemetry, trace
