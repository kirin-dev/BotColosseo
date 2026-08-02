from __future__ import annotations

from collections import Counter
from pathlib import Path

from botcolosseo.agents.extraction_teachers import (
    ExtractionStyle,
    PrivilegedStrongExtractionTeacher,
    StyledExtractionTeacher,
)
from botcolosseo.envs.extraction_layouts import randomized_layout_variant
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv


def main() -> None:
    config = Path(
        "assets/scenarios/crystal_run_extraction_randomized/"
        "crystal_run_extraction_randomized.cfg"
    )
    for seed in (51001, 51037):
        variant = randomized_layout_variant(seed)
        env = SynchronousExtractionEnv(
            config_path=config,
            seed=seed,
            max_decisions=700,
            layout_variant=variant,
        )
        learner = PrivilegedStrongExtractionTeacher(
            side="host", layout_variant=variant
        )
        opponent = StyledExtractionTeacher(
            side="opponent", style=ExtractionStyle.DEFENSIVE
        )
        events: Counter[str] = Counter()
        try:
            _, info = env.reset()
            learner.reset()
            opponent.reset()
            assert env.protocol_snapshot().world_loot_mask == 127
            decisions = 0
            for _ in range(700):
                decisions += 1
                state = env.privileged_state()
                step = env.step(learner.act(state), opponent.act(state))
                events.update(f"{event.side}:{event.type.value}" for event in step.events)
                if step.terminated or step.truncated:
                    break
            state = env.privileged_state()
            print(
                {
                    "seed": seed,
                    "variant": variant,
                    "scenario_hash": info.scenario_hash,
                    "decisions": decisions,
                    "host_banked": state.host_banked,
                    "winner": state.winner,
                    "events": dict(events),
                }
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
