"""Whole-game collection for a frozen executor and a trainable strategic actor."""

from dataclasses import asdict

import torch

from botcolosseo.data.extraction_demonstrations import extraction_scalars
from botcolosseo.training.extraction_rollout import extraction_privileged_tensor
from botcolosseo.training.hierarchical_protocol import ControlCondition, own_payoffs
from botcolosseo.training.hierarchical_smdp import aggregate_strategic_transitions
from botcolosseo.training.hierarchical_strategic_ppo import HIGH_INPUT_KEYS


@torch.no_grad()
def collect_strategic_episode(
    env, controllers, model, *, learner_side, expected_scenario, condition=None
):
    """Collect one role against another frozen hierarchical controller.

    Finish global settlement after local extraction/death. A timeout uses the
    final fair observation and current recurrent state for value bootstrapping.
    Controllers may resolve a declared runtime schedule. Replans retain actual
    sampled conditions; timeout bootstrapping resolves the next condition too.
    """
    condition = ControlCondition() if condition is None else condition
    if learner_side not in {"host", "opponent"} or set(controllers) != {"host", "opponent"}:
        raise ValueError("Need both role controllers and a valid learner role")
    if controllers[learner_side].strategy is not model.actor:
        raise ValueError("Learner controller must use the trained strategic actor")
    if controllers["host"].executor is not controllers["opponent"].executor:
        raise ValueError("Both roles must share one frozen executor")
    device = controllers[learner_side].device
    for controller in controllers.values():
        controller.reset()
    model.eval()
    observations, info = env.reset()
    if info.scenario_hash != expected_scenario:
        raise ValueError("Strategic collection scenario mismatch")
    previous_health = {side: getattr(observations, side).health for side in controllers}
    samples, timeline, rewards = [], [], []
    local_terminal = False
    for decision in range(700):
        actions = {}
        learner_active = False
        for side, controller in controllers.items():
            obs = getattr(observations, side)
            if obs.health <= 0 or obs.banked_value > 0:
                actions[side] = 0
                continue
            record = controller.step(
                torch.as_tensor(obs.frame.copy(), device=device)[None, None, None],
                torch.as_tensor(extraction_scalars(obs), device=device)[None, None],
                torch.tensor([[obs.previous_action]], device=device),
                condition,
                public_event=obs.health < previous_health[side],
            )
            previous_health[side] = obs.health
            actions[side] = record["action"]
            if side == learner_side:
                learner_active = True
                timeline.append(record)
                if record["replanned"]:
                    sample = dict(controller.last_high_inputs)
                    sample["privileged"] = extraction_privileged_tensor(
                        env.privileged_state(), learner_side=side, device=device
                    )
                    sample["old_values"] = model(**sample).values
                    samples.append(sample)
        previous_banked = getattr(observations, learner_side).banked_value
        observations = env.step(actions["host"], actions["opponent"])
        own = getattr(observations, learner_side)
        if learner_active:
            rewards.append((own.banked_value - previous_banked) / 150)
            local_terminal = own.health <= 0 or own.banked_value > 0 or observations.terminated
        if not (observations.terminated or observations.truncated):
            continue
        final = env.privileged_state()
        if abs(sum(rewards) - getattr(final, f"{learner_side}_banked") / 150) > 1e-8:
            raise RuntimeError("Local rewards disagree with final settlement")
        segments = [
            asdict(segment)
            for segment in aggregate_strategic_transitions(
                timeline,
                rewards,
                gamma=0.997,
                terminated=local_terminal,
                truncated=not local_terminal,
            )
        ]
        if len(samples) != len(segments) or not samples:
            raise RuntimeError("Missing strategic samples")
        batch = {
            key: torch.cat([sample[key] for sample in samples], 1)
            for key in (*HIGH_INPUT_KEYS, "privileged", "old_values")
        }
        for key, field, dtype in (
            ("commands", "command", torch.long),
            ("old_log_probs", "log_prob", torch.float32),
            ("rewards", "reward", torch.float32),
            ("durations", "duration", torch.long),
            ("terminated", "terminated", torch.bool),
            ("truncated", "truncated", torch.bool),
        ):
            batch[key] = torch.tensor(
                [[segment[field] for segment in segments]], dtype=dtype, device=device
            )
        bootstrap = torch.zeros(1, 1, device=device)
        if not local_terminal:
            controller = controllers[learner_side]
            bootstrap_condition = controller.resolve_condition(condition)
            bootstrap = model(
                features=controller.features,
                scalars=torch.as_tensor(extraction_scalars(own), device=device)[None, None],
                previous_commands=torch.tensor([[int(controller.command)]], device=device),
                elapsed=torch.tensor([[[float(controller.elapsed)]]], device=device),
                style=torch.tensor(
                    [[bootstrap_condition.as_tuple()[:3]]], device=device, dtype=torch.float32
                ),
                difficulty=torch.tensor(
                    [[[bootstrap_condition.difficulty]]], device=device, dtype=torch.float32
                ),
                masks=torch.ones(1, 1, device=device),
                hidden=controller.high_hidden,
                privileged=extraction_privileged_tensor(
                    final, learner_side=learner_side, device=device
                ),
            ).values
        batch["next_values"] = torch.cat((batch["old_values"][:, 1:], bootstrap), 1)
        return batch, {
            "global_decisions": decision + 1,
            "learner_decisions": len(rewards),
            "strategic_decisions": len(segments),
            "local_terminal": local_terminal,
            "payoffs": own_payoffs(final.host_banked, final.opponent_banked, globally_settled=True),
            "timeline": timeline,
        }
    raise RuntimeError("No global settlement within safety bound")
