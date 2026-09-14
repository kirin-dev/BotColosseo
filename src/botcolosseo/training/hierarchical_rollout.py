"""On-policy command episodes: fair action selection, privileged training signals."""

from __future__ import annotations

from collections.abc import Callable

import torch

from botcolosseo.agents.hierarchical_controller import HierarchicalController
from botcolosseo.agents.hierarchical_model import CommandActorCritic
from botcolosseo.agents.hierarchical_teacher import PrivilegedCommandTeacher
from botcolosseo.data.extraction_demonstrations import extraction_scalars
from botcolosseo.envs.actions import MacroAction
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.extraction_rollout import extraction_privileged_tensor
from botcolosseo.training.hierarchical_protocol import Command, ControlCondition
from botcolosseo.training.hierarchical_rewards import command_potential, command_reward


@torch.no_grad()
def collect_command_episode(
    model: CommandActorCritic,
    env: SynchronousExtractionEnv,
    *,
    side: str,
    layout_variant: int,
    schedule: Callable[[int], Command],
    opponent: object,
    device: str,
    gamma: float = 0.99,
    task_weight: float = 1.0,
    command_selector=None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Stop learner experience on death/extraction; still finish the global game.

    Commands come from a schedule or frozen fair high actor. Oracle scoring is training-only.
    This collector resets env/hidden, so callers must not splice partial episodes.
    """
    if side not in {"host", "opponent"}:
        raise ValueError("Invalid learner side")
    other = "opponent" if side == "host" else "host"
    observations, info = env.reset()
    if isinstance(opponent, HierarchicalController):
        if opponent.executor is not model.actor:
            raise ValueError("Population upgrade must share one executor between roles")
        opponent.reset()
    previous_other_health = getattr(observations, other).health
    model.eval()
    scorer = PrivilegedCommandTeacher(side=side, layout_variant=layout_variant)
    hidden = None
    current_command = None
    credited = False
    applicable = False
    learner_finished = False
    command_counts = {
        c.name: {"steps": 0, "starts": 0, "applicable": 0, "successes": 0} for c in Command
    }
    rows = []
    next_command = (
        command_selector.select(
            None,
            torch.as_tensor(extraction_scalars(getattr(observations, side)), device=device)[
                None, None
            ],
        )
        if command_selector is not None
        else Command(schedule(0))
    )
    for t in range(700):
        before = env.privileged_state()
        command = next_command
        if command != current_command:
            scorer.begin(command, before)
            current_command, credited = command, False
            applicable = scorer.act(before).status == "executing"
            if not learner_finished:
                command_counts[command.name]["starts"] += 1
                command_counts[command.name]["applicable"] += int(applicable)
        obs = getattr(observations, side)
        batch = {
            "frames": torch.as_tensor(obs.frame.copy(), device=device)[None, None, None],
            "scalars": torch.as_tensor(extraction_scalars(obs), device=device)[None, None],
            "previous_actions": torch.tensor([[obs.previous_action]], device=device),
            "masks": torch.tensor([[float(t > 0)]], device=device),
            "commands": torch.tensor([[int(command)]], device=device),
            "difficulty": torch.ones(1, 1, 1, device=device),
            "privileged": extraction_privileged_tensor(before, learner_side=side, device=device),
        }
        if not learner_finished:
            start_hidden = model.actor.initial_state(1, device=device) if hidden is None else hidden
            out = model(**batch, hidden=hidden)
            distribution = torch.distributions.Categorical(logits=out.logits)
            action = distribution.sample()
            hidden = out.hidden
        if isinstance(opponent, HierarchicalController):
            other_obs = getattr(observations, other)
            other_action = MacroAction.IDLE
            if other_obs.health > 0 and other_obs.banked_value <= 0:
                other_action = opponent.step(
                    torch.as_tensor(other_obs.frame.copy(), device=device)[None, None, None],
                    torch.as_tensor(extraction_scalars(other_obs), device=device)[None, None],
                    torch.tensor([[other_obs.previous_action]], device=device),
                    ControlCondition(),
                    public_event=other_obs.health < previous_other_health,
                )["action"]
            previous_other_health = other_obs.health
        else:
            other_action = opponent.act(before)
        moves = {
            side: MacroAction.IDLE if learner_finished else int(action.item()),
            other: other_action,
        }
        snapshot = env.protocol_snapshot()
        step = env.step(moves["host"], moves["opponent"])
        after = env.privileged_state()
        if not learner_finished:
            scorer.observe_transition(snapshot, env.protocol_snapshot())
            success = applicable and scorer.act(after).status == "complete" and not credited
            credited |= success
            command_counts[command.name]["steps"] += 1
            command_counts[command.name]["successes"] += int(success)
            next_obs = getattr(step, side)
            terminal = step.terminated or next_obs.health <= 0 or next_obs.banked_value > 0
            if command_selector is None:
                next_command = Command(schedule(t + 1))
            elif not terminal:
                next_command = command_selector.select(
                    hidden,
                    torch.as_tensor(extraction_scalars(next_obs), device=device)[None, None],
                    public_event=next_obs.health < obs.health,
                )
            reward = command_reward(
                previous_banked=obs.banked_value,
                banked=next_obs.banked_value,
                previous_potential=command_potential(
                    before, side=side, command=command, layout_variant=layout_variant
                ),
                next_potential=command_potential(
                    after,
                    side=side,
                    command=next_command,
                    layout_variant=layout_variant,
                ),
                gamma=gamma,
                terminated=terminal,
                newly_completed=success,
                task_weight=task_weight,
            )
            rows.append(
                {
                    **batch,
                    "actions": action,
                    "old_log_probs": distribution.log_prob(action),
                    "old_values": out.values,
                    "rewards": torch.tensor([[reward.total]], device=device),
                    "task_rewards": torch.tensor([[reward.task]], device=device),
                    "potential_rewards": torch.tensor([[reward.potential]], device=device),
                    "success_rewards": torch.tensor([[reward.command_success]], device=device),
                    "terminal": torch.tensor([[terminal]], device=device),
                    "hidden": start_hidden.transpose(0, 1),
                }
            )
            learner_finished = terminal
        observations = step
        if step.terminated or step.truncated:
            bootstrap = torch.zeros(1, 1, device=device)
            if not learner_finished:
                obs = getattr(step, side)
                final_batch = dict(batch)
                final_batch.update(
                    frames=torch.as_tensor(obs.frame.copy(), device=device)[None, None, None],
                    scalars=torch.as_tensor(extraction_scalars(obs), device=device)[None, None],
                    previous_actions=torch.tensor([[obs.previous_action]], device=device),
                    masks=torch.ones(1, 1, device=device),
                    commands=torch.tensor([[int(next_command)]], device=device),
                    privileged=extraction_privileged_tensor(
                        after, learner_side=side, device=device
                    ),
                )
                bootstrap = model(**final_batch, hidden=hidden).values
            data = {key: torch.cat([row[key] for row in rows], dim=1) for key in rows[0]}
            data["bootstrap"] = bootstrap
            discounts = gamma ** torch.arange(len(rows), device=device, dtype=torch.float64)
            return data, {
                "scenario_hash": info.scenario_hash,
                "decisions": t + 1,
                "learner_steps": len(rows),
                "banked": getattr(after, f"{side}_banked"),
                "terminated": step.terminated,
                "truncated": step.truncated,
                "command_counts": command_counts,
                "discounted_reward_components": {
                    key: float((data[key].double() * discounts).sum())
                    for key in ("task_rewards", "potential_rewards", "success_rewards")
                },
            }
    raise RuntimeError("Environment failed to terminate within configured horizon")
