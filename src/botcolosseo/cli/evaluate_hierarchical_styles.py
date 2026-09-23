"""Same-checkpoint style endpoint rollout against a neutral fixed opponent."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from botcolosseo.agents.hierarchical_controller import HierarchicalController
from botcolosseo.agents.hierarchical_critic import StrategicActorCritic
from botcolosseo.agents.hierarchical_model import CommandExecutor, StrategicActor
from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.data.hierarchical_demonstrations import load_command_episode
from botcolosseo.demo.control_trace_audit import audit_control_report
from botcolosseo.demo.hierarchical_controls import ScheduledController, control_schedule
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.hierarchical_collection import collect_strategic_episode
from botcolosseo.training.hierarchical_protocol import ControlCondition


def finalize_report(result):
    """Do not promote a runtime report to complete before its audit succeeds."""
    candidate = dict(result, complete=True)
    if "switch_mode" in result:
        candidate["control_audit"] = audit_control_report(candidate)
    result.update(candidate)


class NeutralOpponent(HierarchicalController):
    def step(self, frames, scalars, previous_actions, condition, *, public_event=False):
        return super().step(
            frames, scalars, previous_actions, ControlCondition(), public_event=public_event
        )


class EventObservedEnv:
    """Record post-step events for showcase selection, never feed them to actors."""

    def __init__(self, env):
        self.env = env
        self.events = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, host, opponent):
        result = self.env.step(host, opponent)
        self.events.extend(
            {"type": event.type.value, "side": event.side, "engine_tic": event.engine_tic}
            for event in result.events
        )
        return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("executor", "strategy", "opponent", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[96, 97])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--switch", action="store_true")
    parser.add_argument("--switch-mode", choices=("style", "difficulty", "joint"), default="style")
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--compositions", action="store_true")
    parser.add_argument("--only-style", choices=("neutral", "aggressive", "defensive", "explorer"))
    args = parser.parse_args()
    ControlCondition(difficulty=args.difficulty)
    if args.switch_mode != "style" and not args.switch:
        raise ValueError("A switch mode requires --switch")
    if args.switch and args.compositions:
        raise ValueError("Run static compositions and switching as separate diagnostics")
    if args.only_style and (args.switch or args.compositions):
        raise ValueError("Single-style screening cannot also request switching or compositions")
    if args.output.exists():
        raise FileExistsError("Preserve previous endpoint evaluation")
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    payload = torch.load(args.executor, map_location="cpu", weights_only=False)
    scenario = str(
        load_command_episode(Path(next(iter(payload["identity"]["validation"]))))["scenario_hash"]
    )
    low = CommandExecutor().to(args.device)
    low.load_state_dict(payload["actor"])
    actors = []
    for path in (args.strategy, args.opponent):
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item["identity"]["executor"] != digest(args.executor):
            raise ValueError("Endpoint actors must share executor")
        actor = StrategicActor().to(args.device)
        actor.load_state_dict(item["actor"])
        actors.append(actor)
    conditions = {
        "neutral": ControlCondition(difficulty=args.difficulty),
        "aggressive": ControlCondition(aggressive=1, difficulty=args.difficulty),
        "defensive": ControlCondition(defensive=1, difficulty=args.difficulty),
        "explorer": ControlCondition(explorer=1, difficulty=args.difficulty),
    }
    if args.only_style:
        conditions = {args.only_style: conditions[args.only_style]}
    if args.compositions:
        conditions.update(
            {
                "aggressive_defensive": ControlCondition(
                    aggressive=1, defensive=1, difficulty=args.difficulty
                ),
                "aggressive_explorer": ControlCondition(
                    aggressive=1, explorer=1, difficulty=args.difficulty
                ),
                "defensive_explorer": ControlCondition(
                    defensive=1, explorer=1, difficulty=args.difficulty
                ),
            }
        )
    result = {
        "executor": digest(args.executor),
        "strategy": digest(args.strategy),
        "opponent": digest(args.opponent),
        "complete": False,
        "cases": [],
        "difficulty": args.difficulty,
        "opponent_condition": "Neutral/Hard",
        "scope": "static style endpoint diagnostic against fixed Neutral/Hard opponent",
    }
    if args.switch:
        conditions = {"switch": ControlCondition(difficulty=args.difficulty)}
        result["scope"] = "same-checkpoint runtime control diagnostic; fixed Neutral/Hard opponent"
        result["switch_mode"] = args.switch_mode
        result["control_schedule"] = [
            (t, c.as_tuple())
            for t, c in control_schedule(args.switch_mode, difficulty=args.difficulty)
        ]
        result["switch_decisions"] = [0, 81, 161, 241]
    elif args.compositions:
        result["scope"] = (
            "same-checkpoint static endpoints and compositions; fixed Neutral/Hard opponent"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for style, condition in conditions.items():
        for seed in args.seeds:
            for role in ("host", "opponent"):
                controllers = {
                    side: (HierarchicalController if side == role else NeutralOpponent)(
                        low, actors[0 if side == role else 1], seed=1701 + 100 * seed + index
                    )
                    for index, side in enumerate(("host", "opponent"))
                }
                if args.switch:
                    controllers[role] = ScheduledController(
                        low,
                        actors[0],
                        seed=1701 + 100 * seed + (role == "opponent"),
                        schedule=control_schedule(args.switch_mode, difficulty=args.difficulty),
                    )
                env = SynchronousExtractionEnv(
                    config_path=Path(
                        "assets/scenarios/crystal_run_extraction_randomized/crystal_run_extraction_randomized.cfg"
                    ),
                    seed=seed,
                    layout_variant=seed % 128,
                )
                observed = EventObservedEnv(env)
                try:
                    _, report = collect_strategic_episode(
                        observed,
                        controllers,
                        StrategicActorCritic(actors[0]).to(args.device),
                        learner_side=role,
                        expected_scenario=scenario,
                        condition=condition,
                    )
                finally:
                    env.close()
                counts = Counter(row["command"] for row in report["timeline"] if row["replanned"])
                row = {
                    "style": style,
                    "initial_condition": condition.as_tuple(),
                    "seed": seed,
                    "role": role,
                    "payoff": report["payoffs"][0 if role == "host" else 1],
                    "commands": dict(counts),
                    "decisions": report["learner_decisions"],
                    "events": observed.events,
                }
                result["cases"].append(row)
                if args.switch:
                    row["control_trace"] = report["timeline"]
                temporary = args.output.with_suffix(".tmp")
                temporary.write_text(json.dumps(result, indent=2))
                temporary.replace(args.output)
                print(json.dumps(row), flush=True)
    finalize_report(result)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
