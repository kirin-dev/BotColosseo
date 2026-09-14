"""Single live collector check, not a training experiment."""

import copy
from pathlib import Path

import torch

from botcolosseo.agents.extraction_teachers import PrivilegedStrongExtractionTeacher
from botcolosseo.agents.hierarchical_model import CommandActorCritic, CommandExecutor
from botcolosseo.cli.evaluate_hierarchical_executor import scheduled_command
from botcolosseo.data.hierarchical_demonstrations import episode_tensors, load_command_episode
from botcolosseo.envs.synchronous_extraction import SynchronousExtractionEnv
from botcolosseo.training.hierarchical_ppo import update_command_ppo
from botcolosseo.training.hierarchical_rollout import collect_command_episode


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = CommandActorCritic(CommandExecutor()).to("cuda:0")
    saved = torch.load(
        "runs/hierarchical/bc-corrections1-retry/best.pt", map_location="cpu", weights_only=False
    )
    model.actor.load_state_dict(saved["actor"])
    env = SynchronousExtractionEnv(
        config_path=Path(
            "assets/scenarios/crystal_run_extraction_randomized/crystal_run_extraction_randomized.cfg"
        ),
        seed=0,
        layout_variant=0,
    )
    try:
        data, summary = collect_command_episode(
            model,
            env,
            side="host",
            layout_variant=0,
            schedule=lambda t: scheduled_command(t, 0),
            opponent=PrivilegedStrongExtractionTeacher(side="opponent", layout_variant=0),
            device="cuda:0",
        )
        for key, value in data.items():
            assert torch.isfinite(value).all(), key
        # Recompute a short on-policy recurrent prefix before any optimizer mutation.
        n = min(16, summary["learner_steps"])
        inputs = {
            key: data[key][:, :n]
            for key in (
                "frames",
                "scalars",
                "previous_actions",
                "masks",
                "commands",
                "difficulty",
                "privileged",
            )
        }
        with torch.no_grad():
            out = model(**inputs)
            logp = torch.distributions.Categorical(logits=out.logits).log_prob(
                data["actions"][:, :n]
            )
        torch.testing.assert_close(logp, data["old_log_probs"][:, :n], atol=1e-5, rtol=1e-5)
        print(summary, flush=True)
        print("finite rollout and on-policy log-probability replay: PASS", flush=True)
        reference = copy.deepcopy(model.actor).requires_grad_(False).eval()
        replay = episode_tensors(
            load_command_episode(Path("data/hierarchical/train32/seed-0-host.npz")), device="cuda:0"
        )
        metrics = update_command_ppo(
            model, reference, torch.optim.Adam(model.parameters(), lr=1e-5), data, replay
        )
        assert all(torch.isfinite(p).all() for p in model.parameters())
        print({"ppo_update": metrics}, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
