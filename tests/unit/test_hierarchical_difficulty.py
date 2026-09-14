import copy

import torch

from botcolosseo.agents.hierarchical_model import CommandExecutor
from botcolosseo.training.hierarchical_difficulty import distill_episode, freeze_for_difficulty


def test_only_difficulty_changes_and_chunked_validation_preserves_memory():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    actor = CommandExecutor()
    teachers = [copy.deepcopy(actor).eval().requires_grad_(False) for _ in range(3)]
    with torch.no_grad():
        teachers[0].policy.bias[0] += 2
        teachers[1].policy.bias[1] += 1
    initial = {k: v.clone() for k, v in actor.state_dict().items()}
    batch = {
        "frames": torch.randint(256, (1, 5, 1, 84, 84), dtype=torch.uint8),
        "scalars": torch.zeros(1, 5, actor.scalar_dim),
        "previous_actions": torch.zeros(1, 5, dtype=torch.long),
        "masks": torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]]),
        "commands": torch.tensor([[0, 0, 3, 3, 5]]),
        "difficulty": torch.ones(1, 5, 1),
    }
    optimizer = torch.optim.Adam(freeze_for_difficulty(actor), lr=1e-3)
    distill_episode(actor, teachers, batch, optimizer, chunk_size=2)
    changed = [k for k, v in actor.state_dict().items() if not torch.equal(v, initial[k])]
    assert changed and all(k.startswith("difficulty_") for k in changed)
    assert all(p.grad is None for teacher in teachers for p in teacher.parameters())
    full = distill_episode(actor, teachers, batch, chunk_size=5)
    chunked = distill_episode(actor, teachers, batch, chunk_size=2)
    torch.testing.assert_close(
        torch.tensor(full["kl"]), torch.tensor(chunked["kl"]), atol=1e-6, rtol=1e-5
    )
    assert full["agreement"] == chunked["agreement"]


def test_hard_anchor_survives_large_updates_and_checkpoint_roundtrip():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    reference = CommandExecutor().eval()
    # Nonzero pretrained FiLM, as in the real executor.
    with torch.no_grad():
        for module in (reference.difficulty_input, reference.difficulty_output):
            module.network[-1].weight.normal_(0, 0.03)
            module.network[-1].bias.normal_(0, 0.03)
    actor = copy.deepcopy(reference)
    actor.difficulty_input.anchor_at_hard()
    actor.difficulty_output.anchor_at_hard()
    trainable = freeze_for_difficulty(actor)
    assert all(not p.requires_grad for n, p in actor.named_parameters() if "anchor_network" in n)
    with torch.no_grad():
        for parameter in trainable:
            parameter.add_(torch.randn_like(parameter) * 0.3)
    restored = CommandExecutor().eval()
    restored.load_state_dict(actor.state_dict())
    inputs = {
        "frames": torch.randint(256, (1, 6, 1, 84, 84), dtype=torch.uint8),
        "scalars": torch.randn(1, 6, actor.scalar_dim),
        "previous_actions": torch.zeros(1, 6, dtype=torch.long),
        "masks": torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        "commands": torch.tensor([[0, 0, 3, 3, 5, 6]]),
        "difficulty": torch.ones(1, 6, 1),
    }
    with torch.no_grad():
        old = reference(**inputs)
        new = restored(**inputs)
        assert torch.equal(new.logits, old.logits)
        assert torch.equal(new.hidden, old.hidden)
        easy = restored(**{**inputs, "difficulty": torch.zeros(1, 6, 1)})
        assert not torch.equal(easy.logits, old.logits)
        # Ordinary loading can return to a legacy checkpoint without extra keys.
        restored.load_state_dict(reference.state_dict())
        assert restored.difficulty_input.anchor_network is None
        assert torch.equal(restored(**inputs).logits, old.logits)
