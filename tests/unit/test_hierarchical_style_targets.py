import torch

from botcolosseo.training.hierarchical_style_targets import (
    mix_style_teachers,
    style_training_tracks,
)


def test_endpoint_and_combination_targets_keep_probability_mass():
    teachers = torch.eye(4)[:, None].expand(-1, 32, -1)
    styles = style_training_tracks(32, device="cpu", compositions=True)
    targets = mix_style_teachers(teachers, styles)
    assert styles.shape == (13, 32, 3)
    torch.testing.assert_close(targets[:4], teachers)
    torch.testing.assert_close(targets.sum(-1), torch.ones(13, 32))
    torch.testing.assert_close(targets[4, 0], torch.tensor([0, 0.5, 0.5, 0]))
    torch.testing.assert_close(targets[7, 0], torch.tensor([0.5, 0.5, 0, 0]))
    assert all((track[1:] != track[:-1]).any() for track in styles[-3:])


def test_default_tracks_preserve_existing_four_endpoint_protocol():
    styles = style_training_tracks(5, device="cpu")
    assert styles.shape == (4, 5, 3)
    assert torch.equal(styles[:, 0], styles[:, -1])
