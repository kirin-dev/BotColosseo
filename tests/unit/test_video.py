from pathlib import Path

import numpy as np

from botcolosseo.envs.video import (
    normalize_rgb_frame,
    read_video_frames,
    write_gif,
    write_mp4,
)


def test_normalize_rgb_frame_accepts_grayscale() -> None:
    frame = np.zeros((84, 84), dtype=np.uint8)

    result = normalize_rgb_frame(frame)

    assert result.shape == (84, 84, 3)
    assert result.dtype == np.uint8


def test_normalize_rgb_frame_accepts_chw() -> None:
    frame = np.zeros((3, 84, 84), dtype=np.uint8)

    result = normalize_rgb_frame(frame)

    assert result.shape == (84, 84, 3)


def test_write_mp4_uses_atomic_target(tmp_path: Path, monkeypatch) -> None:
    appended: list[np.ndarray] = []

    class FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def append_data(self, frame: np.ndarray) -> None:
            appended.append(frame)

    def fake_get_writer(path: Path, **kwargs):
        Path(path).touch()
        return FakeWriter()

    monkeypatch.setattr("botcolosseo.envs.video.imageio.get_writer", fake_get_writer)
    output = tmp_path / "smoke.mp4"

    result = write_mp4([np.zeros((8, 8), dtype=np.uint8)], output, fps=10)

    assert result == output
    assert output.is_file()
    assert len(appended) == 1


def test_write_gif_is_atomic_and_enforces_size(tmp_path: Path, monkeypatch) -> None:
    appended: list[np.ndarray] = []

    class FakeWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def append_data(self, frame: np.ndarray) -> None:
            appended.append(frame)

    def fake_get_writer(path: Path, **kwargs):
        Path(path).write_bytes(b"GIF89a")
        return FakeWriter()

    monkeypatch.setattr("botcolosseo.envs.video.imageio.get_writer", fake_get_writer)
    target = tmp_path / "comparison.gif"

    result = write_gif(
        [np.zeros((8, 8), dtype=np.uint8)],
        target,
        fps=10,
        max_bytes=10,
    )

    assert result == target
    assert target.read_bytes() == b"GIF89a"
    assert len(appended) == 1


def test_write_gif_downscales_deterministically_to_meet_ceiling(
    tmp_path: Path, monkeypatch
) -> None:
    attempted_shapes: list[tuple[int, int]] = []

    class FakeWriter:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.shape = (0, 0)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            size = 20 if self.shape == (8, 8) else 6
            self.path.write_bytes(b"x" * size)

        def append_data(self, frame: np.ndarray) -> None:
            self.shape = frame.shape[:2]
            attempted_shapes.append(self.shape)

    monkeypatch.setattr(
        "botcolosseo.envs.video.imageio.get_writer",
        lambda path, **kwargs: FakeWriter(Path(path)),
    )
    target = tmp_path / "comparison.gif"

    result = write_gif(
        [np.zeros((8, 8, 3), dtype=np.uint8)],
        target,
        fps=10,
        max_bytes=10,
    )

    assert result == target
    assert target.stat().st_size == 6
    assert attempted_shapes == [(8, 8), (7, 7)]


def test_read_video_frames_normalizes_rgba(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "botcolosseo.envs.video.imageio.get_reader",
        lambda path: iter([np.zeros((5, 7, 4), dtype=np.uint8)]),
    )

    frames = read_video_frames(tmp_path / "episode.mp4")

    assert len(frames) == 1
    assert frames[0].shape == (5, 7, 3)
