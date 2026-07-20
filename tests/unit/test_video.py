from pathlib import Path

import numpy as np

from botcolosseo.envs.video import normalize_rgb_frame, write_mp4


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
