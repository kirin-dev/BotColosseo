from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray


def normalize_rgb_frame(frame: NDArray[np.generic]) -> NDArray[np.uint8]:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported frame shape: {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    return np.ascontiguousarray(array, dtype=np.uint8)


def write_mp4(
    frames: Iterable[NDArray[np.generic]],
    output_path: Path,
    fps: int,
) -> Path:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    try:
        with imageio.get_writer(
            temporary_path,
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            macro_block_size=None,
        ) as writer:
            frame_count = 0
            for frame in frames:
                writer.append_data(normalize_rgb_frame(frame))
                frame_count += 1
        if frame_count == 0:
            raise ValueError("Cannot write an empty video")
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def write_gif(
    frames: Iterable[NDArray[np.generic]],
    output_path: Path,
    *,
    fps: int,
    max_bytes: int,
) -> Path:
    if fps <= 0 or max_bytes <= 0:
        raise ValueError("GIF fps and max_bytes must be positive")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    normalized = tuple(normalize_rgb_frame(frame) for frame in frames)
    if not normalized:
        raise ValueError("Cannot write an empty GIF")
    try:
        for scale in (1.0, 0.875, 0.75, 0.625, 0.5):
            encoded = _scale_frames(normalized, scale)
            with imageio.get_writer(
                temporary,
                format="GIF",
                mode="I",
                duration=1000.0 / fps,
                loop=0,
                palettesize=128,
                subrectangles=True,
            ) as writer:
                for frame in encoded:
                    writer.append_data(frame)
            if temporary.stat().st_size <= max_bytes:
                temporary.replace(output_path)
                return output_path
            temporary.unlink()
        raise ValueError("Showcase GIF exceeds the configured byte ceiling")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _scale_frames(
    frames: tuple[NDArray[np.uint8], ...], scale: float
) -> tuple[NDArray[np.uint8], ...]:
    if scale == 1.0:
        return frames
    height, width = frames[0].shape[:2]
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return tuple(cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in frames)


def read_video_frames(path: Path) -> tuple[NDArray[np.uint8], ...]:
    frames = tuple(normalize_rgb_frame(frame) for frame in imageio.get_reader(path))
    if not frames:
        raise ValueError("Cannot read an empty video")
    return frames
