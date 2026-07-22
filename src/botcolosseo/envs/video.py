from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

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
    try:
        with imageio.get_writer(
            temporary,
            format="GIF",
            mode="I",
            duration=1000.0 / fps,
            loop=0,
        ) as writer:
            count = 0
            for frame in frames:
                writer.append_data(normalize_rgb_frame(frame))
                count += 1
        if count == 0:
            raise ValueError("Cannot write an empty GIF")
        if temporary.stat().st_size > max_bytes:
            raise ValueError("Showcase GIF exceeds the configured byte ceiling")
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def read_video_frames(path: Path) -> tuple[NDArray[np.uint8], ...]:
    frames = tuple(normalize_rgb_frame(frame) for frame in imageio.get_reader(path))
    if not frames:
        raise ValueError("Cannot read an empty video")
    return frames
