from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np

from botcolosseo.data.schema import DEMONSTRATION_FIELDS, validate_demonstration_shard


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def write_demonstration_shard(
    arrays: dict[str, np.ndarray],
    output_path: Path,
    *,
    require_all_valid: bool = True,
) -> Path:
    validate_demonstration_shard(arrays, require_all_valid=require_all_valid)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in DEMONSTRATION_FIELDS:
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 0
                archive.writestr(info, _npy_bytes(arrays[name]), compresslevel=9)
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def load_demonstration_shard(
    path: Path, *, require_all_valid: bool = True
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    validate_demonstration_shard(arrays, require_all_valid=require_all_valid)
    return arrays


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_sha256(
    arrays: dict[str, np.ndarray], *, require_all_valid: bool = True
) -> str:
    validate_demonstration_shard(arrays, require_all_valid=require_all_valid)
    digest = hashlib.sha256()
    for name in DEMONSTRATION_FIELDS:
        if name == "frame":
            continue
        digest.update(name.encode("utf-8"))
        digest.update(_npy_bytes(arrays[name]))
    return digest.hexdigest()
