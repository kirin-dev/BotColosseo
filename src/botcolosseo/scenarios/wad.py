from __future__ import annotations

import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_LUMP_NAME = re.compile(r"[A-Z0-9_]{1,8}")


@dataclass(frozen=True)
class WadLump:
    name: str
    data: bytes

    def __post_init__(self) -> None:
        if _LUMP_NAME.fullmatch(self.name) is None:
            raise ValueError(f"WAD lump name must be 1-8 uppercase ASCII characters: {self.name!r}")


@dataclass(frozen=True)
class WadEntry:
    name: str
    offset: int
    size: int


def build_pwad(lumps: Iterable[WadLump]) -> bytes:
    ordered = tuple(lumps)
    body = bytearray()
    entries: list[WadEntry] = []
    for lump in ordered:
        offset = 12 + len(body)
        body.extend(lump.data)
        entries.append(WadEntry(name=lump.name, offset=offset, size=len(lump.data)))
    directory_offset = 12 + len(body)
    directory = bytearray()
    for entry in entries:
        name = entry.name.encode("ascii").ljust(8, b"\0")
        directory.extend(struct.pack("<II8s", entry.offset, entry.size, name))
    return struct.pack("<4sII", b"PWAD", len(entries), directory_offset) + body + directory


def inspect_pwad(data: bytes) -> tuple[WadEntry, ...]:
    if len(data) < 12:
        raise ValueError("WAD is shorter than its header")
    magic, count, directory_offset = struct.unpack_from("<4sII", data)
    if magic != b"PWAD":
        raise ValueError(f"Expected PWAD magic, got {magic!r}")
    if directory_offset + count * 16 > len(data):
        raise ValueError("WAD directory extends past the file")
    entries = []
    for index in range(count):
        offset, size, raw_name = struct.unpack_from("<II8s", data, directory_offset + index * 16)
        if offset + size > directory_offset:
            raise ValueError("WAD lump extends into or past the directory")
        name = raw_name.rstrip(b"\0").decode("ascii")
        entries.append(WadEntry(name=name, offset=offset, size=size))
    return tuple(entries)


def write_pwad(lumps: Iterable[WadLump], output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        temporary_path.write_bytes(build_pwad(lumps))
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path
