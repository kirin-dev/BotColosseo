import struct
from pathlib import Path

import pytest

from botcolosseo.scenarios.wad import WadLump, build_pwad, inspect_pwad, write_pwad


def test_build_pwad_has_deterministic_header_and_directory() -> None:
    data = build_pwad((WadLump("MAP01", b""), WadLump("TEXTMAP", b"abc")))

    magic, count, directory_offset = struct.unpack_from("<4sII", data)
    assert (magic, count, directory_offset) == (b"PWAD", 2, 15)
    assert [(entry.name, entry.size) for entry in inspect_pwad(data)] == [
        ("MAP01", 0),
        ("TEXTMAP", 3),
    ]


def test_build_pwad_rejects_invalid_lump_name() -> None:
    with pytest.raises(ValueError, match="1-8 uppercase ASCII"):
        build_pwad((WadLump("too_long_name", b""),))


def test_write_pwad_is_atomic(tmp_path: Path) -> None:
    output = tmp_path / "scenario.wad"

    result = write_pwad((WadLump("MAP01", b""),), output)

    assert result == output.resolve()
    assert output.read_bytes().startswith(b"PWAD")
    assert not tuple(tmp_path.glob(".*.tmp"))
