import hashlib

import pytest

from botcolosseo.evaluation.hierarchical_role_paths import role_paths


def test_distinct_role_order(tmp_path):
    paths = [tmp_path / str(i) for i in range(3)]
    for i, path in enumerate(paths):
        path.write_bytes(bytes([i]))
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    identity = {"strategies": hashes, "host_strategies": hashes[:2],
                "opponent_strategies": [hashes[2], hashes[0]]}
    result = role_paths(identity, paths)
    assert result["host"] == paths[:2]
    assert result["opponent"] == [paths[2], paths[0]]
    with pytest.raises(ValueError):
        role_paths(identity, paths[::-1])
