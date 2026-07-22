from pathlib import Path

import pytest

from botcolosseo.data.demonstrations import generate_demonstration_split, sha256_file


@pytest.mark.integration
@pytest.mark.timeout(90)
def test_real_200_transition_generation_is_reproducible(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    kwargs = {
        "root": root,
        "split": "train",
        "cases_path": root / "configs/m2/train.json",
        "transitions": 200,
        "shard_size": 100,
        "case_transition_cap": 40,
    }

    first = generate_demonstration_split(output_dir=tmp_path / "first", **kwargs)
    second = generate_demonstration_split(output_dir=tmp_path / "second", **kwargs)

    assert first["transitions"] == second["transitions"] == 200
    assert first["privileged_leak_count"] == 0
    assert first["test_cases_accessed"] is False
    assert first["opponent_counts"] == {name: 40 for name in first["opponent_counts"]}
    assert [item["trajectory_sha256"] for item in first["shards"]] == [
        item["trajectory_sha256"] for item in second["shards"]
    ]
    for directory, manifest in ((tmp_path / "first", first), (tmp_path / "second", second)):
        assert all(
            sha256_file(directory / item["file"]) == item["sha256"]
            for item in manifest["shards"]
        )
