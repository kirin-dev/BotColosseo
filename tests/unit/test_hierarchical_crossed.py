import pytest

from botcolosseo.evaluation.hierarchical_crossed import audit_crossed_search


def test_crossed_grid_and_observation_equality_are_distinct():
    rows = [
        dict(
            seed=0,
            side=side,
            command=command,
            initial_frame_sha256="a" * 64,
            initial_scalars=[0.0] * 9,
            applicable=True,
        )
        for side in ("host", "opponent")
        for command in ("SEARCH_NORTH", "SEARCH_CENTER", "SEARCH_SOUTH")
    ]
    report = dict(complete=True, privileged_oracle=True, cases=rows)
    assert (
        audit_crossed_search(report, seeds=frozenset({0}))["equal_initial_fair_observation_groups"]
        == 2
    )
    rows[0]["initial_frame_sha256"] = "b" * 64
    result = audit_crossed_search(report, seeds=frozenset({0}))
    assert result["equal_initial_fair_observation_groups"] == 1
    assert result["mismatched_groups"] == [dict(seed=0, side="host")]
    rows.append(rows[0])
    with pytest.raises(ValueError, match="grid"):
        audit_crossed_search(report, seeds=frozenset({0}))
