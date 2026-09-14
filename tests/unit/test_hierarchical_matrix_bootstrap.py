import json

import numpy as np
import pytest

from botcolosseo.cli.bootstrap_hierarchical_matrix import cluster_cube


@pytest.mark.parametrize("condition", [None, [1, 0, 0, 1]])
def test_raw_case_clusters_reconstruct_both_role_matrices(tmp_path, condition):
    matrix = {
        "complete": True,
        "row_role": "host",
        "column_role": "opponent",
        "identity": {"executor": "low", "strategies": ["high"], "seeds": [1, 2], "repeats": 1},
        "counts": [[4]],
        "A": [[0.25]],
        "B": [[0.5]],
    }
    pair = {
        "complete": True,
        "identity": {
            "executor": "low",
            "first": "high",
            "second": "high",
            "seeds": [1, 2],
            "repeats": 1,
            "scenario": "s",
            "protocol": "p",
            "sources": {},
        },
        "cases": [
            {
                "seed": s,
                "layout": s,
                "repeat": 0,
                "first_side": side,
                "host_payoff": 0.25,
                "opponent_payoff": 0.5,
            }
            for s in (1, 2)
            for side in ("host", "opponent")
        ],
    }
    if condition is not None:
        matrix["identity"]["condition"] = condition
        pair["identity"]["condition"] = condition
    (tmp_path / "matrix.json").write_text(json.dumps(matrix))
    path = tmp_path / "pair-0-0.json"
    path.write_text(json.dumps(pair))
    cube, _ = cluster_cube(tmp_path)
    assert cube.shape == (2, 1, 1, 2)
    np.testing.assert_allclose(cube[:, 0, 0], [[0.25, 0.5]] * 2)
    pair["identity"]["condition"] = [0, 1, 0, 0.5]
    path.write_text(json.dumps(pair))
    with pytest.raises(ValueError, match="identity"):
        cluster_cube(tmp_path)
    pair["identity"].pop("condition")
    if condition is not None:
        pair["identity"]["condition"] = condition
    pair["cases"][0]["host_payoff"] = 0.75
    path.write_text(json.dumps(pair))
    with pytest.raises(ValueError, match="reconstruct"):
        cluster_cube(tmp_path)
