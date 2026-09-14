from botcolosseo.cli.compare_hierarchical_mixtures import sample_indices


def test_common_opponents_and_role_specific_fixed_baselines():
    sizes = {"host": 5, "opponent": 5}
    meta = {"host": [0, 0, 1, 0, 0], "opponent": [0, 0, 0, 1, 0]}
    for role in sizes:
        for seed in range(116, 128):
            results = {
                m: sample_indices(seed, 0, role, m, sizes, meta)
                for m in ("fixed", "uniform", "meta")
            }
            assert len({opponent for _, opponent in results.values()}) == 1
            assert results["fixed"][0] == (1 if role == "host" else 0)
            assert results["meta"][0] == (2 if role == "host" else 3)
            assert results["uniform"] == sample_indices(seed, 0, role, "uniform", sizes, meta)
