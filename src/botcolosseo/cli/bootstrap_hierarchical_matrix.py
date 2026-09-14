"""Layout-cluster resampling of complete, role-aware empirical game artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np

from botcolosseo.cli.solve_hierarchical_matrix import solve_report
from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.training.hierarchical_game import solve_bimatrix


def cluster_cube(directory):
    matrix_path = directory / "matrix.json"
    matrix = json.loads(matrix_path.read_text())
    solve_report(matrix)
    identity = matrix["identity"]
    strategies = identity["strategies"]
    rows = identity.get("host_strategies", strategies)
    columns = identity.get("opponent_strategies", strategies)
    seeds = identity["seeds"]
    if len(set(seeds)) != len(seeds) or len({s % 128 for s in seeds}) != len(seeds):
        raise ValueError("Expected distinct layout clusters")
    needed = {(strategies.index(h), strategies.index(o)) for h in rows for o in columns}
    pairs = sorted(needed | {(j, i) for i, j in needed})
    cells = {(h, o, s): [] for h in rows for o in columns for s in seeds}
    sources, protocols = {}, set()
    for i, j in pairs:
        path = directory / f"pair-{i}-{j}.json"
        report = json.loads(path.read_text())
        ident = report["identity"]
        expected = {
            (s, r, side)
            for s in seeds
            for r in range(identity["repeats"])
            for side in ("host", "opponent")
        }
        observed = [(c["seed"], c["repeat"], c["first_side"]) for c in report["cases"]]
        if (
            not report["complete"]
            or set(observed) != expected
            or len(observed) != len(expected)
            or ident["executor"] != identity["executor"]
            or ident["first"] != strategies[i]
            or ident["second"] != strategies[j]
            or ident["seeds"] != seeds
            or ident["repeats"] != identity["repeats"]
        ):
            raise ValueError("Pair identity or case budget mismatch")
        protocols.add(
            json.dumps({k: ident[k] for k in ("scenario", "protocol", "sources")}, sort_keys=True)
        )
        sources[path.name] = digest(path)
        for case in report["cases"]:
            if case["layout"] != case["seed"] % 128:
                raise ValueError("Unexpected layout assignment")
            h, o = (i, j) if case["first_side"] == "host" else (j, i)
            key = (strategies[h], strategies[o], case["seed"])
            if key in cells:
                cells[key].append([case["host_payoff"], case["opponent_payoff"]])
    if len(protocols) != 1:
        raise ValueError("Mixed scenario or inference protocols")
    cube = np.empty((len(seeds), len(rows), len(columns), 2))
    for k, seed in enumerate(seeds):
        for i, h in enumerate(rows):
            for j, o in enumerate(columns):
                values = np.asarray(cells[h, o, seed])
                if (
                    values.shape != (identity["repeats"] * 2, 2)
                    or not np.isfinite(values).all()
                    or (values < 0).any()
                    or (values > 1).any()
                ):
                    raise ValueError("Incomplete or invalid cluster payoff")
                cube[k, i, j] = values.mean(0)
    if not np.allclose(cube.mean(0), np.stack((matrix["A"], matrix["B"]), -1), atol=1e-12, rtol=0):
        raise ValueError("Raw cases do not reconstruct the matrix")
    return cube, {"matrix_sha256": digest(matrix_path), "pairs": sources}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=200)
    args = parser.parse_args()
    if args.draws < 2 or args.output.exists():
        raise ValueError("Need fresh output and at least two draws")
    cube, provenance = cluster_cube(args.directory)
    rng = np.random.default_rng(1701)
    host, opponent, regrets = [], [], []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for draw in range(args.draws):
        sampled = cube[rng.integers(len(cube), size=len(cube))].mean(0)
        result = solve_bimatrix(sampled[:, :, 0], sampled[:, :, 1])
        host.append(result.host.tolist())
        opponent.append(result.opponent.tolist())
        regrets.append(result.residual.maximum_regret)
        report = {
            "complete": draw + 1 == args.draws,
            "draws": draw + 1,
            "requested_draws": args.draws,
            "clusters": len(cube),
            "seed": 1701,
            "host_weights": host,
            "opponent_weights": opponent,
            "maximum_sample_regret": max(regrets),
            "provenance": provenance,
            "scope": "layout-cluster mixture sensitivity; not an equilibrium confidence region",
        }
        for role, values in (("host", host), ("opponent", opponent)):
            report[f"{role}_weight_quantiles"] = np.quantile(
                values, [0.025, 0.5, 0.975], axis=0
            ).tolist()
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(args.output)
        print(json.dumps({"draws": draw + 1, "regret": regrets[-1]}), flush=True)


if __name__ == "__main__":
    main()
