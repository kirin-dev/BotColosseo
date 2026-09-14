"""Bind unchanged historical actors to one candidate executor for regression only."""

import argparse
import json
from pathlib import Path

import torch

from botcolosseo.cli.train_hierarchical_strategic import digest
from botcolosseo.evaluation.hierarchical_role_paths import role_paths


def prepare_candidate(executor, solution_path, population, output):
    if output.exists():
        raise FileExistsError("Preserve existing candidate bundle")
    solution = json.loads(solution_path.read_text())
    role_paths(solution["identity"], population)
    payload = torch.load(executor, map_location="cpu", weights_only=False)
    old_executor = solution["identity"]["executor"]
    if payload["identity"].get("ppo_initial") != old_executor:
        raise ValueError("Candidate does not descend from this population executor")
    upgrade = payload["identity"].get("executor_upgrade", {})
    if upgrade.get("solution") != digest(solution_path):
        raise ValueError("Candidate was trained against another population solution")
    new_executor = digest(executor)
    actors = []
    for path in population:
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item["identity"]["executor"] != old_executor:
            raise ValueError("Mixed executor ancestry in historical population")
        actors.append(item["actor"])
    output.mkdir(parents=True)
    mapping = []
    for index, (path, actor) in enumerate(zip(population, actors, strict=True)):
        candidate = output / f"strategy-{index}.pt"
        torch.save(
            {
                "actor": actor,
                "identity": {
                    "executor": new_executor,
                    "original_strategy": digest(path),
                    "original_executor": old_executor,
                    "population_solution": digest(solution_path),
                    "scope": "unchanged high actor paired with candidate executor; not promoted",
                },
            },
            candidate,
        )
        mapping.append(
            {
                "source": str(path),
                "source_sha256": digest(path),
                "candidate": str(candidate),
                "candidate_sha256": digest(candidate),
            }
        )
    original_hashes = solution["identity"]["strategies"]
    manifest = {
        "status": "candidate_only_requires_regression",
        "executor": str(executor),
        "executor_sha256": new_executor,
        "previous_executor_sha256": old_executor,
        "source_solution_sha256": digest(solution_path),
        "strategies": mapping,
        "role_indices": {
            role: [
                original_hashes.index(key)
                for key in solution["identity"].get(f"{role}_strategies", original_hashes)
            ]
            for role in ("host", "opponent")
        },
        "matrix_status": "missing; all cells must be evaluated with candidate executor",
    }
    temporary = output / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(output / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    for name in ("executor", "solution", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--population", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_candidate(args.executor, args.solution, args.population, args.output)))


if __name__ == "__main__":
    main()
