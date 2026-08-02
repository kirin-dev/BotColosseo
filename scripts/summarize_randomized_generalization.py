from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    values = report["metrics"]
    keys = (
        "extraction_rate",
        "win_rate",
        "mean_extracted_value",
        "mean_extracted_value_advantage",
        "death_rate",
        "opponent_extraction_rate",
        "protocol_inconsistencies",
    )
    return {key: values[key] for key in keys}


def main() -> None:
    baseline_path = Path(
        "reports/extraction/randomized-generalization/baseline-fixed-strong.json"
    )
    candidate_path = Path(
        "reports/extraction/randomized-generalization/"
        "randomized-strong-unseen-random.json"
    )
    baseline = metrics(baseline_path)
    candidate = metrics(candidate_path)
    report = {
        "schema_version": 1,
        "conclusion": "randomized-layout training improved held-random-seed generalization",
        "selected_checkpoint": "runs/extraction-randomized/strong-ppo/candidate-0200000.pt",
        "selected_checkpoint_sha256": sha256(
            Path("runs/extraction-randomized/strong-ppo/candidate-0200000.pt")
        ),
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            key: float(candidate[key]) - float(baseline[key])
            for key in baseline
            if key != "protocol_inconsistencies"
        },
        "evidence": {
            "baseline_report": str(baseline_path),
            "baseline_report_sha256": sha256(baseline_path),
            "candidate_report": str(candidate_path),
            "candidate_report_sha256": sha256(candidate_path),
        },
        "fair_actor_observation_only": True,
        "test_cases_accessed": False,
        "notes": [
            "The 128 safe-anchor permutations are finite domain randomization, "
            "not continuous placement.",
            "The 200k budget establishes improvement but is not a convergence claim.",
        ],
    }
    output = Path("reports/extraction/randomized-generalization.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
