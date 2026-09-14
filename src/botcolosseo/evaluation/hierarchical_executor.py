"""Descriptive command diagnostics, not independent skill-gate evidence."""

from statistics import mean

from botcolosseo.training.hierarchical_protocol import Command


def summarize_executor(report: dict) -> dict:
    """Summarize a completed fixed-schedule report without pooling missing cases."""
    if not report.get("complete") or not report.get("cases"):
        raise ValueError("A completed nonempty evaluation is required")
    cases = report["cases"]
    keys = [(case["seed"], case["side"]) for case in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate seed/role cases")
    commands = {}
    for command in Command:
        records = [
            record
            for case in cases
            for record in case["commands"]
            if record["command"] == command.name
        ]
        applicable = [r for r in records if r["applicable"]]
        completed = [r for r in applicable if r["completed_at"] is not None]
        delays = [r["completed_at"] - r["start"] for r in completed]
        if any(delay <= 0 for delay in delays):
            raise ValueError("Completion must follow command start")
        commands[command.name] = {
            "scheduled": len(records),
            "applicable": len(applicable),
            "successes": len(completed),
            "success_rate": len(completed) / len(applicable) if applicable else None,
            "mean_success_delay_decisions": mean(delays) if delays else None,
        }
    return {
        "cases": len(cases),
        "seeds": len({case["seed"] for case in cases}),
        "positive_banked_rate": mean(case["banked"] > 0 for case in cases),
        "mean_banked": mean(case["banked"] for case in cases),
        "commands": commands,
        "scope": "descriptive fixed-schedule diagnostic; not an independent skill gate",
    }


def compare_executors(baseline: dict, candidate: dict) -> dict:
    """Paired descriptive regression screen; missing opportunities never pass."""
    old, new = summarize_executor(baseline), summarize_executor(candidate)
    # Older reports predate the flag and always evaluated at Hard.
    if baseline.get("difficulty", 1.0) != candidate.get("difficulty", 1.0):
        raise ValueError("Executor comparison difficulty mismatch")
    for key in ("profile", "protocol", "command_schema", "scoring_version", "extraction_endpoint"):
        if key not in baseline or key not in candidate or baseline[key] != candidate[key]:
            raise ValueError("Executor comparison protocol mismatch")
    for report in (baseline, candidate):
        if any(
            report.get(key)
            for key in ("privileged_oracle", "oracle_extraction_only", "initialization_baseline")
        ):
            raise ValueError("Regression comparison requires actual fair executors")
    rows = [
        {(c["seed"], c["side"]): c for c in report["cases"]} for report in (baseline, candidate)
    ]
    if rows[0].keys() != rows[1].keys():
        raise ValueError("Executor comparison needs matching seed/role cases")
    if any(rows[0][key]["scenario_hash"] != rows[1][key]["scenario_hash"] for key in rows[0]):
        raise ValueError("Executor comparison scenario mismatch")
    commands = {}
    for command in Command:
        rates = [summary["commands"][command.name]["success_rate"] for summary in (old, new)]
        common_deltas = []
        for key in rows[0]:
            records = [
                {
                    r["start"]: r
                    for r in group[key]["commands"]
                    if r["command"] == command.name and r["applicable"]
                }
                for group in rows
            ]
            for start in records[0].keys() & records[1].keys():
                common_deltas.append(
                    int(records[1][start]["completed_at"] is not None)
                    - int(records[0][start]["completed_at"] is not None)
                )
        delta = rates[1] - rates[0] if None not in rates else None
        commands[command.name] = {
            "baseline_rate": rates[0],
            "candidate_rate": rates[1],
            "rate_delta": delta,
            "baseline_applicable": old["commands"][command.name]["applicable"],
            "candidate_applicable": new["commands"][command.name]["applicable"],
            "common_applicable": len(common_deltas),
            "common_success_delta": mean(common_deltas) if common_deltas else None,
            "descriptive_retention": None if delta is None else delta >= -0.05 - 1e-9,
        }
    return {
        "cases": len(rows[0]),
        "extraction_rate_delta": new["positive_banked_rate"] - old["positive_banked_rate"],
        "mean_banked_delta": new["mean_banked"] - old["mean_banked"],
        "commands": commands,
        "scope": "paired descriptive screen; opportunities are policy-dependent; not promotion",
    }
