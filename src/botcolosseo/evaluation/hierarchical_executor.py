"""Descriptive command diagnostics, not independent skill-gate evidence."""

import math
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


def decide_executor_promotion(comparison: dict, *, weak_commands: tuple[str, ...]) -> dict:
    """Screen descriptive rates; these inputs can never authorize promotion.

    Formal promotion also needs adequate evaluation coverage and full high-level
    regression evidence, neither of which a fixed-command comparison provides.
    Keep the legacy function name and ``promote`` field for callers, but separate
    a numerical screen pass from the unavailable formal decision.
    """
    names = {command.name for command in Command}
    if (not weak_commands or len(set(weak_commands)) != len(weak_commands)
            or not set(weak_commands) <= names):
        raise ValueError("Need unique known pre-specified weak commands")

    def rejected(reason, **details):
        return {"promote": False, "screening_passed": False, "reason": reason, **details}

    def finite(value):
        return (isinstance(value, (int, float))
                and not isinstance(value, bool) and math.isfinite(value))

    commands = comparison.get("commands", {})
    missing = sorted(names - commands.keys())
    missing += sorted(name for name in names & commands.keys() if any(
        commands[name].get(key) is None for key in ("baseline_rate", "candidate_rate", "rate_delta")
    ))
    if missing:
        return rejected("missing_comparable_command", commands=missing)
    deltas = {}
    for name in sorted(names):
        row = commands[name]
        old, new, delta = (row[key] for key in ("baseline_rate", "candidate_rate", "rate_delta"))
        if (not all(finite(v) for v in (old, new, delta))
                or not 0 <= old <= 1 or not 0 <= new <= 1
                or not math.isclose(delta, new - old, abs_tol=1e-9, rel_tol=0)):
            return rejected("invalid_command_rates", command=name)
        counts = [row.get(k) for k in
                  ("baseline_applicable", "candidate_applicable", "common_applicable")]
        if (any(type(n) is not int or n <= 0 for n in counts)
                or counts[2] > min(counts[:2])):
            return rejected("missing_comparable_opportunities", command=name)
        deltas[name] = new - old
    extraction_delta = comparison.get("extraction_rate_delta")
    if not finite(extraction_delta) or not -1 <= extraction_delta <= 1:
        return rejected("invalid_extraction_delta")
    if extraction_delta < -0.05 - 1e-9:
        return rejected("extraction_regression")
    regressions = sorted(name for name, delta in deltas.items() if delta < -0.05 - 1e-9)
    if regressions:
        return rejected("command_regression", regressions=regressions)
    weak_improvements = [name for name in weak_commands if deltas[name] > 0]
    if not weak_improvements:
        return rejected("no_weak_skill_improvement", weak_commands=list(weak_commands))
    return {
        "promote": False,
        "screening_passed": True,
        "reason": "formal_promotion_evidence_missing",
        "missing_evidence": ["sample_adequacy", "full_high_level_regression"],
        "improved_weak_commands": weak_improvements,
    }
