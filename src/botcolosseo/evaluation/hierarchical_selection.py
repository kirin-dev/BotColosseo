"""Role-specific validation screening, separate from independent promotion tests."""

import math


def screen_value(reports, *, role, opponent_mixture):
    if role not in ("host", "opponent"):
        raise ValueError("Invalid learner role")
    if (not opponent_mixture or any(not math.isfinite(p) or p < 0 for p in opponent_mixture)
            or not math.isclose(sum(opponent_mixture), 1, abs_tol=1e-8)):
        raise ValueError("Invalid target marginal")
    support = {i for i, probability in enumerate(opponent_mixture) if probability > 0}
    if set(reports) != support:
        raise ValueError("Need exactly the target support")
    values = None
    for index in sorted(support):
        report = reports[index]
        if not report.get("complete"):
            raise ValueError("Incomplete candidate evaluation")
        expected = {(seed, repeat) for seed in report["identity"]["seeds"]
                    for repeat in range(report["identity"]["repeats"])}
        cases = [case for case in report["cases"] if case["first_side"] == role]
        by_case = {(case["seed"], case["repeat"]): case["first_payoff"] for case in cases}
        if len(by_case) != len(cases) or set(by_case) != expected:
            raise ValueError("Missing or repeated role cases")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in by_case.values()):
            raise ValueError("Invalid candidate payoff")
        if values is None:
            values = {key: 0.0 for key in by_case}
        if set(values) != set(by_case):
            raise ValueError("Opponent reports use different cases")
        for key in values:
            values[key] += opponent_mixture[index] * by_case[key]
    return {"mean_payoff": sum(values.values()) / len(values), "cases": len(values),
            "role": role, "scope": "checkpoint screening only, not response improvement"}
