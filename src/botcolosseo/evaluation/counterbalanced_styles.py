"""Phase-matched descriptive analysis; repeated orders are not independent cases."""

from collections import defaultdict
from itertools import permutations
from statistics import mean

from botcolosseo.demo.control_trace_audit import audit_control_report

STYLES = ("aggressive", "defensive", "explorer")


def summarize(reports):
    orders = {}
    identity = None
    keys = None
    prefixes = {}
    prefix_differences = []
    groups = defaultdict(list)
    outcomes = {}
    for report in reports:
        audit_control_report(report)
        if report["switch_mode"] != "style" or report["difficulty"] != 1:
            raise ValueError("Requires style-only Hard evaluations")
        current = tuple(report[k] for k in ("executor", "strategy", "opponent"))
        if identity is not None and current != identity:
            raise ValueError("Checkpoint identities differ")
        identity = current
        order = tuple(report.get("style_order", STYLES))
        if order in orders:
            raise ValueError("Duplicate order")
        cases = {(c["seed"], c["role"]): c for c in report["cases"]}
        if keys is not None and set(cases) != keys:
            raise ValueError("Case sets differ")
        keys = set(cases)
        orders[order] = cases
        outcomes["-".join(order)] = {
            "mean_banked_value": mean(c["payoff"] * 150 for c in cases.values()),
            "positive_value_cases": sum(c["payoff"] > 0 for c in cases.values()),
            "cases": len(cases),
        }
        for key, case in cases.items():
            trace = case["control_trace"]
            prefix = trace[:81]
            if key in prefixes:
                reference = prefixes[key]
                prefix_differences.append(
                    {
                        "seed": key[0],
                        "role": key[1],
                        "order": list(order),
                        "action_differences": sum(
                            a["action"] != b["action"]
                            for a, b in zip(reference, prefix, strict=True)
                        ),
                        "command_differences": sum(
                            a["command"] != b["command"]
                            for a, b in zip(reference, prefix, strict=True)
                        ),
                        "exact_trace_equal": reference == prefix,
                    }
                )
            else:
                prefixes[key] = prefix
            for phase, (start, stop) in enumerate(((88, 161), (168, 241), (248, 321)), 1):
                style = order[phase - 1]
                target = tuple(int(s == style) for s in STYLES)
                rows = trace[start:stop]
                if not rows:
                    continue
                if any(tuple(r["style"]) != target for r in rows):
                    raise ValueError("Window contains a different applied style")
                replans = [r for r in rows if r["replanned"]]
                groups[(phase, style)].append(
                    {
                        "seed": key[0],
                        "role": key[1],
                        "order": list(order),
                        "decisions": len(rows),
                        "attack_fraction": mean(r["action"] >= 9 for r in rows),
                        "search_fraction": mean(r["command"] <= 2 for r in rows),
                        "engage_fraction": mean(r["command"] == 3 for r in rows),
                        "extract_fraction": mean(r["command"] >= 5 for r in rows),
                        "search_regions": len({r["command"] for r in replans if r["command"] <= 2}),
                    }
                )
    if set(orders) != set(permutations(STYLES)):
        raise ValueError("All six orders required")
    first_style_reproducible = True
    for style in STYLES:
        matching = [cases for order, cases in orders.items() if order[0] == style]
        for key in keys:
            if matching[0][key]["control_trace"][:161] != matching[1][key]["control_trace"][:161]:
                first_style_reproducible = False
    metrics = (
        "attack_fraction",
        "search_fraction",
        "engage_fraction",
        "extract_fraction",
        "search_regions",
    )
    phases = {}
    for (phase, style), rows in groups.items():
        # Average repeats within each paired case before averaging cases.
        paired = defaultdict(list)
        for row in rows:
            paired[(row["seed"], row["role"])].append(row)
        phases[f"phase_{phase}_{style}"] = {
            "unique_cases": len(paired),
            "windows": len(rows),
            "short_windows": sum(r["decisions"] < 73 for r in rows),
            **{
                metric: mean(mean(r[metric] for r in repeats) for repeats in paired.values())
                for metric in metrics
            },
        }
    return {
        "identity": dict(zip(("executor", "strategy", "opponent"), identity, strict=True)),
        "episodes": sum(len(cases) for cases in orders.values()),
        "unique_cases": len(keys),
        "unique_layouts": len({k[0] for k in keys}),
        "pre_switch_prefix_identical": all(r["exact_trace_equal"] for r in prefix_differences),
        "prefix_comparisons": prefix_differences,
        "first_style_prefix_reproducible": first_style_reproducible,
        "windows": [[88, 161], [168, 241], [248, 321]],
        "scope": (
            "Matched seed/role development diagnostic, not exact state replay: inspect prefix "
            "differences. Later phases include carryover and survival effects. Not independent "
            "generalization, combat effectiveness, or a formal skill-retention gate."
        ),
        "phases": phases,
        "outcomes": outcomes,
    }
