"""Validate recorded controls against a predeclared schedule, not each other."""

from botcolosseo.demo.hierarchical_controls import control_schedule


def audit_control_report(report):
    if not report.get("complete") or not report.get("cases"):
        raise ValueError("Complete nonempty report required")
    schedule = [(t, list(c.as_tuple())) for t, c in control_schedule(
        report["switch_mode"], difficulty=report["difficulty"]
    )]
    if report["control_schedule"] != [[t, c] for t, c in schedule]:
        raise ValueError("Schedule differs from declared protocol")
    seen = set()
    reached = applied = pending = 0
    delays = []
    for case in report["cases"]:
        key = (case["seed"], case["role"])
        if key in seen:
            raise ValueError("Duplicate case")
        seen.add(key)
        trace = case["control_trace"]
        if not trace or len(trace) != case["decisions"]:
            raise ValueError("Incomplete decision trace")
        high = None
        for decision, row in enumerate(trace):
            if row["decision"] != decision:
                raise ValueError("Missing or reordered decision")
            expected = next(c for t, c in reversed(schedule) if t <= decision)
            if list(row["requested_style"]) + [row["requested_difficulty"]] != expected:
                raise ValueError("Request differs from schedule")
            if row["difficulty"] != expected[3]:
                raise ValueError("Low-level difficulty not applied immediately")
            if row["replanned"]:
                high = expected
            if high is None or list(row["style"]) + [row["high_difficulty"]] != high:
                raise ValueError("High-level condition changed outside its boundary")
        for start, condition in schedule[1:]:
            rows = trace[start:]
            if not rows:
                continue
            reached += 1
            match = next((r for r in rows if list(r["style"]) +
                          [r["high_difficulty"]] == condition), None)
            if match is None:
                if len(rows) >= 8:
                    raise ValueError("Missed high-level application deadline")
                pending += 1
            else:
                delay = match["decision"] - start
                if delay > 7:
                    raise ValueError("Late high-level application")
                applied += 1
                delays.append(delay)
    return {"cases": len(seen), "reached": reached, "applied": applied,
            "pending_at_episode_end": pending, "max_delay": max(delays, default=None)}
