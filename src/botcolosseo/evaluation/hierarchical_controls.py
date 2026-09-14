"""Requested/applied control timing from actual contiguous learner decisions."""


def summarize_control_trace(trace, schedule):
    if not trace or [r["decision"] for r in trace] != list(range(len(trace))):
        raise ValueError("Expected a nonempty contiguous learner control trace")
    times = [t for t, _ in schedule]
    if not times or times[0] != 0 or times != sorted(set(times)):
        raise ValueError("Expected ordered control schedule beginning at zero")
    for row in trace:
        expected = next(c for t, c in reversed(schedule) if t <= row["decision"])
        if (
            tuple(row["requested_style"]) != tuple(expected[:3])
            or row["requested_difficulty"] != expected[3]
        ):
            raise ValueError("Recorded requests differ from the declared schedule")
    events = []
    for index, (start, requested) in enumerate(schedule[1:], 1):
        previous = schedule[index - 1][1]
        stop = schedule[index + 1][0] if index + 1 < len(schedule) else len(trace)
        window = trace[start:stop]
        result = {"requested_decision": start, "requested_condition": list(requested)}
        for axis, target, old, key, limit in (
            ("style", tuple(requested[:3]), tuple(previous[:3]), "style", 8),
            ("low_difficulty", requested[3], previous[3], "difficulty", 0),
            ("high_difficulty", requested[3], previous[3], "high_difficulty", 8),
        ):
            status, applied = "unchanged", None
            if target != old:
                status = "not_reached" if start >= len(trace) else "not_applied_before_end"
                for row in window:
                    value = tuple(row[key]) if axis == "style" else row[key]
                    if value == target:
                        applied = row["decision"]
                        status = "applied" if applied - start <= limit else "late"
                        if axis != "low_difficulty" and not row["replanned"]:
                            raise ValueError("High control changed outside a replan boundary")
                        break
                if status == "not_applied_before_end" and len(window) > limit:
                    status = "missed_deadline"
            result[axis] = {
                "status": status,
                "applied_decision": applied,
                "delay_decisions": None if applied is None else applied - start,
                "delay_game_seconds": None if applied is None else (applied - start) * 4 / 35,
            }
        events.append(result)
    return {
        "learner_decisions": len(trace),
        "events": events,
        "scope": "input-application latency; not evidence of behavioral style or task success",
    }
