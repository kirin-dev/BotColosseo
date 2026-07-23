from __future__ import annotations

from collections.abc import Mapping, Sequence

_STYLES = ("aggressive", "defensive", "explorer")


def rank_user_study_candidates(
    *,
    aggressive_records: Sequence[Mapping[str, object]],
    defensive_records: Sequence[Mapping[str, object]],
    explorer_records: Sequence[Mapping[str, object]],
    limit: int = 4,
) -> dict[str, tuple[dict[str, object], ...]]:
    if limit < 2:
        raise ValueError("User-study candidate limit must be at least two")
    sources = {
        "aggressive": _policy_rows(aggressive_records, "aggressive"),
        "defensive": _policy_rows(defensive_records, "defensive"),
        "explorer": _policy_rows(explorer_records, "explorer"),
    }
    rankings = {
        "aggressive": sorted(
            sources["aggressive"],
            key=lambda item: (
                -_integer(item[1], "valid_hits"),
                -_integer(item[1], "engagement_initiations"),
                -_integer(item[1], "attack_decisions"),
                _case_id(item[1]),
            ),
        ),
        "defensive": sorted(
            sources["defensive"],
            key=lambda item: (
                -_integer(item[1], "low_health_opportunities"),
                -int(_integer(item[1], "recoveries") > 0),
                -_integer(item[1], "successful_escapes"),
                -_integer(item[1], "risk_decisions"),
                _case_id(item[1]),
            ),
        ),
        "explorer": sorted(
            sources["explorer"],
            key=lambda item: (
                -_route_type_count(item[1]),
                -_number(item[1], "route_entropy"),
                -_integer(item[1], "completed_routes"),
                -_integer(item[1], "unique_regions"),
                _case_id(item[1]),
            ),
        ),
    }
    result = {}
    for style in _STYLES:
        rows = []
        for ordinal, record in rankings[style][:limit]:
            rows.append(
                {
                    "style": style,
                    "case_id": _case_id(record),
                    "formal_episode_ordinal": ordinal,
                    "formal_metrics": dict(record),
                }
            )
        if len(rows) != limit:
            raise ValueError(f"Insufficient {style} user-study candidates")
        result[style] = tuple(rows)
    return result


def _policy_rows(
    records: Sequence[Mapping[str, object]],
    policy: str,
) -> list[tuple[int, Mapping[str, object]]]:
    rows = [row for row in records if row.get("policy") == policy]
    if not rows:
        raise ValueError(f"Missing {policy} formal records")
    result = []
    for ordinal, row in enumerate(rows):
        if (
            row.get("split") != "validation"
            or row.get("objective_completed") is not True
            or row.get("protocol_inconsistent") is not False
            or row.get("environment_attempts") != 1
            or (row.get("terminated") is not True and row.get("truncated") is not True)
        ):
            continue
        _case_id(row)
        result.append((ordinal, row))
    if not result:
        raise ValueError(f"No eligible {policy} formal records")
    return result


def _case_id(row: Mapping[str, object]) -> str:
    opponent = row.get("opponent")
    pair_index = row.get("pair_index")
    learner_side = row.get("learner_side")
    if (
        not isinstance(opponent, str)
        or not opponent
        or type(pair_index) is not int
        or learner_side not in ("host", "opponent")
    ):
        raise ValueError("User-study candidate case identity is invalid")
    return f"{opponent}:{pair_index}:{learner_side}"


def _route_type_count(row: Mapping[str, object]) -> int:
    return sum(
        _integer(row, field) > 0
        for field in (
            "upper_completions",
            "lower_completions",
            "flank_completions",
        )
    )


def _integer(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if type(value) is not int or value < 0:
        raise ValueError(f"User-study candidate {field} is invalid")
    return value


def _number(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"User-study candidate {field} is invalid")
    return float(value)
