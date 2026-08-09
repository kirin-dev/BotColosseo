from __future__ import annotations

import argparse
import json
from pathlib import Path

from botcolosseo.data.demonstrations import sha256_file
from botcolosseo.evaluation.extraction import ExtractionEpisodeMetrics
from botcolosseo.evaluation.extraction_gates import _style_score
from botcolosseo.evaluation.extraction_protocol import (
    load_extraction_evaluation_protocol,
)

POLICIES = ("strong", "aggressive", "defensive", "explorer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select representative validation-only Extraction replays"
    )
    for policy in POLICIES:
        parser.add_argument(f"--{policy}-report", type=Path, required=True)
        parser.add_argument(f"--{policy}-manifest", type=Path, required=True)
        parser.add_argument(
            f"--{policy}-case-index",
            type=int,
            help="Bind a validation case whose live replay has already been verified",
        )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/extraction/evaluation.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _score(
    policy: str,
    episode: ExtractionEpisodeMetrics,
    strong: ExtractionEpisodeMetrics | None = None,
) -> tuple[float | int, ...]:
    duration_ok = int(episode.decisions >= 240)
    if policy == "strong":
        return (
            int(episode.extracted),
            duration_ok,
            int(episode.won),
            episode.extracted_value,
            episode.valid_hits,
        )
    if policy == "aggressive":
        paired_delta = (
            _style_score(policy, episode) - _style_score(policy, strong)
            if strong is not None
            else 0.0
        )
        return (
            int(episode.aggressive_chains > 0),
            int(episode.extracted),
            duration_ok,
            paired_delta,
            episode.aggressive_chains,
            episode.kills,
            episode.cache_looted,
            episode.valid_hits,
            episode.extracted_value,
        )
    if policy == "defensive":
        paired_delta = (
            _style_score(policy, episode) - _style_score(policy, strong)
            if strong is not None
            else 0.0
        )
        return (
            int(
                episode.meaningful_extractions > 0
                and episode.extracted
                and not episode.died
            ),
            int(episode.successful_disengagements > 0),
            duration_ok,
            int(episode.kills == 0),
            paired_delta,
            episode.extracted_value,
            episode.successful_disengagements,
            -episode.combat_with_meaningful_value,
        )
    if policy == "explorer":
        paired_delta = (
            _style_score(policy, episode) - _style_score(policy, strong)
            if strong is not None
            else 0.0
        )
        return (
            int(episode.upgrade_to_extraction_conversions > 0),
            int(episode.extracted),
            duration_ok,
            int(episode.valid_hits == 0 and episode.kills == 0),
            paired_delta,
            episode.backpack_upgrades,
            episode.meaningful_loot_regions,
            episode.extracted_value,
        )
    raise ValueError("Unknown Extraction showcase policy")


def _representative(
    policy: str,
    episode: ExtractionEpisodeMetrics,
    strong: ExtractionEpisodeMetrics,
) -> bool:
    if not episode.extracted or episode.decisions < 240 or episode.truncated:
        return False
    if policy == "aggressive":
        complete = (
            episode.aggressive_chains > 0
            and episode.valid_hits > 0
            and episode.kills > 0
            and episode.cache_looted > 0
        )
    elif policy == "defensive":
        complete = (
            episode.successful_disengagements > 0
            and episode.meaningful_extractions > 0
            and not episode.died
        )
    elif policy == "explorer":
        complete = (
            episode.backpack_upgrades > 0
            and episode.upgrade_to_extraction_conversions > 0
            and episode.meaningful_loot_regions > 0
        )
    else:
        return policy == "strong"
    return complete and _style_score(policy, episode) > _style_score(policy, strong)


def _evidence_tier(
    policy: str,
    report: dict[str, object],
    manifest: dict[str, object],
) -> str:
    checkpoint_sha256 = report.get("checkpoint_sha256")
    if policy == "strong":
        if (
            manifest.get("gate_schema_version") == 2
            and manifest.get("eligible") is True
            and manifest.get("selected_checkpoint_sha256") == checkpoint_sha256
        ):
            valid = True
            tier = "research_selection"
        elif (
            manifest.get("admission_kind") == "strong_product_showcase"
            and manifest.get("product_showcase_eligible") is True
            and manifest.get("research_gate_passed") is False
            and manifest.get("official_test_eligible") is False
            and manifest.get("showcase_checkpoint_sha256") == checkpoint_sha256
        ):
            valid = True
            tier = "product_showcase"
        else:
            valid = False
            tier = "unknown"
    elif (
        manifest.get("gate_schema_version") == 2
        and manifest.get("eligible") is True
    ):
        valid = manifest.get("selected_checkpoint_sha256") == checkpoint_sha256
        tier = "research_selection"
    elif manifest.get("admission_kind") == "directional_showcase":
        valid = (
            manifest.get("showcase_eligible") is True
            and manifest.get("showcase_checkpoint_sha256") == checkpoint_sha256
        )
        tier = "directional_showcase"
    elif manifest.get("evidence_tier") == "validation_demonstration":
        valid = (
            manifest.get("product_demo_eligible") is True
            and manifest.get("official_test_eligible") is False
            and manifest.get("showcase_checkpoint_sha256") == checkpoint_sha256
        )
        tier = "validation_demonstration"
    elif manifest.get("evidence_tier") == "representative_case_demonstration":
        valid = (
            manifest.get("product_demo_eligible") is True
            and manifest.get("research_gate_passed") is False
            and manifest.get("official_test_eligible") is False
            and manifest.get("claim_scope")
            == "representative_validation_cases_only"
            and manifest.get("aggregate_style_gate_passed") is False
            and isinstance(manifest.get("representative_case_count"), int)
            and manifest["representative_case_count"] > 0
            and manifest.get("showcase_checkpoint_sha256") == checkpoint_sha256
        )
        tier = "representative_case_demonstration"
    else:
        valid = False
        tier = "unknown"
    if (
        manifest.get("policy") != policy
        or manifest.get("test_cases_accessed") is not False
        or not valid
    ):
        raise ValueError(f"{policy} Showcase artifact manifest does not match")
    return tier


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    protocol_path = _resolve(root, args.protocol)
    output = _resolve(root, args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite showcase selection: {output}")
    protocol = load_extraction_evaluation_protocol(protocol_path)
    cases = protocol.cases("validation")
    case_indices = {
        (case.seed, case.learner_side, case.opponent_style): index
        for index, case in enumerate(cases)
    }
    reports: dict[str, tuple[Path, dict[str, object]]] = {}
    for policy in POLICIES:
        report_path = _resolve(root, getattr(args, f"{policy}_report"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("policy") != policy
            or report.get("split") != "validation"
            or report.get("complete") is not True
            or report.get("protocol_sha256") != protocol.sha256
            or report.get("test_cases_accessed") is not False
        ):
            raise ValueError(f"{policy} showcase report identity does not match")
        reports[policy] = (report_path, report)

    _, strong_report = reports["strong"]
    strong_episodes = tuple(
        ExtractionEpisodeMetrics(**item)
        for item in strong_report["metrics"]["episodes"]
    )
    strong_by_case = {
        (item.seed, item.learner_side, item.opponent_style): item
        for item in strong_episodes
    }
    if len(strong_by_case) != len(cases):
        raise ValueError("Strong Showcase report is not uniquely paired")

    selections: dict[str, object] = {}
    for policy in POLICIES:
        report_path, report = reports[policy]
        manifest_path = _resolve(root, getattr(args, f"{policy}_manifest"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        evidence_tier = _evidence_tier(policy, report, manifest)
        episodes = tuple(
            ExtractionEpisodeMetrics(**item)
            for item in report["metrics"]["episodes"]
        )
        by_case = {
            (item.seed, item.learner_side, item.opponent_style): item
            for item in episodes
        }
        if len(by_case) != len(cases) or set(by_case) != set(strong_by_case):
            raise ValueError(f"{policy} Showcase report is not uniquely paired")
        eligible = [
            episode
            for identity, episode in by_case.items()
            if _representative(policy, episode, strong_by_case[identity])
        ]
        if not eligible:
            raise ValueError(f"{policy} has no complete representative replay")
        requested_index = getattr(args, f"{policy}_case_index")
        if requested_index is None:
            selected = max(
                eligible,
                key=lambda item: _score(
                    policy,
                    item,
                    strong_by_case[
                        (item.seed, item.learner_side, item.opponent_style)
                    ],
                ),
            )
        else:
            if not 0 <= requested_index < len(cases):
                raise ValueError(f"{policy} Showcase case index is out of range")
            selected = by_case[
                (
                    cases[requested_index].seed,
                    cases[requested_index].learner_side,
                    cases[requested_index].opponent_style,
                )
            ]
            if selected not in eligible:
                raise ValueError(
                    f"{policy} requested Showcase case is not representative"
                )
        identity = (
            selected.seed,
            selected.learner_side,
            selected.opponent_style,
        )
        paired_strong = strong_by_case[identity]
        selections[policy] = {
            "case_index": case_indices[identity],
            "episode": vars(selected),
            "paired_strong_episode": vars(paired_strong),
            "paired_style_difference": (
                0.0
                if policy == "strong"
                else _style_score(policy, selected)
                - _style_score(policy, paired_strong)
            ),
            "score": list(_score(policy, selected, paired_strong)),
            "validation_report": str(report_path.relative_to(root)),
            "validation_report_sha256": sha256_file(report_path),
            "checkpoint_sha256": report["checkpoint_sha256"],
            "artifact_manifest": str(manifest_path.relative_to(root)),
            "artifact_manifest_sha256": sha256_file(manifest_path),
            "evidence_tier": evidence_tier,
        }
    payload = {
        "schema_version": 2,
        "protocol": str(protocol_path.relative_to(root)),
        "protocol_sha256": protocol.sha256,
        "selection_split": "validation",
        "selections": selections,
        "test_cases_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
