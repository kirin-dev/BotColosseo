# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from botcolosseo.cli.select_extraction_candidate import _atomic_json
from botcolosseo.data.demonstrations import sha256_file

# Static SVG templates are clearer when their markup stays on complete lines.
WINDOW_STEPS = 50_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the randomized Strong training-selection figure"
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--screening-root", type=Path, required=True)
    parser.add_argument("--confirmation-report", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_curve_data(
    *,
    root: Path,
    metrics_path: Path,
    screening_root: Path,
    confirmation_path: Path,
    selected_checkpoint: Path,
) -> dict[str, object]:
    selected_sha256 = sha256_file(selected_checkpoint)
    screening: list[dict[str, object]] = []
    screening_paths = sorted(screening_root.glob("candidate-*.json"))
    for path in screening_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = Path(str(report["checkpoint"]))
        steps = int(checkpoint.stem.removeprefix("candidate-"))
        episodes = report["metrics"]["episodes"]
        screening.append(
            {
                "environment_steps": steps,
                "episodes": len(episodes),
                "extraction_rate": sum(item["extracted"] for item in episodes)
                / len(episodes),
                "win_rate": sum(item["won"] for item in episodes) / len(episodes),
                "checkpoint_sha256": report["checkpoint_sha256"],
                "selected": report["checkpoint_sha256"] == selected_sha256,
            }
        )
    if len(screening) != 20 or sum(item["selected"] for item in screening) != 1:
        raise ValueError("Training curve requires 20 screens and one selected point")

    buckets: dict[int, dict[str, list[float]]] = {}
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("kind") != "train":
                continue
            step = int(item["environment_steps"])
            bucket = min(1_000_000, math.ceil(step / WINDOW_STEPS) * WINDOW_STEPS)
            values = buckets.setdefault(bucket, {"agreement": [], "kl": []})
            values["agreement"].append(float(item["replay_agreement"]))
            values["kl"].append(float(item["reference_kl"]))
    diagnostics = [
        {
            "environment_steps": step,
            "replay_agreement": _mean(values["agreement"]),
            "reference_kl": _mean(values["kl"]),
            "updates": len(values["agreement"]),
        }
        for step, values in sorted(buckets.items())
    ]
    if len(diagnostics) != 20:
        raise ValueError("Training diagnostics do not cover all 50k windows")

    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    episodes = confirmation["metrics"]["episodes"]
    if (
        confirmation.get("checkpoint_sha256") != selected_sha256
        or confirmation.get("split") != "validation"
        or len(episodes) != 240
        or confirmation.get("test_cases_accessed") is not False
    ):
        raise ValueError("Training curve confirmation identity does not match")
    sources = [metrics_path, confirmation_path, *screening_paths]
    return {
        "schema_version": 1,
        "selected_checkpoint": str(selected_checkpoint.relative_to(root)),
        "selected_checkpoint_sha256": selected_sha256,
        "screening": screening,
        "confirmation": {
            "environment_steps": 950_000,
            "episodes": 240,
            "extraction_rate": sum(item["extracted"] for item in episodes)
            / len(episodes),
            "win_rate": sum(item["won"] for item in episodes) / len(episodes),
        },
        "training_diagnostics": diagnostics,
        "diagnostic_window_steps": WINDOW_STEPS,
        "metric_definitions": {
            "screening": "frozen 32-episode randomized validation",
            "confirmation": "frozen 240-episode randomized validation",
            "replay_agreement": "training-time action agreement on fixed BC replay",
            "reference_kl": "training-time KL to frozen BC reference policy",
        },
        "sources_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in sources
        },
        "test_cases_accessed": False,
    }


def _points(
    items: list[dict[str, object]],
    key: str,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    maximum: float,
    minimum: float = 0.0,
) -> str:
    return " ".join(
        f"{left + width * float(item['environment_steps']) / 1_000_000:.1f},"
        f"{top + height * (1 - (float(item[key]) - minimum) / (maximum - minimum)):.1f}"
        for item in items
    )


def render_svg(data: dict[str, object]) -> str:
    screening = data["screening"]
    diagnostics = data["training_diagnostics"]
    confirmation = data["confirmation"]
    left_a, left_b, top, width, height = 66, 664, 75, 468, 315
    capability_minimum = 0.25
    extraction = _points(
        screening, "extraction_rate", left=left_a, top=top, width=width,
        height=height, maximum=1.0, minimum=capability_minimum
    )
    wins = _points(
        screening, "win_rate", left=left_a, top=top, width=width,
        height=height, maximum=1.0, minimum=capability_minimum
    )
    agreement = _points(
        diagnostics, "replay_agreement", left=left_b, top=top, width=width,
        height=height, maximum=1.0
    )
    max_kl = max(float(item["reference_kl"]) for item in diagnostics) * 1.1
    kl = _points(
        diagnostics, "reference_kl", left=left_b, top=top, width=width,
        height=height, maximum=max_kl
    )
    confirm_x = left_a + width * float(confirmation["environment_steps"]) / 1e6
    confirm_y = top + height * (
        1
        - (float(confirmation["extraction_rate"]) - capability_minimum)
        / (1.0 - capability_minimum)
    )
    selected = next(item for item in screening if item["selected"])
    selected_y = top + height * (
        1
        - (float(selected["extraction_rate"]) - capability_minimum)
        / (1.0 - capability_minimum)
    )
    grid = []
    for rate in (0.25, 0.5, 0.75, 1.0):
        y = top + height * (1 - (rate - capability_minimum) / (1.0 - capability_minimum))
        grid.append(
            f'<path d="M{left_a} {y:.1f}H{left_a + width}" class="grid"/>'
        )
        grid.append(f'<text x="18" y="{y + 4:.1f}">{rate:.0%}</text>')
    for rate in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + height * (1 - rate)
        grid.append(
            f'<path d="M{left_b} {y:.1f}H{left_b + width}" class="grid"/>'
        )
        grid.append(f'<text x="616" y="{y + 4:.1f}">{rate:.0%}</text>')
    ticks = []
    for step in (0, 250_000, 500_000, 750_000, 1_000_000):
        label = "0" if step == 0 else f"{step // 1000}k"
        for left in (left_a, left_b):
            x = left + width * step / 1_000_000
            ticks.append(f'<text x="{x:.1f}" y="416" text-anchor="middle">{label}</text>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="500" viewBox="0 0 1200 500" role="img" aria-labelledby="title desc">
<title id="title">Randomized Strong training and selection</title>
<desc id="desc">Twenty 32-episode validation screens select the 950k checkpoint, confirmed on 240 episodes. Strong PPO diagnostics show BC replay agreement and KL to the frozen BC reference.</desc>
<style>text{{font:13px Arial,sans-serif;fill:#586171}}.title{{font-size:18px;font-weight:700;fill:#111827}}.label{{font-size:12px;font-weight:700}}.grid{{stroke:#e7eaf0;stroke-width:1}}.axis{{stroke:#9aa3b2}}.extract{{fill:none;stroke:#2878c8;stroke-width:3}}.win{{fill:none;stroke:#6aa8dc;stroke-width:2.5;stroke-dasharray:7 5}}.agree{{fill:none;stroke:#288f67;stroke-width:3}}.kl{{fill:none;stroke:#bd7c16;stroke-width:2.5;stroke-dasharray:7 5}}</style>
<rect width="1200" height="500" rx="18" fill="#fff" stroke="#dfe3e8"/>
<text x="66" y="35" class="title">A · Capability selection</text><text x="664" y="35" class="title">B · Strong PPO diagnostics</text>
{''.join(grid)}
<path d="M{left_a} {top}V{top+height}H{left_a+width} M{left_b} {top}V{top+height}H{left_b+width}" class="axis" fill="none"/>
<polyline points="{extraction}" class="extract"/><polyline points="{wins}" class="win"/>
<polyline points="{agreement}" class="agree"/><polyline points="{kl}" class="kl"/>
<text x="1142" y="{top + 4}" text-anchor="start">{max_kl:.3f}</text><text x="1142" y="{top + height + 4}" text-anchor="start">0 KL</text>
<circle cx="{confirm_x:.1f}" cy="{selected_y:.1f}" r="5" fill="#2878c8"/><circle cx="{confirm_x:.1f}" cy="{confirm_y:.1f}" r="7" fill="#fff" stroke="#df3152" stroke-width="3"/>
<text x="{confirm_x-8:.1f}" y="{confirm_y-14:.1f}" text-anchor="end" class="label">950k · 240-episode confirmation</text>
{''.join(ticks)}
<line x1="66" y1="466" x2="96" y2="466" class="extract"/><text x="104" y="470" class="label">extraction</text>
<line x1="190" y1="466" x2="220" y2="466" class="win"/><text x="228" y="470" class="label">win rate · 32-episode screens</text>
<line x1="664" y1="466" x2="694" y2="466" class="agree"/><text x="702" y="470" class="label">BC replay agreement</text>
<line x1="850" y1="466" x2="880" y2="466" class="kl"/><text x="888" y="470" class="label">KL to frozen BC reference · right axis</text>
<text x="1132" y="438" text-anchor="end">environment steps</text>
</svg>'''


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    output_data = _resolve(root, args.output_data)
    output_svg = _resolve(root, args.output_svg)
    if output_data.exists() or output_svg.exists():
        raise FileExistsError("Refusing to overwrite training curve artifacts")
    data = build_curve_data(
        root=root,
        metrics_path=_resolve(root, args.metrics),
        screening_root=_resolve(root, args.screening_root),
        confirmation_path=_resolve(root, args.confirmation_report),
        selected_checkpoint=_resolve(root, args.selected_checkpoint),
    )
    output_data.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(data, output_data)
    output_svg.write_text(render_svg(data) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
