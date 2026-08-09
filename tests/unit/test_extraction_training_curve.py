from __future__ import annotations

import json
import re
from pathlib import Path

from botcolosseo.cli.build_extraction_training_curve import (
    build_curve_data,
    render_svg,
)
from botcolosseo.data.demonstrations import sha256_file


def test_training_curve_binds_screening_confirmation_and_diagnostics(
    tmp_path: Path,
) -> None:
    run = tmp_path / "runs/strong"
    screens = tmp_path / "reports/screens"
    run.mkdir(parents=True)
    screens.mkdir(parents=True)
    selected = run / "candidate-0950000.pt"
    selected.write_bytes(b"selected")
    selected_sha256 = sha256_file(selected)
    for index, step in enumerate(range(50_000, 1_000_001, 50_000)):
        checkpoint = run / f"candidate-{step:07d}.pt"
        if step != 950_000:
            checkpoint.write_bytes(str(step).encode())
        payload = {
            "checkpoint": str(checkpoint.relative_to(tmp_path)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "metrics": {
                "episodes": [
                    {"extracted": item < 24 + index % 4, "won": item < 16}
                    for item in range(32)
                ]
            },
        }
        (screens / f"candidate-{step:07d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    metrics = run / "metrics.jsonl"
    metrics.write_text(
        "".join(
            json.dumps(
                {
                    "kind": "train",
                    "environment_steps": step,
                    "replay_agreement": 0.95,
                    "reference_kl": 0.01,
                }
            )
            + "\n"
            for step in range(50_000, 1_000_001, 50_000)
        ),
        encoding="utf-8",
    )
    confirmation = run / "confirmation.json"
    confirmation.write_text(
        json.dumps(
            {
                "checkpoint_sha256": selected_sha256,
                "split": "validation",
                "test_cases_accessed": False,
                "metrics": {
                    "episodes": [
                        {"extracted": item < 200, "won": item < 136}
                        for item in range(240)
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    data = build_curve_data(
        root=tmp_path,
        metrics_path=metrics,
        screening_root=screens,
        confirmation_path=confirmation,
        selected_checkpoint=selected,
    )

    assert len(data["screening"]) == 20
    assert len(data["training_diagnostics"]) == 20
    assert data["confirmation"]["episodes"] == 240
    svg = render_svg(data)
    assert "950k · 240-episode confirmation" in svg
    assert 'width="1200" height="500"' in svg
    assert 'd="M66 75V390H534 M664 75V390H1132"' in svg
    assert re.search(r'<text x="18"[^>]*>0%</text>', svg) is None
    assert '<line x1="66" y1="466" x2="96" y2="466" class="extract"/>' in svg
    assert '<line x1="190" y1="466" x2="220" y2="466" class="win"/>' in svg
    assert '<line x1="664" y1="466" x2="694" y2="466" class="agree"/>' in svg
    assert '<line x1="850" y1="466" x2="880" y2="466" class="kl"/>' in svg
    assert ".win{fill:none;stroke:#6aa8dc;stroke-width:2.5;stroke-dasharray:7 5}" in svg
    assert ".kl{fill:none;stroke:#bd7c16;stroke-width:2.5;stroke-dasharray:7 5}" in svg
