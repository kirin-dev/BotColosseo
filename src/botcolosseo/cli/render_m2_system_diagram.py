from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from botcolosseo.demo.m2_system_diagram import render_system_diagram


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the M2 learning-system diagram")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/m2-system.png"),
    )
    args = parser.parse_args(argv)
    print(render_system_diagram(args.output))
    return 0
