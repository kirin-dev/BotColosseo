#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-/home/wencong/miniconda3/envs/botcolosseo/bin/python}"
export PYTHONPATH="$ROOT/src"
cd "$ROOT"

"$PYTHON" -m botcolosseo.cli.promote_extraction_calibration
