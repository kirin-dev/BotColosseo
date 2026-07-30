#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=src
PYTHON="${BOTCOLOSSEO_PYTHON:-python}"
OUTPUT=reports/extraction-v2/aggressive-showcase-search.json

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite $OUTPUT" >&2
  exit 1
fi

"$PYTHON" scripts/evaluate_extraction_policy.py \
  --checkpoint runs/extraction-v2/aggressive/best.pt \
  --base-checkpoint runs/extraction-v2/aggressive-finisher/best.pt \
  --aggressive-governor \
  --governor-carried 35 \
  --governor-health 1 \
  --governor-remaining 1 \
  --style aggressive \
  --cases configs/extraction_v2/aggressive-showcase-search.json \
  --stop-on-aggressive-chain \
  --device cuda:1 \
  --output "$OUTPUT"

"$PYTHON" - "$OUTPUT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if not report["search"]["success"]:
    raise SystemExit("No complete Aggressive showcase chain found")
print(json.dumps(report["search"]["selected_episode"], indent=2, sort_keys=True))
PY
