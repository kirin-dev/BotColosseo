#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]] || [[ ! "$1" =~ ^(aggressive|defensive|explorer)$ ]]; then
  echo "Usage: $0 {aggressive|defensive|explorer}" >&2
  exit 2
fi

STYLE="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOTCOLOSSEO_PYTHON:-/home/wencong/miniconda3/envs/botcolosseo/bin/python}"
GPU="${BOTCOLOSSEO_GPU:-0}"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

OUTPUT="runs/extraction/styles/$STYLE"
EVAL_ROOT="$OUTPUT/evaluation-v2"
BASE="runs/extraction/strong-ppo/selected.pt"
STRONG_SELECTION="runs/extraction/strong-ppo/selection.json"
mkdir -p "$EVAL_ROOT"

if [[ "$STYLE" != "aggressive" ]]; then
  AGGRESSIVE_SELECTION="runs/extraction/styles/aggressive/selection.json"
  if [[ ! -f "$AGGRESSIVE_SELECTION" ]] || \
    ! "$PYTHON" - "$AGGRESSIVE_SELECTION" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(not (
    selection.get("policy") == "aggressive"
    and selection.get("gate_schema_version") == 2
    and selection.get("eligible") is True
    and selection.get("test_cases_accessed") is False
))
PY
  then
    echo "Defensive/Explorer require a passing Aggressive vertical slice" >&2
    exit 1
  fi
fi

strong_validation="$("$PYTHON" - "$STRONG_SELECTION" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1], encoding="utf-8"))
matches = [
    path for path in selection["evidence"]
    if path.endswith("-validation.json")
]
if len(matches) != 1:
    raise SystemExit("Strong selection has no unique validation evidence")
print(matches[0])
PY
)"
strong_heldout="$("$PYTHON" - "$STRONG_SELECTION" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1], encoding="utf-8"))
matches = [
    path for path in selection["evidence"]
    if path.endswith("-heldout.json")
]
if len(matches) != 1:
    raise SystemExit("Strong selection has no unique heldout evidence")
print(matches[0])
PY
)"
reports=()
for checkpoint in "$OUTPUT"/candidate-*.pt; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "No $STYLE candidate checkpoints found" >&2
    exit 1
  fi
  tag="$(basename "${checkpoint%.pt}")"
  report="$EVAL_ROOT/$tag-validation.json"
  if [[ ! -f "$report" ]]; then
    "$PYTHON" -u -m botcolosseo.cli.evaluate_extraction_v3 \
      --checkpoint "$checkpoint" \
      --base-checkpoint "$BASE" \
      --policy "$STYLE" \
      --split validation \
      --device cuda:0 \
      --output "$report"
  fi
  reports+=(--report "$report")
done

ranking="$EVAL_ROOT/ranking.json"
ranking_tmp="$EVAL_ROOT/.ranking.json.tmp"
if [[ -e "$ranking_tmp" ]]; then
  echo "Refusing to overwrite stale temporary ranking: $ranking_tmp" >&2
  exit 1
fi
"$PYTHON" -u -m botcolosseo.cli.rank_extraction_candidates \
  --policy "$STYLE" \
  "${reports[@]}" \
  --strong-validation-report "$strong_validation" \
  --output "$ranking_tmp"
mv "$ranking_tmp" "$ranking"

if [[ -f "$OUTPUT/selection.json" ]]; then
  if "$PYTHON" - "$OUTPUT/selection.json" "$STYLE" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(not (
    selection.get("gate_schema_version") == 2
    and selection.get("policy") == sys.argv[2]
    and selection.get("eligible") is True
    and selection.get("test_cases_accessed") is False
))
PY
  then
    echo "SKIP complete objective-aligned $STYLE selection"
    exit 0
  fi
  archive="$OUTPUT/archive/pre-objective-alignment"
  mkdir -p "$archive"
  for artifact in selection.json selected.pt; do
    if [[ -e "$OUTPUT/$artifact" ]]; then
      if [[ -e "$archive/$artifact" ]]; then
        echo "Refusing to overwrite archived legacy $STYLE $artifact" >&2
        exit 1
      fi
      mv "$OUTPUT/$artifact" "$archive/$artifact"
    fi
  done
fi

selected=false
while IFS=$'\t' read -r selected_checkpoint selected_report; do
  selected_tag="$(basename "${selected_checkpoint%.pt}")"
  heldout_report="$EVAL_ROOT/$selected_tag-heldout.json"
  full_gate_report="$EVAL_ROOT/$selected_tag-full-gate.json"
  if [[ ! -f "$heldout_report" ]]; then
    "$PYTHON" -u -m botcolosseo.cli.evaluate_extraction_v3 \
      --checkpoint "$selected_checkpoint" \
      --base-checkpoint "$BASE" \
      --policy "$STYLE" \
      --split heldout \
      --device cuda:0 \
      --output "$heldout_report"
  fi
  if "$PYTHON" -u -m botcolosseo.cli.select_extraction_candidate \
    --policy "$STYLE" \
    --checkpoint "$selected_checkpoint" \
    --validation-report "$selected_report" \
    --strong-validation-report "$strong_validation" \
    --heldout-report "$heldout_report" \
    --strong-heldout-report "$strong_heldout" \
    --output-checkpoint "$OUTPUT/selected.pt" \
    --output-report "$full_gate_report"
  then
    mv "$full_gate_report" "$OUTPUT/selection.json"
    selected=true
    break
  fi
done < <("$PYTHON" - "$ranking" <<'PY'
import json
import sys

ranking = json.load(open(sys.argv[1], encoding="utf-8"))
for candidate in sorted(
    ranking["candidates"],
    key=lambda item: tuple(item["score"]),
    reverse=True,
):
    if candidate["eligible"]:
        print(candidate["checkpoint"], candidate["report"], sep="\t")
PY
)
if [[ "$selected" != true ]]; then
  echo "No $STYLE candidate passed validation and heldout gates" >&2
  exit 2
fi

echo "Crystal Run: Extraction $STYLE selection complete"
