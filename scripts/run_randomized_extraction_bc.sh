#!/usr/bin/env bash
set -euo pipefail

PYTHON="${BOTCOLOSSEO_PYTHON:-python}"
CONFIG=configs/extraction/randomized/aligned-v2/demonstrations.yaml

"$PYTHON" -m botcolosseo.cli.generate_extraction_demonstrations --config "$CONFIG" --split train --style strong
"$PYTHON" -m botcolosseo.cli.generate_extraction_demonstrations --config "$CONFIG" --split validation --style strong
"$PYTHON" -m botcolosseo.cli.train_extraction_bc \
  --config configs/extraction/randomized/aligned-v2/bc.yaml \
  --style strong \
  --device cuda:0
