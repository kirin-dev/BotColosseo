#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/wencong/miniconda3/envs/botcolosseo/bin/python
CONFIG=configs/extraction/randomized/demonstrations.yaml

"$PYTHON" -m botcolosseo.cli.generate_extraction_demonstrations --config "$CONFIG" --split train --style strong
"$PYTHON" -m botcolosseo.cli.generate_extraction_demonstrations --config "$CONFIG" --split validation --style strong
"$PYTHON" -m botcolosseo.cli.train_extraction_bc --config configs/extraction/randomized/bc.yaml --style strong --device cuda:0
