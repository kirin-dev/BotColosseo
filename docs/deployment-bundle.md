# BotColosseo portable inference bundle

Contains the selected planner, shared visual executor and frozen opponent,
inference source, exact compiled scene, licenses and SHA256 manifest.
Weights are exported with identical tensor values; optimizer state, training
paths, trajectories and private interview notes are excluded.

## Run on Linux, Python 3.10

From this extracted directory, create or activate a Python 3.10 environment:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-runtime.txt
export PYTHONPATH="$PWD/src"
python -m botcolosseo.cli.run_deployment --check-only
python -m botcolosseo.cli.run_deployment --seed 104 --switch-to defensive --output demo.mp4
```

The demo runs headlessly on CPU and saves a video and JSON evidence. FFmpeg is
provided by imageio-ffmpeg; rebuilding the WAD or installing ACC is unnecessary.
For static styles use `--style aggressive`, `--style defensive` or
`--style explorer`. Difficulty is a scalar in [0,1], default Hard=1.
The opponent stays Neutral/Hard. A single switch starts Neutral at decision 0
and requests the chosen style at decision 81, preserving recurrent memory.

All paths are relative to the bundle; no original server or training dataset
is needed. `--bundle /path/to/bundle` is supported from another working directory,
provided that the bundle's src is on PYTHONPATH. Outputs must be new paths.

This artifact restores inference, not training continuation or full experiment
history. Engine multiplayer timing means repeated videos and outcomes need not
match byte-for-byte even with identical model weights and seeds.
Keep the separate full private backup for research data and resumable training.
