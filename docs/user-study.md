# Anonymous style-recognition study

This is a small product-perception study, not a population-level behavioral
experiment. Aggressive is a learned style policy; Defensive and Explorer are
honestly disclosed in public documentation as hybrid governors. Study clips
remain label-blind so the study measures visible behavior rather than wording.
Run it only after all three public videos pass their applicable product gates.

## Prepare the blind package

```bash
PYTHONPATH=src python scripts/prepare_user_study.py \
  --aggressive artifacts/m6-curated-clips-v2/aggressive-1.mp4 \
               artifacts/m6-curated-clips-v2/aggressive-2.mp4 \
  --defensive artifacts/m6-curated-clips-v2/defensive-1.mp4 \
              artifacts/m6-curated-clips-v2/defensive-2.mp4 \
  --explorer artifacts/m6-curated-clips-v2/explorer-1.mp4 \
             artifacts/m6-curated-clips-v2/explorer-2.mp4 \
  --output-dir artifacts/m6-user-study-v2 \
  --assignments 10
```

Keep `answer-key.json` and `manifest.json` private until collection closes.
Give each respondent `participant-instructions.md`, one row group from
`assignments.csv`, the corresponding six opaque files under `clips/`, and the
six response fields from `response-template.csv`. On a phone, send the six
videos in that assignment's numbered order and have the respondent return one
row per video. Do not send the answer key or style-labeled curated files.

Ask the respondent to:

1. choose `aggressive`, `defensive`, `explorer`, or `unsure` for every clip;
2. rate style clarity from 1 (unclear) to 5 (very clear);
3. rate perceived difficulty from 1 (easy) to 5 (hard).

Use a short anonymous identifier such as `r01`. Do not collect names, email
addresses, IP addresses, free text, or other personal information.

## Analyze anonymous responses

```bash
PYTHONPATH=src python scripts/analyze_user_study.py \
  --package-dir artifacts/m6-user-study-v2 \
  --responses reports/m6/user-study/responses.csv \
  --output reports/m6/user-study/summary.json \
  --chart docs/assets/showcase/m6-user-study.png
```

The analyzer fails closed on incomplete assignments, duplicate responses,
unknown choices, invalid ratings, or changed clips. It reports the raw sample
size, confusion matrix, per-style recognition, macro/micro recognition,
Wilson 95% intervals, clarity, perceived difficulty, and source hashes.
It also renders a recognition-rate/confusion-matrix card for the README.

Commit only genuinely anonymous responses. In README wording, call a 5--10
person run a “small anonymous product-perception study” and show its exact
sample size.
