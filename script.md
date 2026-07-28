# Crystal Run: Extraction v3 runbook

All commands run from the repository root on branch
`feat/crystal-run-extraction-v3`.

```bash
cd /path/to/BotColosseo
conda activate botcolosseo
export BOTCOLOSSEO_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONPATH="$PWD/src"
```

The scripts use the explicitly selected `BOTCOLOSSEO_PYTHON`; they never fall
back to another user's Conda installation or the ambient shell interpreter.
Select the physical GPU with
`BOTCOLOSSEO_GPU=0` or `BOTCOLOSSEO_GPU=1`; the process sees it as `cuda:0`.

## Verify code and scenario

```bash
"$BOTCOLOSSEO_PYTHON" -m pip check
"$BOTCOLOSSEO_PYTHON" -m ruff check src tests scripts
"$BOTCOLOSSEO_PYTHON" -m pytest tests/unit -q

"$BOTCOLOSSEO_PYTHON" scripts/build_crystal_run_extraction.py \
  --check \
  --acc "$ACC_ROOT/build/acc" \
  --acc-include "$ACC_ROOT"
```

Expected engineering baseline: all Ruff checks pass, all unit tests pass, and
the tracked WAD matches a clean ACC build.

## Long stage 1: Strong

This one resumable script runs:

1. 100,000 train and 20,000 validation Teacher transitions;
2. 10,000 BC updates;
3. 2,000,000 recurrent PPO environment steps.

Demonstrations commit one hashed episode shard and `progress.json` at a time.
Their identity includes the privileged Teacher source hash, so labels from an
older Teacher cannot be resumed or silently reused. BC and PPO resume from
`latest.pt`. Completed stages are skipped only after their input hashes,
checkpoint hashes, calibrated PPO settings, and no-test-access fields pass.

```bash
mkdir -p runs/extraction
nohup env BOTCOLOSSEO_GPU=0 \
  bash scripts/run_extraction_v3_strong.sh \
  > runs/extraction/strong-pipeline.log 2>&1 &
echo $! > runs/extraction/strong-pipeline.pid
```

Check launch:

```bash
pid="$(cat runs/extraction/strong-pipeline.pid)"
ps -p "$pid" -o pid,etime,%cpu,%mem,stat,cmd
tail -n 60 runs/extraction/strong-pipeline.log
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

Safe resume after interruption uses the same command. Do not delete partial
shards or checkpoints. Do not resume after changing the scenario, configs, or
case manifests; provenance checks intentionally reject that.

## Strong candidate selection

This evaluates every historical candidate on the frozen 240-episode scripted
validation protocol, ranks without test access, then evaluates candidates in
rank order on 120 heldout episodes and 40 frozen solo/idle-opponent episodes
until one passes the complete gate. The Strong
gate requires at least 90% solo extraction in addition to the scripted and
heldout capability checks:

When a complete legacy ranking exists from the pre-audit evaluator, it is used
only to order candidates by unchanged task metrics. The candidate that may be
selected is always re-evaluated from scratch with metric schema v2 on
validation, heldout, and solo. This preserves the expensive all-candidate
ranking without allowing legacy style proxies or protocol claims into release
evidence.

```bash
BOTCOLOSSEO_GPU=0 bash scripts/run_extraction_v3_select_strong.sh \
  > runs/extraction/strong-selection.log 2>&1
```

Required output:

```text
runs/extraction/strong-ppo/selected.pt
runs/extraction/strong-ppo/selection.json
```

Exit code 2 means the frozen capability gate failed. Preserve the evidence;
do not relax thresholds.

## Long stage 2: learned styles

Aggressive is first:

```bash
nohup env BOTCOLOSSEO_GPU=0 \
  bash scripts/run_extraction_v3_style.sh aggressive \
  > runs/extraction/aggressive.log 2>&1 &
echo $! > runs/extraction/aggressive.pid
```

After training:

```bash
BOTCOLOSSEO_GPU=0 \
  bash scripts/run_extraction_v3_select_style.sh aggressive \
  > runs/extraction/aggressive-selection.log 2>&1
```

After the Aggressive engineering path is valid, Defensive and Explorer may use
the two A100s concurrently:

```bash
nohup env BOTCOLOSSEO_GPU=0 \
  bash scripts/run_extraction_v3_style.sh defensive \
  > runs/extraction/defensive.log 2>&1 &
echo $! > runs/extraction/defensive.pid

nohup env BOTCOLOSSEO_GPU=1 \
  bash scripts/run_extraction_v3_style.sh explorer \
  > runs/extraction/explorer.log 2>&1 &
echo $! > runs/extraction/explorer.pid
```

Then select using the validation Style Fidelity/Skill Retention Pareto frontier
plus paired heldout evidence. If the first validation candidate fails heldout,
the selector tries the next validation-eligible candidate in frozen rank
order:

```bash
BOTCOLOSSEO_GPU=0 \
  bash scripts/run_extraction_v3_select_style.sh defensive \
  > runs/extraction/defensive-selection.log 2>&1
BOTCOLOSSEO_GPU=1 \
  bash scripts/run_extraction_v3_select_style.sh explorer \
  > runs/extraction/explorer-selection.log 2>&1
```

## Freeze release and run the official test once

The official-test cases do not exist during training or validation. After all
four policies pass selection, generate the 400-case side-balanced manifest
from system entropy, then bind its hash into the immutable release:

```bash
python scripts/seal_extraction_official_test.py

python -m botcolosseo.cli.freeze_extraction_release \
  --strong-selection runs/extraction/strong-ppo/selection.json \
  --aggressive-selection runs/extraction/styles/aggressive/selection.json \
  --defensive-selection runs/extraction/styles/defensive/selection.json \
  --explorer-selection runs/extraction/styles/explorer/selection.json \
  --official-test-manifest \
    runs/extraction/release/official-test-manifest.json
```

The official runner writes a release lock before the first test episode and
persists each completed episode. Re-running after infrastructure failure
resumes the same immutable release. Once `receipt.json` exists, another run is
refused.

```bash
nohup env BOTCOLOSSEO_GPU=0 \
  "$CONDA_PREFIX/bin/python" -u \
  -m botcolosseo.cli.run_extraction_official_test \
  > runs/extraction/release/official-test.log 2>&1 &
echo $! > runs/extraction/release/official-test.pid
```

## Generate validation-only showcase media

```bash
BOTCOLOSSEO_GPU=0 bash scripts/render_extraction_v3_showcase.sh
python scripts/audit_extraction_v3_release.py
```

The selector searches the full validation ledgers for representative cases:

- Strong: successful high-value extraction;
- Aggressive: ordered hit → kill → corpse cache → loot → extraction;
- Defensive: a real low-resource disengagement followed by meaningful extraction;
- Explorer: distinct loot regions, a real backpack upgrade, then extraction.

Generated videos contain viewer-only telemetry. They are not policy inputs and
are never selected from official-test episodes.

## Final verification

```bash
python -m ruff check src tests scripts
python -m pytest tests/unit -q
python scripts/build_crystal_run_extraction.py \
  --check \
  --acc "$ACC_ROOT/build/acc" \
  --acc-include "$ACC_ROOT"
git status --short
```
