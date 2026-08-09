# BotColosseo v3 runbook

All commands run from the repository root.

```bash
cd /path/to/BotColosseo
conda activate botcolosseo
export BOTCOLOSSEO_PYTHON="$CONDA_PREFIX/bin/python"
export PYTHONPATH="$PWD/src"
```

The scripts use `BOTCOLOSSEO_PYTHON` when set and otherwise use `python` from
the active environment. They contain no machine-specific Conda paths. Select
the physical GPU with
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
3. a first 600,000-step recurrent PPO stage.

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

Run Strong selection after the 600k stage. If no candidate passes, or the
validation trend still has clear headroom, resume the same optimizer and PFSP
state to 800k, then at most 1M before reassessing:

```bash
BOTCOLOSSEO_GPU=0 BOTCOLOSSEO_STOP_AFTER_STEPS=800000 \
  bash scripts/run_extraction_v3_strong.sh

BOTCOLOSSEO_GPU=0 BOTCOLOSSEO_STOP_AFTER_STEPS=1000000 \
  bash scripts/run_extraction_v3_strong.sh
```

Do not run to the 2M configuration ceiling by default. The older frozen run
peaked near 600k; later stages are justified only by new validation evidence.

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

## Current randomized style repair: opportunity-conditioned PBRS

### Aligned completion-semantics v2

Defensive completion now requires a real disengagement before extraction;
Explorer completion requires both distinct-region exploration and a real
backpack upgrade before extraction. Retrain only these two adapters to 51.2k
steps under the original 100k learning-rate schedule. The two GPU lanes are
independent and preserve all v1 artifacts:

```bash
mkdir -p runs/extraction-randomized/styles-opportunity-pbrs-aligned-v2/control
nohup env \
  BOTCOLOSSEO_PYTHON="$CONDA_PREFIX/bin/python" \
  bash scripts/run_extraction_randomized_aligned_opportunity_styles.sh \
  > runs/extraction-randomized/styles-opportunity-pbrs-aligned-v2/control/pipeline.log \
  2>&1 &
echo $! \
  > runs/extraction-randomized/styles-opportunity-pbrs-aligned-v2/control/pipeline.pid
```

Inspect without restarting anything:

```bash
pid="$(cat runs/extraction-randomized/styles-opportunity-pbrs-aligned-v2/control/pipeline.pid)"
ps -p "$pid" -o pid,etime,%cpu,%mem,stat,cmd
tail -n 12 runs/extraction-randomized/styles-opportunity-pbrs-aligned-v2/control/*.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader
```

Screen both 51.2k candidates on 32 paired randomized cases before any 240-case
validation. Do not extend or tune coefficients unless that screen confirms the
intended directional shift.

### Defensive disengagement-action v3

If the v2 Defensive 32-case screen is direction-negative, use the v3
disengagement labeler. It steers away from the opponent until the measured
640-unit safe distance is reached, then converts that safety into extraction:

```bash
mkdir -p runs/extraction-randomized/styles-opportunity-pbrs-disengagement-v3/control
BOTCOLOSSEO_PYTHON="$CONDA_PREFIX/bin/python" \
BOTCOLOSSEO_GPU=0 \
  bash scripts/run_extraction_randomized_disengagement_style.sh \
  > runs/extraction-randomized/styles-opportunity-pbrs-disengagement-v3/control/defensive.log \
  2>&1
```

Run the same 32-case paired screen before full validation. This v3 is a
Defensive-only repair; keep the accepted Aggressive and Explorer checkpoints
frozen.

If the v3 full validation remains behaviorally indistinguishable from Strong,
run one bounded preference-strength calibration. It changes only
`eta_preference` from 0.08 to 0.20 and the probability-lift margin from 0.15 to
0.30:

```bash
mkdir -p runs/extraction-randomized/styles-opportunity-pbrs-defensive-v4/control
CUDA_VISIBLE_DEVICES=0 "$CONDA_PREFIX/bin/python" -u \
  -m botcolosseo.cli.train_extraction_style \
  --config configs/extraction/randomized/styles-opportunity-pbrs-defensive-v4.yaml \
  --style defensive \
  --device cuda:0 \
  --output-dir runs/extraction-randomized/styles-opportunity-pbrs-defensive-v4/defensive \
  --stop-after-steps 51200 \
  > runs/extraction-randomized/styles-opportunity-pbrs-defensive-v4/control/defensive.log \
  2>&1
```

Do not increase residual capacity or relax the outside-opportunity KL in this
calibration.

The frozen Strong Base is
`runs/extraction-randomized/strong-ppo-conservative-v2/candidate-0950000.pt`
(SHA-256 `d05fa0cbcdf9d0aa3df363e66f6aa8957a35c76d9d3fb936b15d6fe5b5f22484`).
Train all three fresh adapters for an initial 100k-step budget. GPU 0 runs
Aggressive then Explorer; GPU 1 runs Defensive. The pipeline is resumable and
does not read validation, heldout, or official-test cases:

```bash
mkdir -p runs/extraction-randomized/styles-opportunity-pbrs-v1/control
nohup env \
  BOTCOLOSSEO_PYTHON="$CONDA_PREFIX/bin/python" \
  BOTCOLOSSEO_STYLE_STEPS=100000 \
  bash scripts/run_extraction_randomized_opportunity_styles.sh \
  > runs/extraction-randomized/styles-opportunity-pbrs-v1/control/pipeline.log \
  2>&1 &
echo $! \
  > runs/extraction-randomized/styles-opportunity-pbrs-v1/control/pipeline.pid
```

Inspect without restarting anything:

```bash
pid="$(cat runs/extraction-randomized/styles-opportunity-pbrs-v1/control/pipeline.pid)"
ps -p "$pid" -o pid,etime,%cpu,%mem,stat,cmd
tail -n 12 runs/extraction-randomized/styles-opportunity-pbrs-v1/control/*.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader
```

Do not extend past 100k before paired validation. The 50k and 100k candidates
are both retained so validation can select the actual peak rather than the
last checkpoint.

## Legacy fixed-layout stage 2: learned styles

Aggressive is first. Run styles in resumable 200k-step stages and evaluate the
newest candidate before deciding whether to extend to 400k or 600k. The runner
defaults to 200k if `BOTCOLOSSEO_STOP_AFTER_STEPS` is omitted:

```bash
nohup env BOTCOLOSSEO_GPU=0 BOTCOLOSSEO_STOP_AFTER_STEPS=200000 \
  bash scripts/run_extraction_v3_style.sh aggressive \
  > runs/extraction/aggressive.log 2>&1 &
echo $! > runs/extraction/aggressive.pid
```

To continue the same run without changing its config hash:

```bash
BOTCOLOSSEO_GPU=0 BOTCOLOSSEO_STOP_AFTER_STEPS=400000 \
  bash scripts/run_extraction_v3_style.sh aggressive
```

After training:

```bash
BOTCOLOSSEO_GPU=0 \
  bash scripts/run_extraction_v3_select_style.sh aggressive \
  > runs/extraction/aggressive-selection.log 2>&1
```

If the 600k candidate passes every Aggressive check except the paired style
CI lower bound, run the approved weights-only calibration. It preserves the
600k lineage, resets optimizer state, uses a 1.5x bounded Aggressive ledger,
and writes to an isolated directory:

```bash
nohup env BOTCOLOSSEO_GPU=0 BOTCOLOSSEO_STOP_AFTER_STEPS=100000 \
  bash scripts/run_extraction_v3_aggressive_calibration.sh \
  > runs/extraction/aggressive-calibration.log 2>&1 &
echo $! > runs/extraction/aggressive-calibration.pid
```

Evaluate the calibrated candidates against the unchanged validation and
heldout gates, then promote only a fully eligible selection:

```bash
BOTCOLOSSEO_GPU=0 \
  BOTCOLOSSEO_STYLE_OUTPUT=runs/extraction/styles/aggressive-calibration-v2 \
  bash scripts/run_extraction_v3_select_style.sh aggressive \
  > runs/extraction/aggressive-calibration-selection.log 2>&1

bash scripts/promote_extraction_v3_aggressive_calibration.sh
```

If 100k fails validation, resume once with
`BOTCOLOSSEO_STOP_AFTER_STEPS=200000`. Failure at 200k ends this calibration;
do not increase reward scale or relax the gate.

If calibration ends without a research-eligible selection, generate the
approved product-only directional admission for the original 600k candidate.
This evaluates its paired heldout capability if needed, copies the exact
candidate to `showcase.pt`, and records both research failures:
`style_ci_lower` on validation and the original heldout gate's absolute
worst-opponent floor. The separate product gate uses relative retention within
20 percentage points and records that it was adopted after heldout review:

```bash
BOTCOLOSSEO_GPU=0 \
  bash scripts/run_extraction_v3_admit_aggressive_showcase.sh \
  > runs/extraction/aggressive-showcase-admission.log 2>&1
```

The admission can unblock style development and validation-only media. It
cannot be used by research selection, release freezing, or official-test
commands.

After Aggressive has either a strict selection or this directional admission,
Defensive and Explorer may use two GPUs concurrently when available. Start with
a 200k stage and resume to 400k/600k only if the frozen validation evidence
warrants it:

```bash
nohup env BOTCOLOSSEO_GPU=0 BOTCOLOSSEO_STOP_AFTER_STEPS=200000 \
  bash scripts/run_extraction_v3_style.sh defensive \
  > runs/extraction/defensive.log 2>&1 &
echo $! > runs/extraction/defensive.pid

nohup env BOTCOLOSSEO_GPU=1 BOTCOLOSSEO_STOP_AFTER_STEPS=200000 \
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

If Explorer reaches the approved positive directional validation gate but
misses only `style_ci_lower`, evaluate heldout and create its product-only
admission using the rule frozen before heldout access:

```bash
BOTCOLOSSEO_GPU=1 \
  bash scripts/run_extraction_v3_admit_style_showcase.sh explorer \
  > runs/extraction/explorer-showcase-admission.log 2>&1
```

The original Defensive 200k/400k route is retained as failed development
evidence. Run the single approved targeted calibration from a fresh adapter:

```bash
BOTCOLOSSEO_GPU=0 BOTCOLOSSEO_STOP_AFTER_STEPS=200000 \
  bash scripts/run_extraction_v3_defensive_calibration.sh \
  > runs/extraction/defensive-calibration-v2.log 2>&1

BOTCOLOSSEO_GPU=0 \
  BOTCOLOSSEO_STYLE_OUTPUT=runs/extraction/styles/defensive-calibration-v2 \
  bash scripts/run_extraction_v3_select_style.sh defensive \
  > runs/extraction/defensive-calibration-v2-selection.log 2>&1
```

Resume to 400k only when the frozen validation mean is positive and every
capability, anti-hacking, and protocol check passes but directional Showcase
evidence is incomplete. Once a calibrated candidate meets the directional
validation rule, bind heldout evidence and copy it into the canonical
Defensive Showcase location:

```bash
BOTCOLOSSEO_GPU=0 \
  BOTCOLOSSEO_STYLE_SOURCE=\
runs/extraction/styles/defensive-calibration-v2 \
  bash scripts/run_extraction_v3_admit_style_showcase.sh defensive \
  > runs/extraction/defensive-showcase-admission.log 2>&1
```

When validation direction, capability, anti-hacking, and protocol checks pass
but the stricter directional or heldout product rule does not, the final
product Showcase may use the explicitly lower
`validation_demonstration` tier:

```bash
BOTCOLOSSEO_GPU=1 \
  BOTCOLOSSEO_STYLE_SOURCE=runs/extraction/styles/explorer \
  bash scripts/run_extraction_v3_create_style_demonstration.sh explorer

BOTCOLOSSEO_GPU=0 \
  BOTCOLOSSEO_STYLE_SOURCE=\
runs/extraction/styles/defensive-calibration-v2 \
  bash scripts/run_extraction_v3_create_style_demonstration.sh defensive
```

This tier binds the exact Strong lineage, complete paired validation, complete
heldout disclosure, and the learned residual checkpoint. It is eligible only
for validation media: `research_gate_passed=false` and
`official_test_eligible=false` are written into the immutable manifest.

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
```

The selector searches the full validation ledgers for representative cases:

- Strong: successful high-value extraction;
- Aggressive: ordered hit → kill → corpse cache → loot → extraction;
- Defensive: a real low-resource disengagement followed by meaningful extraction;
- Explorer: distinct loot regions, a real backpack upgrade, then extraction.

Generated videos contain viewer-only telemetry. They are not policy inputs and
are never selected from official-test episodes. Rendering retries the same
preselected validation case at most five times and writes media only when the
actual replay completes its advertised causal chain. The script finishes with
a separate product Showcase audit; it does not run or claim the strict
research release audit.

## Run the matched 200k style ablations

The frozen ablation compares the existing Full method against Reward+KL and
Reward-only variants for all three styles. It reuses the Full 200k reports and
launches six new train-then-validation jobs over two GPU lanes:

```bash
mkdir -p runs/extraction/ablations/control

BOTCOLOSSEO_ABLATION_PREFLIGHT_ONLY=1 \
BOTCOLOSSEO_PYTHON="$CONDA_PREFIX/bin/python" \
  scripts/run_extraction_v3_ablations.sh

nohup bash -c '
  BOTCOLOSSEO_PYTHON="'"$CONDA_PREFIX"'/bin/python" \
    scripts/run_extraction_v3_ablations.sh
  code=$?
  printf "%s\n" "$code" \
    > runs/extraction/ablations/control/pipeline.exit
  exit "$code"
' > runs/extraction/ablations/control/pipeline.log 2>&1 &
echo $! > runs/extraction/ablations/control/pipeline.pid
```

The expected wall time is about 4.5 hours. Apply the 50%-of-estimate monitoring
rule: first inspect progress about 2 hours 15 minutes after launch, then use
half of the revised remaining estimate for any later check.

```bash
pid="$(cat runs/extraction/ablations/control/pipeline.pid)"
ps -p "$pid" -o pid,etime,%cpu,%mem,stat,cmd
tail -n 40 runs/extraction/ablations/control/pipeline.log
tail -n 8 runs/extraction/ablations/control/*.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader
```

After all six validation reports finish:

```bash
python -m botcolosseo.cli.summarize_extraction_ablations \
  --output reports/extraction/style-ablation.json \
  --markdown-output reports/extraction/style-ablation.md
```

The ablation uses validation only and never opens heldout or official-test
cases.

## Final verification

```bash
python -m ruff check src tests scripts
python -m pytest tests/unit -q
python scripts/build_crystal_run_extraction.py \
  --check \
  --acc "$ACC_ROOT/build/acc" \
  --acc-include "$ACC_ROOT"

audit_dir="$(mktemp -d)"
python -m botcolosseo.cli.audit_extraction_showcase \
  --selection reports/extraction/showcase/selection.json \
  --board-manifest reports/extraction/showcase/manifest.json \
  --method docs/assets/extraction/method.svg \
  --output "$audit_dir/audit.json"

git status --short
```
