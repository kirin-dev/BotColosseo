# Bot Colosseo dependency commands

The `botcolosseo` Conda environment already contains the M0 runtime and
training dependencies. Do not recreate the environment or reinstall PyTorch.

## Verified M0 environment status

Verified on 2026-07-20:

- `python -m pip check`: `No broken requirements found.`
- `python -m ruff --version`: `ruff 0.15.22`

No additional M0 Python packages are pending. The commands below are retained
for reproducing the verification after future dependency changes:

```bash
conda activate botcolosseo
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY python -m pip check
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY python -m ruff --version
```

## Verified: later-stage Python packages

Verified on 2026-07-20: `psutil`, `pytest-timeout`, `pytest-cov`, `scipy`, and
`seaborn` are installed. They support multiplayer process management,
experiment analysis, visualization, and test hardening. The installation
commands are retained for environment reproduction:

```bash
conda activate botcolosseo
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY python -m pip install \
  'psutil>=5.9,<8.0' \
  'pytest-timeout>=2.3,<3.0' \
  'pytest-cov>=5.0,<8.0' \
  'scipy>=1.11,<2.0' \
  'seaborn>=0.13,<1.0'
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY python -m pip check
```

Verify:

```bash
python -c "import psutil, scipy, seaborn; print(psutil.__version__, scipy.__version__, seaborn.__version__)"
python -m pytest --help | grep -E 'timeout|cov'
```

## Verified: ACC 1.60 ACS compiler

ACC compiles the ACS scenario scripts used by the UDMF Crystal Run map. The
binary was built and verified on 2026-07-20 at
`/home/wencong/.local/bin/acc`; it reports version 1.60. The machine already
has Git, CMake, Make, GCC, and G++. The build commands are retained for
reproduction:

```bash
mkdir -p /home/wencong/.local/src /home/wencong/.local/bin
git clone --branch 1.60 --depth 1 \
  https://github.com/ZDoom/acc.git \
  /home/wencong/.local/src/acc-1.60
cmake \
  -S /home/wencong/.local/src/acc-1.60 \
  -B /home/wencong/.local/src/acc-1.60/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /home/wencong/.local/src/acc-1.60/build --parallel 4
install -m 0755 \
  /home/wencong/.local/src/acc-1.60/build/acc \
  /home/wencong/.local/bin/acc
/home/wencong/.local/bin/acc
```

Expected: the final command prints the ACC usage text. Add
`/home/wencong/.local/bin` to the interactive shell PATH if desired; project
build scripts should accept an explicit `ACC_PATH` and must not assume a global
installation.

## Verified: SLADE 3.2.12 map editor

SLADE is useful for visually editing and inspecting UDMF maps. Ubuntu 20.04
uses the official DRD Team package repository for older Debian/Ubuntu releases.
Version 3.2.12 is installed. The installation commands are retained for
reproduction:

```bash
sudo install -d -m 0755 /etc/apt/keyrings
sudo wget -4\
  https://debian.drdteam.org/drdteam.gpg \
  -O /etc/apt/keyrings/drdteam.gpg
echo 'deb [signed-by=/etc/apt/keyrings/drdteam.gpg] https://debian.drdteam.org/ stable multiverse' \
  | sudo tee /etc/apt/sources.list.d/drdteam.list
sudo apt-get update
sudo apt-get install slade
dpkg-query -W -f='${Version}\n' slade
```

Expected: `3.2.12`.

SLADE is a GUI application, and even `slade --version` initializes GTK. In a
headless SSH shell with no `DISPLAY`, that command fails with `Unable to
initialize GTK+`; this does not mean the package installation failed. Use
`dpkg-query` for headless version checks. Interactive map authoring requires
X11 forwarding, a remote desktop, or local editing. ACC is sufficient for
command-line ACS builds, and SLADE is not required for M0.

## Already installed: do not reinstall

- Python 3.10.20, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124
- ViZDoom 1.3.0 and bundled `freedoom2.wad`/`vizdoom.pk3`
- Gymnasium, NumPy, Pandas, PyYAML, Matplotlib, OpenCV, TensorBoard, tqdm
- ImageIO, imageio-ffmpeg, system FFmpeg/ffprobe
- Ruff 0.15.22 and pytest 9.1.1
- CMake, Make, GCC/G++, Git, Git LFS, tmux, screen, Xvfb, zip/unzip, jq, rsync

## Deliberately excluded until a concrete requirement appears

- Weights & Biases: TensorBoard is sufficient for the approved experiment plan.
- ONNX and ONNX Runtime: model export is not an approved milestone requirement.
- Hugging Face Hub: checkpoints can use GitHub Releases or another later-chosen
  artifact host.
- `omgifol`: the M1 plan will define the WAD packaging boundary; do not add a
  second map representation before that decision.
- GZDoom and original Doom assets: ViZDoom plus Freedoom is the approved runtime
  and licensing path.

## Verified: M2 full demonstration generation

Completed and passed the full artifact gate on 2026-07-20. The reproducibility
code, 200-transition real smoke, and full 100,000-train /
20,000-validation generation all passed. The commands are retained for
reproduction:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
mkdir -p runs/m2
nohup env \
  PYTHONPATH=/home/wencong/BotColosseo/.worktrees/m2-base-training/src \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/generate_demonstrations.py \
  --split all \
  --output-dir data/generated/m2 \
  --report reports/m2/demonstrations-manifest.json \
  --plot docs/assets/m2-demonstration-distribution.png \
  > runs/m2/demonstrations.log 2>&1 &
echo $! > runs/m2/demonstrations.pid
cat runs/m2/demonstrations.pid
```

Monitor without modifying the run:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
tail -n 40 runs/m2/demonstrations.log
ps -p "$(cat runs/m2/demonstrations.pid)" -o pid,etime,%cpu,%mem,stat,cmd
```

When the PID has exited, the log must contain the final JSON and no
`Traceback`. Run this exact artifact gate:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
env PYTHONPATH="$PWD/src" \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python - <<'PY'
import json
from pathlib import Path

from botcolosseo.data.demonstrations import (
    load_demonstration_shard,
    sha256_file,
)
from botcolosseo.data.schema import find_privileged_keys

report = json.loads(Path("reports/m2/demonstrations-manifest.json").read_text())
expected = {"train": 100_000, "validation": 20_000}
assert report["test_cases_accessed"] is False
assert not find_privileged_keys(report)
assert {item["split"] for item in report["splits"]} == set(expected)
for split in report["splits"]:
    name = split["split"]
    assert split["transitions"] == split["requested_transitions"] == expected[name]
    assert split["privileged_leak_count"] == 0
    assert split["test_cases_accessed"] is False
    assert sum(split["opponent_counts"].values()) == expected[name]
    assert len(set(split["opponent_counts"].values())) == 1
    shard_total = 0
    for item in split["shards"]:
        path = Path("data/generated/m2") / name / item["file"]
        assert sha256_file(path) == item["sha256"]
        shard_total += load_demonstration_shard(path)["frame"].shape[0]
    assert shard_total == expected[name]
assert Path("docs/assets/m2-demonstration-distribution.png").stat().st_size > 10_000
print("M2 full demonstration gate: PASS")
PY
! grep -n "Traceback" runs/m2/demonstrations.log
ps -eo pid,ppid,stat,cmd | grep -E '[b]otcolosseo-duel|[v]izdoom' || true
git status --short
```

Expected tracked changes after success are only the refreshed
`reports/m2/demonstrations-manifest.json` and
`docs/assets/m2-demonstration-distribution.png`. The full NPZ shards are under
ignored `data/generated/` and must not be committed. Return the final log tail,
artifact-gate output, and `git status --short` before the Task 6 gate is marked
passed.

## Action required: M2 full behavioral cloning

The real-shard overfit test and the interrupted 1,000-update single-A100 smoke
have passed. The smoke resumed at update 500, reached 89.0% validation action
accuracy at update 1,000, and completed the held-out validation objective. The
full run uses all 100,000 training and 20,000 validation transitions and is the
next required gate. Start it in the M2 worktree while the machine can be left
unattended:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
mkdir -p runs/m2/bc-full
nohup env \
  PYTHONPATH=/home/wencong/BotColosseo/.worktrees/m2-base-training/src \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python -u \
  scripts/train_bc.py \
  --device cuda:0 \
  --updates 10000 \
  --output-dir runs/m2/bc-full \
  > runs/m2/bc-full.log 2>&1 &
echo $! > runs/m2/bc-full.pid
cat runs/m2/bc-full.pid
```

Monitor without modifying the run:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
tail -n 40 runs/m2/bc-full.log
ps -p "$(cat runs/m2/bc-full.pid)" -o pid,etime,%cpu,%mem,stat,cmd
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

If the process was interrupted and `latest.pt` exists, resume the same
10,000-update scheduler instead of deleting the directory or starting over:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
nohup env \
  PYTHONPATH=/home/wencong/BotColosseo/.worktrees/m2-base-training/src \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python -u \
  scripts/train_bc.py \
  --device cuda:0 \
  --updates 10000 \
  --output-dir runs/m2/bc-full \
  --resume runs/m2/bc-full/latest.pt \
  >> runs/m2/bc-full.log 2>&1 &
echo $! > runs/m2/bc-full.pid
cat runs/m2/bc-full.pid
```

Do not resume after a config, scenario, or demonstration manifest change; the
trainer intentionally rejects provenance drift. When the PID has exited, run
this exact artifact gate:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
env PYTHONPATH="$PWD/src" \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python - <<'PY'
import hashlib
import json
import math
from pathlib import Path

import torch

root = Path("runs/m2/bc-full")
summary = json.loads((root / "summary.json").read_text())
closed_loop = json.loads((root / "closed-loop-validation.json").read_text())
records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
validations = [record for record in records if record["kind"] == "validation"]
checkpoint_path = root / "best.pt"
checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

assert summary["updates"] == 10_000
assert summary["train_transitions_loaded"] == 100_000
assert summary["validation_transitions_loaded"] == 20_000
assert summary["pure_behavioral_cloning"] is True
assert summary["checkpoint_sha256"] == checkpoint_sha
assert summary["closed_loop"] == closed_loop
assert [record["update"] for record in validations] == list(range(250, 10_001, 250))
assert all(math.isfinite(record["loss"]) for record in validations)
assert all(0.0 <= record["accuracy"] <= 1.0 for record in validations)
assert all(record["objective_rate"] in (0.0, 1.0) for record in validations)
best = min(validations, key=lambda record: (-record["objective_rate"], record["loss"]))
assert summary["best_update"] == best["update"]
assert summary["best_objective_rate"] == best["objective_rate"]
assert summary["best_validation_loss"] == best["loss"]
metadata = checkpoint["metadata"]
assert metadata["counters"]["updates"] == summary["best_update"]
assert metadata["config_hash"] == summary["config_hash"]
assert metadata["scenario_hash"] == summary["scenario_hash"]
print("M2 full BC artifact gate: PASS")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
! grep -n "Traceback" runs/m2/bc-full.log
ps -eo pid,ppid,stat,cmd | grep -E '[b]otcolosseo-duel|[v]izdoom|[t]rain_bc.py' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
git status --short
```

Return the final `tail -n 80 runs/m2/bc-full.log`, the artifact-gate output,
the process/GPU checks, and `git status --short`. Do not start PPO until this
pure-BC checkpoint gate has been reviewed and passed.

## Completed: M2 full PPO training and validation-only selection

The 20,000-step A100 pilot, 2,000-step stop/resume gate, candidate snapshot
gate, and validation-only selector passed. The full run used 1,000,000
environment steps on GPU 0. It saves one atomic candidate per 100,000-step
bucket, then evaluates every candidate on three frozen validation seed-pairs
per opponent with both learner sides. Selection is lexicographic by objective
rate, win rate, mean score difference, then earlier environment step. No test
case is read by training or selection.

The full run was completed on 2026-07-21. It hit one transient ViZDoom respawn
timeout at 898,304 environment steps, then completed through the documented
atomic resume path. The retained commands below reproduce that process:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
test ! -e runs/m2/ppo-full
mkdir -p runs/m2/ppo-full
nohup bash -lc '
  set -euo pipefail
  cd /home/wencong/BotColosseo/.worktrees/m2-base-training
  export PYTHONPATH=/home/wencong/BotColosseo/.worktrees/m2-base-training/src
  /home/wencong/miniconda3/envs/botcolosseo/bin/python -u \
    scripts/train_ppo.py \
    --device cuda:0 \
    --environment-steps 1000000 \
    --rollout-steps 256 \
    --checkpoint-interval-steps 100000 \
    --output-dir runs/m2/ppo-full
  /home/wencong/miniconda3/envs/botcolosseo/bin/python -u \
    scripts/select_ppo.py \
    --run-dir runs/m2/ppo-full \
    --device cuda:0 \
    --pairs-per-opponent 3
' > runs/m2/ppo-full.log 2>&1 &
echo $! > runs/m2/ppo-full.pid
cat runs/m2/ppo-full.pid
```

Monitor without modifying the run:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
tail -n 60 runs/m2/ppo-full.log
ps -p "$(cat runs/m2/ppo-full.pid)" -o pid,etime,%cpu,%mem,stat,cmd
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
/home/wencong/miniconda3/envs/botcolosseo/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("runs/m2/ppo-full/summary.json")
if path.exists():
    summary = json.loads(path.read_text())
    print({
        "environment_steps": summary["environment_steps"],
        "target": 1_000_000,
        "episodes": summary["episode_count"],
        "updates": summary["updates"],
        "kl_early_stops": summary["kl_early_stop_count"],
        "candidate_count": len(summary["candidate_checkpoints"]),
    })
else:
    print("summary.json not written yet")
PY
```

Expected progress is a new JSON summary every 256 environment steps, candidate
checkpoints near each 100,000-step boundary, and finally 30 validation episodes
per candidate. During selection, log lines report `M2 evaluation progress`.
The process is complete only after the PID exits successfully and both
`runs/m2/ppo-full/selection.json` and `runs/m2/ppo-full/selected.pt` exist.

If the process was interrupted and `latest.pt` exists, do not delete the run
directory. Resume the same 1,000,000-step scheduler and then continue selection:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
nohup bash -lc '
  set -euo pipefail
  cd /home/wencong/BotColosseo/.worktrees/m2-base-training
  export PYTHONPATH=/home/wencong/BotColosseo/.worktrees/m2-base-training/src
  /home/wencong/miniconda3/envs/botcolosseo/bin/python -u \
    scripts/train_ppo.py \
    --device cuda:0 \
    --environment-steps 1000000 \
    --rollout-steps 256 \
    --checkpoint-interval-steps 100000 \
    --output-dir runs/m2/ppo-full \
    --resume runs/m2/ppo-full/latest.pt
  /home/wencong/miniconda3/envs/botcolosseo/bin/python -u \
    scripts/select_ppo.py \
    --run-dir runs/m2/ppo-full \
    --device cuda:0 \
    --pairs-per-opponent 3
' >> runs/m2/ppo-full.log 2>&1 &
echo $! > runs/m2/ppo-full.pid
cat runs/m2/ppo-full.pid
```

Do not resume after changing the PPO config, scenario, BC checkpoint, target
steps, or rollout length; provenance checks intentionally reject that drift.
After the process exits, run this exact artifact gate:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
env PYTHONPATH="$PWD/src" \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python - <<'PY'
import hashlib
import json
import math
from pathlib import Path

import torch

root = Path("runs/m2/ppo-full")
summary_path = root / "summary.json"
summary = json.loads(summary_path.read_text())
selection = json.loads((root / "selection.json").read_text())
metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
candidates = summary["candidate_checkpoints"]

assert summary["completed"] is True
assert summary["environment_steps"] == 1_000_000
assert summary["test_cases_accessed"] is False
assert summary["bc_checkpoint_sha256"] == "7eef23a06ea7177d5090ba90be65f8f2f1a847ecb15d81035c21a7e4567949d4"
assert summary["train_cases_hash"] == "e7e2f566e84d457303be2c50c59da8d83add757d0c2f791991ee7f478d400dcb"
assert summary["scenario_hash"] == "91569d20cd52844cfa31284fe8df2886b3d8f2860bacfb6070c5d828511a7cb8"
assert len(candidates) == 10
assert [item["environment_steps"] // 100_000 for item in candidates] == list(range(1, 11))
assert candidates[-1]["environment_steps"] == 1_000_000
for item in candidates:
    path = root / item["checkpoint"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]

training = [item for item in metrics if item["kind"] == "train"]
rollouts = [item for item in metrics if item["kind"] == "rollout"]
assert training and rollouts
assert all(
    math.isfinite(float(item[name]))
    for item in training
    for name in (
        "total_loss", "policy_loss", "value_loss", "entropy",
        "approximate_kl", "pre_clip_grad_norm", "post_clip_grad_norm",
    )
)
assert all(float(item["post_clip_grad_norm"]) <= 0.50001 for item in training)
assert summary["kl_early_stop_count"] < len(rollouts)

assert selection["split"] == "validation"
assert selection["test_cases_accessed"] is False
assert selection["pairs_per_opponent"] == 3
assert selection["episodes_per_candidate"] == 30
assert len(selection["candidates"]) == len(candidates)
assert all(item["episodes"] == 30 for item in selection["candidates"])
assert all(item["protocol_inconsistencies"] == 0 for item in selection["candidates"])
assert selection["training_summary_sha256"] == hashlib.sha256(summary_path.read_bytes()).hexdigest()
selected_path = root / "selected.pt"
selected_sha = hashlib.sha256(selected_path.read_bytes()).hexdigest()
assert selected_sha == selection["selected_checkpoint_sha256"]
assert selected_sha == selection["selected"]["checkpoint_sha256"]
checkpoint = torch.load(selected_path, map_location="cpu", weights_only=False)
assert checkpoint["metadata"]["scenario_hash"] == summary["scenario_hash"]
assert checkpoint["metadata"]["config_hash"] == summary["config_hash"]
assert checkpoint["metadata"]["counters"]["environment_steps"] == selection["selected"]["environment_steps"]
print("M2 full PPO training and validation-selection gate: PASS")
print(json.dumps(selection["selected"], indent=2, sort_keys=True))
PY
test "$(rg -c '^Traceback \(most recent call last\):' runs/m2/ppo-full.log)" -eq 1
test "$(rg -c '^RuntimeError: Duel respawn did not complete within the warm-up limit$' runs/m2/ppo-full.log)" -eq 1
ps -eo pid,ppid,stat,cmd | rg '[b]otcolosseo-duel|[v]izdoom|[t]rain_ppo.py|[s]elect_ppo.py' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
git status --short
```

Return the final `tail -n 100 runs/m2/ppo-full.log`, the artifact-gate output,
the process/GPU checks, and `git status --short`. Do not run the official M2
test evaluator; checkpoint selection and its tracked provenance must be
reviewed and committed first.

## Action required: recovered official M2 paired learning gate

The evaluator, frozen thresholds, BC/PPO validation selections, and bounded
same-case respawn retry are committed before this run. The official test uses
three policies, five opponents, 50 seed-pairs per opponent, and both learner
sides: exactly 1,500 games. It is expected to take approximately 2.5--3 hours.
Do not inspect partial policy outcomes or change code/config after starting.

Two attempts on 2026-07-21 stopped at 1,470/1,500 without writing any official
or temporary evidence. Both exposed the same environment lifecycle bug: a
death immediately before the 2,100-tic engine timeout entered the respawn loop
after the episode had finished. Commit `acf4d8a` makes that boundary a normal
truncation and adds a deterministic regression test. This is infrastructure
recovery, not a policy rerun after observing test results; selections, seeds,
thresholds, episode horizon, and evaluation rows remain frozen.

Run this preflight first. It verifies tracked selection provenance and loads
both learned Actors without parsing test episode rows:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
test -z "$(git status --porcelain)"
git merge-base --is-ancestor acf4d8a HEAD
test ! -e reports/m2/episodes.csv
test ! -e reports/m2/summary.json
test ! -e reports/m2/manifest.json
test ! -e reports/m2/.episodes.csv.tmp
test ! -e reports/m2/.summary.json.tmp
test ! -e reports/m2/.manifest.json.tmp
env PYTHONPATH="$PWD/src" \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

import torch

from botcolosseo.evaluation.m2 import load_actor_policy

root = Path.cwd()
scenario = json.loads(
    (root / "assets/scenarios/crystal_run/manifest.json").read_text()
)["wad_sha256"]
expected = {
    "bc": root / "runs/m2/bc-full/best.pt",
    "ppo": root / "runs/m2/ppo-full/selected.pt",
}
for name, checkpoint in expected.items():
    report = json.loads(
        (root / f"reports/m2/{name}-training-summary.json").read_text()
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report["selected_checkpoint_sha256"] == digest
    assert report["test_cases_accessed"] is False
    load_actor_policy(
        name,
        checkpoint,
        device=torch.device("cpu"),
        expected_scenario_hash=scenario,
    )
assert hashlib.sha256((root / "configs/m2/test.json").read_bytes()).hexdigest() == (
    "b26dedbd7a82d0ac53082a55dd723cbb2c9e05559781e84ecb7295f68c4f2bb5"
)
print("M2 official preflight: PASS")
PY
```

Archive the second failed-attempt log and start one clean recovery run. Do not
append to a failed log because the final evidence gate rejects historical
tracebacks:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
mkdir -p runs/m2
if test -f runs/m2/official-evaluation.pid && \
   ps -p "$(cat runs/m2/official-evaluation.pid)" >/dev/null 2>&1; then
  echo "official evaluation is already running" >&2
  exit 1
fi
if test -f runs/m2/official-evaluation.log; then
  failed_evaluation_log="runs/m2/official-evaluation.failed-$(date +%Y%m%d-%H%M%S).log"
  mv runs/m2/official-evaluation.log "$failed_evaluation_log"
  echo "archived $failed_evaluation_log"
fi
nohup env \
  PYTHONPATH=/home/wencong/BotColosseo/.worktrees/m2-base-training/src \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python -u \
  scripts/evaluate_m2.py \
  --split test \
  --device cuda:0 \
  --output reports/m2 \
  > runs/m2/official-evaluation.log 2>&1 &
echo $! > runs/m2/official-evaluation.pid
cat runs/m2/official-evaluation.pid
```

Monitor process health only; do not use the partial log to tune or compare
policies:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
ps -p "$(cat runs/m2/official-evaluation.pid)" -o pid,etime,%cpu,%mem,stat,cmd
tail -n 20 runs/m2/official-evaluation.log
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

The evaluator writes the three official artifacts only after all games finish.
If the process is externally interrupted before those files exist, preserve
that failed log under a timestamped `official-evaluation.failed-*.log` name and
restart with a fresh `>` log; do not change any selection, seed, threshold, or
policy. After the PID exits, run this gate:

```bash
cd /home/wencong/BotColosseo/.worktrees/m2-base-training
env PYTHONPATH="$PWD/src" \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/audit_m2_evidence.py
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m json.tool \
  reports/m2/summary.json
! rg -n "Traceback|ValueError|RuntimeError|FloatingPointError" runs/m2/official-evaluation.log
ps -eo pid,ppid,stat,args | rg '[b]otcolosseo-duel|[v]izdoom|[e]valuate_m2.py' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
git status --short
```

Return the official gate output, final log tail, and process/GPU checks. A
failed performance gate is still the official result: do not rerun, tune, or
replace checkpoints after observing it.

## Action required: M3 Strong Base integrity-qualified pipeline

The recovered M2 run completed 1,500/1,500 rows with zero protocol and artifact
inconsistencies, but the frozen capability gate failed. The result remains M2
FAIL: PPO win rate was 77.0% versus BC 75.2%, PPO objective completion was
93.2% versus BC 97.8%, and PPO won 23% against `objective_first`. M3 therefore
uses the reviewed `integrity-qualified` route. This does not change the M2
thresholds or consume M2 test rows during M3 training, admission, or selection.

The driver is serial and restart-aware. It copies the immutable M2 checkpoint
and evidence into the M3 worktree, verifies strict-audit failure plus
integrity-audit success, produces fresh validation-only anchor evidence, trains
to 2,000,000 environment steps, validates candidates at 200k boundaries,
updates PFSP only at completed side-swapped boundaries, selects Strong Base on
validation, then opens M3 test exactly once. It refuses test access unless the
active pool contains 8--12 policies. Physical GPU 1 is exposed as the process's
`cuda:0`; physical GPU 0 is not used.

Run this short preflight first. It does not start validation or training:

```bash
cd /home/wencong/BotColosseo/.worktrees/m3-strong-base
export PYTHONPATH="$PWD/src"
bash -n scripts/run_m3_pipeline.sh
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests scripts
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pip check
M3_PREFLIGHT_ONLY=1 M3_PHYSICAL_GPU=1 scripts/run_m3_pipeline.sh
```

Expected final line:

```text
M3 PIPELINE PREFLIGHT PASS
```

Start the complete long pipeline with this single command:

```bash
cd /home/wencong/BotColosseo/.worktrees/m3-strong-base
mkdir -p runs/m3
test ! -f runs/m3/pipeline.pid || \
  ! ps -p "$(cat runs/m3/pipeline.pid)" >/dev/null 2>&1
: > runs/m3/pipeline.pid
nohup setsid -f bash -c '
  echo $$ > runs/m3/pipeline.pid
  set -o pipefail
  cd /home/wencong/BotColosseo/.worktrees/m3-strong-base
  env \
    BOTCOLOSSEO_PYTHON=/home/wencong/miniconda3/envs/botcolosseo/bin/python \
    M2_WORKTREE=/home/wencong/BotColosseo/.worktrees/m2-base-training \
    M3_PHYSICAL_GPU=1 \
    scripts/run_m3_pipeline.sh
  status=$?
  printf "%s\n" "$status" > runs/m3/pipeline.exit
  exit "$status"
' >> runs/m3/pipeline.log 2>&1
while [[ ! -s runs/m3/pipeline.pid ]]; do sleep 1; done
cat runs/m3/pipeline.pid
```

The recorded PID is the detached session leader. If the pipeline must be
stopped, terminate the entire M3 process group so that the active trainer or
evaluator cannot survive its wrapper:

```bash
cd /home/wencong/BotColosseo/.worktrees/m3-strong-base
kill -TERM -- "-$(cat runs/m3/pipeline.pid)"
```

The expected end-to-end duration is approximately 12--20 hours. The largest
components are the fresh 500-game M2 validation anchor, ten league phases,
validation cross-play as the pool grows, and the final 1,340--1,660-game M3
official suite. Environment synchronization, rather than A100 utilization, is
usually the bottleneck.

Use these commands to inspect progress without changing the run:

```bash
cd /home/wencong/BotColosseo/.worktrees/m3-strong-base
ps -p "$(cat runs/m3/pipeline.pid)" -o pid,etime,%cpu,%mem,stat,cmd
tail -n 80 runs/m3/pipeline.log
rg 'M2 evaluation progress|M3 cross-play progress|M3 evaluation progress|environment_steps|PIPELINE' \
  runs/m3/pipeline.log | tail -n 30
test ! -f runs/m3/league-full/summary.json || \
  jq '{environment_steps,episode_count,updates,opponent_source_counts,pfsp_probabilities}' \
  runs/m3/league-full/summary.json
test ! -f runs/m3/league-full/pipeline-state.json || \
  jq . runs/m3/league-full/pipeline-state.json
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader
test ! -f runs/m3/pipeline.exit || cat runs/m3/pipeline.exit
```

Recovery rules:

- If training is interrupted, rerun the same `nohup` command. The driver reads
  `pipeline-state.json`; if `latest.pt` matches the current pool/payoff identity,
  it uses exact `--resume`. If a validated pool changed, it uses the archived
  parent candidate with `--transition-from`.
- The 2026-07-22 run exposed an old boundary-padding defect at 1.8M: repeatedly
  launching 256-step processes could not finish the pending episode. The fixed
  driver detects this exact signature, archives the pre-fix `latest.pt`,
  `summary.json`, and `metrics.jsonl` under `runs/m3/league-full/recovery/`,
  then resumes from the immutable 1.8M checkpoint and finishes the side-swapped
  pair inside one trainer process. Do not manually delete or edit those rows.
- Official M3 rows are append-only. When `reports/m3/official/episodes.jsonl`
  exists, the driver passes `--resume`; it verifies the complete run identity,
  suppresses only exact duplicates, and never deletes valid rows.
- Candidate cross-play writes its three evidence files only after completing
  the matrix. If the driver reports a partial candidate directory, preserve it
  by moving that one directory to
  `runs/m3/recovery/<timestamp>/`; never remove a completed matrix whose
  `manifest.json` hashes match its CSV and matrix.
- Do not edit configs, manifests, pool JSON, payoff JSON, checkpoint files, or
  selection reports between attempts. Identity drift is intentionally fatal.
- A nonzero `pipeline.exit` caused by pool size below 8 or a failed M3 gate is
  an experimental result, not permission to extend the 2M budget or tune from
  M3 test rows.

Completion requires all of the following:

```bash
cd /home/wencong/BotColosseo/.worktrees/m3-strong-base
test "$(cat runs/m3/pipeline.exit)" -eq 0
tail -n 1 runs/m3/pipeline.log | rg '^M3 PIPELINE PASS$'
env PYTHONPATH="$PWD/src" \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/audit_m3_evidence.py --report-dir reports/m3/official
jq '{passed, gates, selected_checkpoint_sha256, pool_manifest_sha256}' \
  reports/m3/official/summary.json
```

Process exit alone is not success. Only an audit return code of zero, a frozen
M3 gate PASS, matching selected checkpoint/pool hashes, and the literal final
marker `M3 PIPELINE PASS` complete M3.

## M4 Showcase foundation

This foundation is not an M4 pass. The development command below uses the
frozen M2 PPO/BC checkpoints and one M2 validation case only to verify real
ViZDoom recording, MP4/GIF encoding, deterministic selection, and hash-bound
publication mechanics. Its output is ignored under
`artifacts/showcase-development/`, never updates README, and is not an official
test result or evidence of a learned style.

Run the development renderer from the M4 worktree:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-showcase-foundation
env \
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$PWD/src" \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/render_showcase.py \
  --config configs/showcase/development.yaml \
  --checkpoint-root /home/wencong/BotColosseo/.worktrees/m3-strong-base \
  --device cuda:0
```

The real Strong Base and Aggressive hashes and the passing 200-episode M4
validation evidence are now frozen. Export the publication metrics and logged
highlight scores from those artifacts and the eight predefined validation
cases:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/export_showcase_metrics.py \
  --config configs/showcase/m4.yaml \
  --evaluation reports/m4/evaluation/aggressive-alpha-025/summary.json \
  --checkpoint-root . \
  --device cuda:0
```

Commit the resulting `reports/m4/showcase-metrics.json` so production Git
provenance is clean, then publish the media and manifest:

```bash
env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/render_showcase.py \
  --config configs/showcase/m4.yaml \
  --checkpoint-root . \
  --device cuda:0
```

The publication loader rejects dirty provenance, checkpoint drift, failed gate
booleans, non-validation cases, or any policy/hash mismatch. The tracked M4
publication is qualitative validation material and is not an official test
result.

## M4 Aggressive candidate training

The 1,024-step CUDA/ViZDoom smoke completed on 2026-07-22 with a real episode,
16 optimizer updates, capped reward-component logs, and finite style-to-base
KL. Run the first 400k production candidate on the second physical GPU:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
mkdir -p runs/m4/aggressive-main
nohup env \
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH="$PWD/src" \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python -u \
  scripts/train_league.py \
  --style aggressive \
  --config configs/m4/aggressive.yaml \
  --base-checkpoint runs/m3/league-full/candidate-boundary-0200000.pt \
  --pool /home/wencong/BotColosseo/.worktrees/m3-strong-base/reports/m3/pools/pool-v1.json \
  --payoffs /home/wencong/BotColosseo/.worktrees/m3-strong-base/reports/m3/pools/payoffs-v1.json \
  --run-dir runs/m4/aggressive-main \
  --device cuda:0 \
  > runs/m4/aggressive-main/train.log 2>&1 &
echo $! > runs/m4/aggressive-main/train.pid
```

Monitor without modifying the run:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
tail -f runs/m4/aggressive-main/train.log
```

Or inspect structured progress:

```bash
jq '{completed, environment_steps, episode_count, updates,
     style_reward_components, kl_early_stop_count}' \
  runs/m4/aggressive-main/summary.json
```

Expected completion is `environment_steps: 400000` and `completed: true`.
This completes training only; it does not claim the Aggressive or M4 gate.

## M4 Aggressive alpha 0.25 formal evaluation

The predefined interpolation grid selected alpha 0.25. Its 20-episode smoke
passed six of seven gates; the only miss was an engagement bootstrap interval
with lower bound exactly zero. The explicit 2026-07-23 route approval permits
the fixed candidate to enter the unchanged 200-episode validation evaluation.
It does not permit more alpha tuning or relax any formal gate.

The preflight uses 10 side-swapped pairs for each of five opponents and two
policies, for exactly 200 episodes:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
env \
  PYTHONPATH=src \
  CUDA_VISIBLE_DEVICES=0 \
  /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/evaluate_style.py \
  --base-checkpoint runs/m3/league-full/candidate-boundary-0200000.pt \
  --aggressive-checkpoint runs/m4/aggressive-interpolation/alpha-025.pt \
  --output-dir reports/m4/evaluation/aggressive-alpha-025 \
  --pairs-per-opponent 10 \
  --max-decisions 525 \
  --max-attempts 2 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260722 \
  --device cuda:0 \
  --preflight
```

Run the resumable evaluation in a detached session:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
mkdir -p runs/m4 reports/m4/evaluation
test ! -s runs/m4/aggressive-alpha-025-evaluation.pid || \
  ! ps -p "$(cat runs/m4/aggressive-alpha-025-evaluation.pid)" >/dev/null 2>&1
: > runs/m4/aggressive-alpha-025-evaluation.pid
rm -f runs/m4/aggressive-alpha-025-evaluation.exit
nohup setsid -f bash -c '
  echo $$ > runs/m4/aggressive-alpha-025-evaluation.pid
  cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
  env \
    PYTHONPATH=src \
    CUDA_VISIBLE_DEVICES=0 \
    /home/wencong/miniconda3/envs/botcolosseo/bin/python -u \
    scripts/evaluate_style.py \
    --base-checkpoint runs/m3/league-full/candidate-boundary-0200000.pt \
    --aggressive-checkpoint runs/m4/aggressive-interpolation/alpha-025.pt \
    --output-dir reports/m4/evaluation/aggressive-alpha-025 \
    --pairs-per-opponent 10 \
    --max-decisions 525 \
    --max-attempts 2 \
    --bootstrap-samples 10000 \
    --bootstrap-seed 20260722 \
    --device cuda:0
  status=$?
  printf "%s\n" "$status" > runs/m4/aggressive-alpha-025-evaluation.exit
  exit "$status"
' >> runs/m4/aggressive-alpha-025-evaluation.log 2>&1
while [[ ! -s runs/m4/aggressive-alpha-025-evaluation.pid ]]; do sleep 1; done
cat runs/m4/aggressive-alpha-025-evaluation.pid
```

Monitor without changing the append-only ledger:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
ps -p "$(cat runs/m4/aggressive-alpha-025-evaluation.pid)" \
  -o pid,etime,%cpu,%mem,stat,cmd
wc -l reports/m4/evaluation/aggressive-alpha-025/episodes.jsonl
tail -n 40 runs/m4/aggressive-alpha-025-evaluation.log
test ! -f runs/m4/aggressive-alpha-025-evaluation.exit || \
  cat runs/m4/aggressive-alpha-025-evaluation.exit
```

Rerunning the same command resumes from verified episode identities. Do not
delete a partial `episodes.jsonl`. Completion requires 200 rows plus matching
`summary.json` and `manifest.json`; exit code zero means the frozen formal gate
passed, while exit code one is a complete but failed experimental result.

## M5 Defensive production pipeline

This is the frozen Defensive route: 50,000 risk-conditioned train transitions,
1,000 adapter-distillation updates, the fixed alpha grid `0.25/0.50/0.75`,
three 20-episode validation smokes, deterministic candidate selection, and a
200-episode formal validation gate. It uses no test cases. A failed data,
offline, selection, or formal gate preserves its artifacts and exits nonzero.

The completed production data manifest recorded 95 of the predefined 100
denial/recovery windows, while every other data check passed. The project owner
approved one manifest-specific waiver on 2026-07-23. The source manifest remains
`passed: false`; `reports/m5/defensive/data-waiver.json` is bound to its exact
SHA-256 and permits distillation only for that artifact. All offline, smoke,
formal, and publication thresholds remain unchanged.

Launch it on the second physical GPU:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
mkdir -p runs/m5
test ! -s runs/m5/defensive.pid || \
  ! ps -p "$(cat runs/m5/defensive.pid)" >/dev/null 2>&1
rm -f runs/m5/defensive.exit
nohup setsid -f bash -c '
  echo $$ > runs/m5/defensive.pid
  cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
  CUDA_VISIBLE_DEVICES=1 scripts/run_m5_defensive.sh
  status=$?
  printf "%s\n" "$status" > runs/m5/defensive.exit
  exit "$status"
' >> runs/m5/defensive.log 2>&1
while [[ ! -s runs/m5/defensive.pid ]]; do sleep 1; done
cat runs/m5/defensive.pid
```

Monitor without modifying artifacts:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
ps -p "$(cat runs/m5/defensive.pid)" -o pid,etime,%cpu,%mem,stat,cmd
tail -n 60 runs/m5/defensive.log
find reports/m5/defensive -name episodes.jsonl -print -exec wc -l {} \;
test ! -f runs/m5/defensive.exit || cat runs/m5/defensive.exit
```

Do not delete a partial smoke or formal `episodes.jsonl`; those stages resume
from hash-bound identities. If selection exits one, none of the predefined
alpha candidates passed every smoke gate, so the formal evaluation is not run.

## M5 Explorer production pipeline

This route generates 50,000 score-conditioned route demonstrations, performs
1,000 frozen-base adapter-distillation updates, evaluates the fixed alpha grid
`0.25/0.50/0.75`, and selects a candidate only when its 20-episode validation
smoke passes every capability, route-diversity, anti-wandering, and protocol
gate. The selected candidate then receives a 200-episode formal validation and
hash-chain audit. No test cases are accessed.

Launch it on the second physical GPU:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
mkdir -p runs/m5
test ! -s runs/m5/explorer.pid || \
  ! ps -p "$(cat runs/m5/explorer.pid)" >/dev/null 2>&1
rm -f runs/m5/explorer.exit
nohup setsid -f bash -c '
  echo $$ > runs/m5/explorer.pid
  cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
  CUDA_VISIBLE_DEVICES=1 scripts/run_m5_explorer.sh
  status=$?
  printf "%s\n" "$status" > runs/m5/explorer.exit
  exit "$status"
' >> runs/m5/explorer.log 2>&1
while [[ ! -s runs/m5/explorer.pid ]]; do sleep 1; done
cat runs/m5/explorer.pid
```

Monitor without changing artifacts:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
ps -p "$(cat runs/m5/explorer.pid)" -o pid,etime,%cpu,%mem,stat,cmd
tail -n 60 runs/m5/explorer.log
find reports/m5/explorer -name episodes.jsonl -print -exec wc -l {} \;
test ! -f runs/m5/explorer.exit || cat runs/m5/explorer.exit
```

The data generator prints only its final manifest, so GPU utilization and log
growth are the progress signals during that first stage. Distillation prints
metrics every 25 updates; evaluation prints an episode counter. Partial
evaluation ledgers resume from exact identities. If production data or all
three smoke candidates fail their frozen gates, artifacts are preserved and the
pipeline exits nonzero without inventing a waiver.

## M5 difficulty production pipeline

The frozen difficulty controller uses the same Strong Base and passed
Aggressive checkpoints at every tier. Hard is the native policy, Normal adds a
one-decision reaction delay, and Easy adds a two-decision delay plus a
two-decision policy-update interval. No health, damage, observation, or model
weight changes are permitted.

The pipeline first runs a 60-episode validation smoke. A complete,
protocol-clean smoke may proceed even if its small-sample monotonic gate is
inconclusive. Formal evidence uses 600 paired validation episodes and must pass
all monotonicity, objective-capability, style-direction, integrity, and
zero-test-access checks.

Launch it on the first physical GPU:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
mkdir -p runs/m5
test ! -s runs/m5/difficulty.pid || \
  ! ps -p "$(cat runs/m5/difficulty.pid)" >/dev/null 2>&1
rm -f runs/m5/difficulty.exit
nohup setsid -f bash -c '
  echo $$ > runs/m5/difficulty.pid
  cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
  CUDA_VISIBLE_DEVICES=0 scripts/run_m5_difficulty.sh
  status=$?
  printf "%s\n" "$status" > runs/m5/difficulty.exit
  exit "$status"
' >> runs/m5/difficulty.log 2>&1
while [[ ! -s runs/m5/difficulty.pid ]]; do sleep 1; done
cat runs/m5/difficulty.pid
```

Monitor the append-only ledgers:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
ps -p "$(cat runs/m5/difficulty.pid)" -o pid,etime,%cpu,%mem,stat,cmd
tail -n 60 runs/m5/difficulty.log
find reports/m5/difficulty -name episodes.jsonl -print -exec wc -l {} \;
test ! -f runs/m5/difficulty.exit || cat runs/m5/difficulty.exit
```

The evaluator prints one progress line per episode and resumes only when the
checkpoint, config, cases, and schedule identities are exact matches.

## M5 Defensive closed-loop PPO repair

The first Defensive distillation route preserved skill but failed to create a
stable protective-presence shift. This repair keeps the frozen gate unchanged,
warm-starts the existing alpha-0.25 checkpoint, and trains the adapter, copied
policy head, and critic for 200,000 environment steps with capped
risk-conditioned reward plus style-to-base KL.

The pipeline first runs a 2,000-step real CUDA/ViZDoom smoke and requires
non-empty Defensive reward components. It then runs the 200k production
training, a 20-episode all-gate smoke, the unchanged 200-episode formal
validation, and a separate PPO hash-chain audit.

Launch it when one physical GPU is free:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
mkdir -p runs/m5
test ! -s runs/m5/defensive-ppo.pid || \
  ! ps -p "$(cat runs/m5/defensive-ppo.pid)" >/dev/null 2>&1
rm -f runs/m5/defensive-ppo.exit
nohup setsid -f bash -c '
  echo $$ > runs/m5/defensive-ppo.pid
  cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
  CUDA_VISIBLE_DEVICES=1 scripts/run_m5_defensive_ppo.sh
  status=$?
  printf "%s\n" "$status" > runs/m5/defensive-ppo.exit
  exit "$status"
' >> runs/m5/defensive-ppo.log 2>&1
while [[ ! -s runs/m5/defensive-ppo.pid ]]; do sleep 1; done
cat runs/m5/defensive-ppo.pid
```

Monitor training and later evaluation:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
ps -p "$(cat runs/m5/defensive-ppo.pid)" -o pid,etime,%cpu,%mem,stat,cmd
tail -n 60 runs/m5/defensive-ppo.log
wc -l runs/m5/defensive-ppo-main/metrics.jsonl 2>/dev/null || true
find reports/m5/defensive/ppo-repair -name episodes.jsonl \
  -print -exec wc -l {} \;
test ! -f runs/m5/defensive-ppo.exit || cat runs/m5/defensive-ppo.exit
```

An exit code of one after an evaluation means the complete experimental result
failed a frozen style gate; its artifacts must be retained.

## M5 Explorer closed-loop PPO repair

The first Explorer route passed the offline KL gate but never completed a
flank route in closed loop. This repair keeps the frozen evaluator unchanged,
warm-starts alpha 0.25, and trains the adapter, copied policy head, and critic
for 200,000 environment steps. Its capped training-only reward recognizes
score-conditioned route milestones only while carrying the core; non-carry
wandering earns no style reward.

The pipeline runs a 2,000-step real CUDA/ViZDoom smoke, requires non-empty
Explorer reward components, then runs production training, a 20-episode
all-gate smoke, the unchanged 200-episode formal validation, and a separate
hash-chain audit.

Launch it when one physical GPU is free:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
mkdir -p runs/m5
test ! -s runs/m5/explorer-ppo.pid || \
  ! ps -p "$(cat runs/m5/explorer-ppo.pid)" >/dev/null 2>&1
rm -f runs/m5/explorer-ppo.exit
nohup setsid -f bash -c '
  echo $$ > runs/m5/explorer-ppo.pid
  cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
  CUDA_VISIBLE_DEVICES=1 scripts/run_m5_explorer_ppo.sh
  status=$?
  printf "%s\n" "$status" > runs/m5/explorer-ppo.exit
  exit "$status"
' >> runs/m5/explorer-ppo.log 2>&1
while [[ ! -s runs/m5/explorer-ppo.pid ]]; do sleep 1; done
cat runs/m5/explorer-ppo.pid
```

Monitor training and later evaluation:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
ps -p "$(cat runs/m5/explorer-ppo.pid)" -o pid,etime,%cpu,%mem,stat,cmd
tail -n 60 runs/m5/explorer-ppo.log
wc -l runs/m5/explorer-ppo-main/metrics.jsonl 2>/dev/null || true
find reports/m5/explorer/ppo-repair -name episodes.jsonl \
  -print -exec wc -l {} \;
test ! -f runs/m5/explorer-ppo.exit || cat runs/m5/explorer-ppo.exit
```

An exit code of one after evaluation is a preserved frozen-gate failure, not
permission to change the evaluator.

## M6 anonymous user-study package

Run this only after the three public style videos are generated from passing
M5 checkpoints:

```bash
cd /home/wencong/BotColosseo/.worktrees/m4-aggressive
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/prepare_user_study.py \
  --aggressive docs/assets/showcase/m6-aggressive.mp4 \
  --defensive docs/assets/showcase/m6-defensive.mp4 \
  --explorer docs/assets/showcase/m6-explorer.mp4 \
  --output-dir artifacts/m6-user-study \
  --assignments 10
```

After collecting the six anonymous fields documented in
`docs/user-study.md`, analyze them with:

```bash
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/analyze_user_study.py \
  --package-dir artifacts/m6-user-study \
  --responses reports/m6/user-study/responses.csv \
  --output reports/m6/user-study/summary.json
```
