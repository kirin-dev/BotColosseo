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

## Action required: M2 full PPO training and validation-only selection

The 20,000-step A100 pilot, 2,000-step stop/resume gate, candidate snapshot
gate, and validation-only selector have passed. The full run uses 1,000,000
environment steps on GPU 0. It saves one atomic candidate per 100,000-step
bucket, then evaluates every candidate on three frozen validation seed-pairs
per opponent with both learner sides. Selection is lexicographic by objective
rate, win rate, mean score difference, then earlier environment step. No test
case is read by training or selection.

The combined training and selection run is expected to take roughly six hours
based on the measured 20,000-step pilot. Start it exactly once from the M2
worktree:

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
! rg -n "Traceback|ValueError|RuntimeError|FloatingPointError" runs/m2/ppo-full.log
ps -eo pid,ppid,stat,cmd | rg '[b]otcolosseo-duel|[v]izdoom|[t]rain_ppo.py|[s]elect_ppo.py' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
git status --short
```

Return the final `tail -n 100 runs/m2/ppo-full.log`, the artifact-gate output,
the process/GPU checks, and `git status --short`. Do not run the official M2
test evaluator; checkpoint selection and its tracked provenance must be
reviewed and committed first.
