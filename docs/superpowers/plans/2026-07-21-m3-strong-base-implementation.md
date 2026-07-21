# Milestone 3 Auditable League and Strong Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox notation so progress survives context changes.

**Goal:** Turn the M2 PPO policy into an auditable Strong Base through immutable historical opponents, deterministic PFSP league training, complete cross-play, held-out evaluation, and machine-verifiable selection evidence.

**Architecture:** Add a league overlay around the existing recurrent PPO runtime. M3 owns new split manifests, opponent descriptors, historical-pool state, PFSP scheduling, training/evaluation CLIs, and evidence. It reuses the legal Actor observation, model, rollout buffer, PPO loss, duel runtime, and reward ledger without changing the M2 evaluator or scenario bytes.

**Tech Stack:** Python 3.10, PyTorch 2.6, NumPy, PyYAML, Matplotlib, ViZDoom 1.3.0, pytest, Ruff.

**Design:** `docs/superpowers/specs/2026-07-21-m3-strong-base-design.md`

## Global Constraints

- Run Python as `/home/wencong/miniconda3/envs/botcolosseo/bin/python`.
- Before Python/pytest/script commands in the linked worktree, run `export PYTHONPATH="$PWD/src"`; the conda environment's editable install points at the main checkout and must not shadow worktree code.
- Work only in `.worktrees/m3-strong-base`; do not inspect, modify, or monitor the running M2 official evaluation.
- Treat `runs/m2/ppo-full/selected.pt` as provisional until the M2 artifact gate passes.
- Do not change the WAD, scenario hash, 13-action contract, Actor observation schema, `configs/m2/`, or frozen M2 evaluator.
- Learned opponents receive only `DuelActorObservation`; privileged state is available only to named script Teachers and the Critic during training.
- Freeze train/validation/test/held-out manifests before long training. Never use M3 test rows for scheduling, admission, tuning, or selection.
- Use strict TDD: add one focused failing test, record RED, implement the minimum, record GREEN, then commit.
- Use atomic writes (`tempfile` in the destination directory plus `os.replace`) for pool, payoff, checkpoint, and evidence artifacts.
- Generated runs remain ignored. Only compact manifests, reports, plots, model cards, and deliberately selected public artifacts are tracked.
- When the first required command exceeds a short foreground smoke, write the exact `nohup` handoff into `script.md`, verify its preflight command, mark the goal blocked, and stop.

## Frozen Interfaces

The following names are the cross-task contract. Change them only through a reviewed plan amendment.

```python
@dataclass(frozen=True)
class LeagueCase:
    split: str
    pair_index: int
    seed: int
    learner_side: str
    core_spawn_index: int
    route: str

@dataclass(frozen=True)
class OpponentSpec:
    opponent_id: str
    kind: Literal["script", "checkpoint"]
    checkpoint: str | None
    checkpoint_sha256: str | None
    scenario_hash: str
    selection_evidence: str

@dataclass(frozen=True)
class PoolEntry:
    policy_id: str
    checkpoint: str
    checkpoint_sha256: str
    scenario_hash: str
    config_hash: str
    source_git_commit: str
    parent_checkpoint_sha256: str
    environment_steps: int
    admitted_at_utc: str
    validation_report: str
    validation_report_sha256: str
    script_average_win_rate: float
    script_worst_case_win_rate: float
    objective_rate: float
    payoff_by_policy: dict[str, float]
    anchor: bool
    admission_reason: str

@dataclass(frozen=True)
class LeagueEpisodeAssignment:
    pair_slot: int
    case: LeagueCase
    opponent: OpponentSpec
    source: Literal["script", "pfsp", "uniform_history"]
    sampling_probability: float
```

The pool manifest schema version is `1`. The league master seed and bootstrap seed are both `20260721`.

---

### Task 1: Freeze the neutral M3 split contract

**Files:**

- Create: `src/botcolosseo/scenarios/league_splits.py`
- Create: `src/botcolosseo/cli/generate_league_splits.py`
- Create: `scripts/generate_league_splits.py`
- Create: `configs/m3/train.json`
- Create: `configs/m3/validation.json`
- Create: `configs/m3/test.json`
- Create: `configs/m3/heldout.json`
- Create: `tests/unit/test_league_splits.py`

- [x] Write tests requiring 250/50/50/50 seed-pairs, two side-swapped rows per pair, signed-32-bit seeds, split-disjoint seeds, balanced core/route labels, deterministic bytes, and strict schema rejection.
- [x] Run RED:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_league_splits.py -q
```

Expected: import failure for `botcolosseo.scenarios.league_splits`.

- [x] Implement these public functions:

```python
PAIR_COUNTS = {"train": 250, "validation": 50, "test": 50, "heldout": 50}

def generate_league_splits(*, master_seed: int = 20260721) -> dict[str, tuple[LeagueCase, ...]]: ...
def write_league_manifests(cases: Mapping[str, Sequence[LeagueCase]], root: Path) -> None: ...
def load_league_cases(path: Path, *, expected_split: str, expected_pairs: int) -> tuple[LeagueCase, ...]: ...
```

Generation must use one local `random.Random(master_seed)`, sample unique seeds before assigning splits, and emit canonical sorted/indented JSON with a final newline. `LeagueCase.to_duel_case(opponent_id)` may adapt to the existing runtime; core/route are protocol strata, not claims that the current WAD consumes them.

- [x] Add a thin CLI with `--output-root` and `--master-seed`; generate the four committed manifests once.
- [x] Run GREEN and determinism check:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_league_splits.py -q
/home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/generate_league_splits.py --output-root /tmp/m3-splits-a
/home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/generate_league_splits.py --output-root /tmp/m3-splits-b
diff -ru /tmp/m3-splits-a /tmp/m3-splits-b
```

Expected: tests pass and `diff` is empty.

- [x] Commit: `feat: freeze M3 league splits`.

### Task 2: Implement auditable opponent descriptors and legal checkpoint policies

**Files:**

- Create: `src/botcolosseo/agents/league_opponents.py`
- Create: `tests/unit/test_league_opponents.py`

- [x] Test `OpponentSpec` validation, checkpoint hash mismatch, scenario mismatch, deterministic recurrent reset, greedy action selection, episode-only hidden-state reset, and absence of privileged arguments from `CheckpointOpponentPolicy.act`.
- [x] Run RED for the focused file.
- [x] Implement:

```python
class CheckpointOpponentPolicy:
    @classmethod
    def load(cls, spec: OpponentSpec, *, device: torch.device) -> "CheckpointOpponentPolicy": ...
    def reset(self) -> None: ...
    def act(self, observation: DuelActorObservation) -> MacroAction: ...

class ScriptOpponentPolicy:
    def reset(self, *, seed: int) -> None: ...
    def act(self, state: DuelPrivilegedState) -> MacroAction: ...

def sha256_file(path: Path) -> str: ...
```

`CheckpointOpponentPolicy.load` must instantiate the existing recurrent actor architecture, verify declared hashes before loading, use `torch.inference_mode()`, take `argmax`, and retain hidden state between decisions. Never add `DuelPrivilegedState` to its method signature or object fields.

- [x] Run GREEN plus `ruff check` on the two files.
- [x] Commit: `feat: add legal league opponent policies`.

### Task 3: Add the immutable historical-pool schema and admission rules

**Files:**

- Create: `src/botcolosseo/training/historical_pool.py`
- Create: `tests/unit/test_historical_pool.py`

- [x] Test schema versioning, duplicate IDs/hashes, missing artifacts, canonical round-trip, manifest self-hash, immutable old entries, 8–12 capacity behavior, and each admission rejection independently.
- [x] Run RED.
- [x] Implement:

```python
@dataclass(frozen=True)
class AdmissionMetrics:
    integrity_ok: bool
    validation_complete: bool
    paired_side_swapped: bool
    protocol_inconsistencies: int
    source_split: str
    candidate_script_average: float
    active_script_average: float
    candidate_historical_worst_case: float
    active_historical_worst_case: float
    candidate_payoffs: dict[str, float]
    active_payoffs: dict[str, dict[str, float]]

def load_pool(path: Path, *, verify_artifacts: bool = True, artifact_root: Path | None = None) -> HistoricalPoolManifest: ...
def write_pool_atomic(pool: HistoricalPoolManifest, path: Path) -> None: ...
def admission_decision(pool: HistoricalPoolManifest, entry: PoolEntry, metrics: AdmissionMetrics) -> AdmissionDecision: ...
def admit_candidate(pool: HistoricalPoolManifest, entry: PoolEntry, metrics: AdmissionMetrics) -> HistoricalPoolManifest: ...
```

Admission passes only if integrity and complete paired finite rows are true, script average is no more than `0.10` below active, and either historical worst-case strictly improves or the payoff-vector L1 distance is at least `0.10` from every active non-anchor entry. Until capacity 12, every eligible candidate is admitted. At capacity, protect the M2 anchor and newest admitted entry, and replace a deterministic redundant non-anchor only when the candidate increases the minimum pairwise distance or improves historical worst-case performance; ties use `(environment_steps, policy_id)`.

- [x] Run GREEN and a negative test against a tampered checkpoint.
- [x] Commit: `feat: add immutable historical policy pool`.

### Task 4: Implement deterministic PFSP and paired scheduling

**Files:**

- Create: `src/botcolosseo/training/pfsp.py`
- Create: `src/botcolosseo/training/league_schedule.py`
- Create: `tests/unit/test_pfsp.py`
- Create: `tests/unit/test_league_schedule.py`

- [x] Test `w=max(0.05,(1-p)^2)`, normalization, stable ordering, 40/50/10 source mixture after two historical policies, script/history fallback before that point, side pairing, and invariance to process restart.
- [x] Run RED.
- [x] Implement:

```python
def pfsp_probabilities(win_rates: Mapping[str, float], *, floor: float = 0.05) -> dict[str, float]: ...

def stable_uniform(*, master_seed: int, pair_slot: int, pool_hash: str, payoff_hash: str, stream: str) -> float: ...

class LeagueSchedule:
    def assignments(self, pair_slot: int) -> tuple[LeagueEpisodeAssignment, LeagueEpisodeAssignment]: ...
```

Use SHA-256 over canonical JSON to derive deterministic random bits. A pair slot chooses one opponent and one neutral seed-pair, then yields host and join rows consecutively; it must not resample between sides. Store the actual source probability in every assignment.

- [x] Run GREEN and verify that assignments `0..999` are byte-identical in two fresh processes.
- [x] Commit: `feat: add deterministic PFSP league schedule`.

### Task 5: Integrate learned opponents into the existing rollout path

**Files:**

- Modify: `src/botcolosseo/training/duel_rollout.py`
- Modify: `src/botcolosseo/training/league_schedule.py`
- Create: `src/botcolosseo/training/league_rollout.py`
- Modify: `tests/unit/test_duel_rollout.py`
- Create: `tests/unit/test_league_rollout.py`

- [x] Add regression tests proving the default M2 script path produces the same actions/metadata, then tests proving a checkpoint opponent consumes the opposite side's public observation, keeps hidden state within an episode, resets at the next episode, and never evaluates a privileged-state callback.
- [x] Run both test files RED.
- [x] Add a narrow controller boundary:

```python
class DuelOpponentController(Protocol):
    def reset(self, *, seed: int) -> None: ...
    def act(self, observation: DuelActorObservation, privileged_state: Callable[[], DuelPrivilegedState]) -> MacroAction: ...
```

The script adapter calls `privileged_state()`; the checkpoint adapter must not. Preserve the existing constructor behavior by supplying the script adapter by default. `LeagueRolloutCollector` supplies assignments from `LeagueSchedule` and records `opponent_id`, `opponent_kind`, `source`, `pair_slot`, and probability with each completed episode.

- [x] Run GREEN, then all rollout/curriculum tests.
- [x] Commit: `feat: support learned opponents in duel rollouts`.

### Task 6: Add M2 warm-start and exact league resume contracts

**Files:**

- Create: `src/botcolosseo/training/league_checkpoint.py`
- Create: `tests/unit/test_league_checkpoint.py`

- [x] Test warm-start copies model weights but not optimizer/scheduler state; resume restores all learner/RNG/schedule state; pool/payoff/config/base/scenario hash drift rejects resume; atomic save leaves no partial file.
- [x] Run RED.
- [x] Implement `LeagueRunIdentity`, `warm_start_from_m2`, `save_league_checkpoint`, and `load_league_checkpoint`. The payload must include environment steps, updates, model/optimizer/scheduler state, Python/NumPy/Torch RNG states, next pair slot, run identity, and schema version.
- [x] Run GREEN and a save/load continuation test whose next ten assignments and losses exactly match uninterrupted execution.
- [x] Commit: `feat: add exact M3 checkpoint resume`.

### Task 7: Build the separate league-training CLI

**Files:**

- Create: `configs/m3/league.yaml`
- Create: `src/botcolosseo/cli/train_league.py`
- Create: `scripts/train_league.py`
- Create: `tests/unit/test_train_league_cli.py`
- Create: `tests/integration/test_train_league_smoke.py`

- [x] Test config validation, provisional-base refusal unless `--allow-provisional-base` is explicit, candidate cadence, resume identity, no test-manifest reads, and sorted JSON status output.
- [x] Run RED.
- [x] Implement a separate CLI that reuses the M2 PPO update functions, starts with fresh optimizer/scheduler state, trains for `2_000_000` environment steps, and writes candidates every `200_000` steps. Required arguments: `--config`, `--base-checkpoint`, `--pool`, `--payoffs`, `--run-dir`, `--device`, and optional `--resume`.
- [x] Add a CPU fake-runtime smoke and a real CUDA 2,048-step smoke; neither is a long experiment.
- [x] Run GREEN and confirm no files under `configs/m2/` changed.
- [x] Commit: `feat: add auditable M3 league trainer`.

### Task 8: Implement raw cross-play and validation payoff evidence

**Files:**

- Create: `src/botcolosseo/evaluation/crossplay.py`
- Create: `src/botcolosseo/cli/evaluate_crossplay.py`
- Create: `scripts/evaluate_crossplay.py`
- Create: `tests/unit/test_crossplay.py`
- Create: `tests/integration/test_crossplay_smoke.py`

- [ ] Test ordered matrix cells including diagonals, first five validation seed-pairs, both sides, deterministic row order, public-observation checkpoint policies, script Teacher isolation, and protocol inconsistency propagation.
- [ ] Run RED.
- [ ] Implement `CrossplayRow`, `evaluate_crossplay`, `write_crossplay_csv_atomic`, and `summarize_payoff_matrix`. Raw CSV is authoritative; JSON is a deterministic summary derived from it.
- [ ] Run a two-policy fake matrix and a one-pair real duel smoke. Verify `5*N*(N+1)` executed raw rows for `N` learned policies: each unordered pair including the diagonal uses five seed-pairs and both sides, then deterministically populates both ordered matrix cells without rerunning equivalent matchups.
- [ ] Commit: `feat: add complete M3 cross-play evaluation`.

### Task 9: Implement validation-only candidate selection and pool updates

**Files:**

- Create: `src/botcolosseo/evaluation/strong_base_selection.py`
- Create: `src/botcolosseo/cli/select_strong_base.py`
- Create: `src/botcolosseo/cli/update_historical_pool.py`
- Create: `scripts/select_strong_base.py`
- Create: `scripts/update_historical_pool.py`
- Create: `tests/unit/test_strong_base_selection.py`
- Create: `tests/unit/test_update_historical_pool_cli.py`

- [ ] Test the exact lexicographic order: integrity, historical worst-case, script average, full objective rate, earliest environment step. Test that test/held-out evidence paths are rejected as selection inputs.
- [ ] Run RED.
- [ ] Implement pure selection functions first, then thin CLIs. Pool update must verify every referenced hash and emit an admission decision report even when rejected.
- [ ] Run GREEN with tied synthetic candidates covering every tie-break.
- [ ] Commit: `feat: add validation-only Strong Base selection`.

### Task 10: Implement the frozen M3 evaluator and statistical gate

**Files:**

- Create: `configs/m3/evaluation.yaml`
- Create: `src/botcolosseo/evaluation/m3.py`
- Create: `src/botcolosseo/evaluation/paired_bootstrap.py`
- Create: `tests/unit/test_m3_evaluation.py`
- Create: `tests/unit/test_paired_bootstrap.py`

- [ ] Test official episode counts, category thresholds, zero inconsistency requirement, baseline pairing, deterministic 10,000-resample bootstrap, score-difference lower confidence bound, and fail-closed missing rows.
- [ ] Run RED.
- [ ] Implement the frozen counts: 500 script episodes, 100 no-opponent episodes, 100 held-out episodes, and `pool_size*20*2*2` historical episodes (20 pairs, both sides, Strong Base and M2 baseline). Require script average `>=0.70`, every major script `>=0.55`, no-opponent full objective `>=0.90`, held-out `>=0.80`, historical worst-case strictly above M2, paired bootstrap LCB `>=0`, pool size 8–12, and zero inconsistencies. Report Wilson 95% intervals for rates and actual protocol core-coordinate strata for held-out rows.
- [ ] No-opponent evaluation must use a literal no-op controller; do not substitute an easy script.
- [ ] Run GREEN and snapshot the exact report schema.
- [ ] Commit: `feat: freeze the M3 Strong Base gate`.

### Task 11: Add official-evaluation CLI and artifact audit

**Files:**

- Create: `src/botcolosseo/cli/evaluate_m3.py`
- Create: `scripts/evaluate_m3.py`
- Create: `src/botcolosseo/cli/audit_m3_evidence.py`
- Create: `scripts/audit_m3_evidence.py`
- Create: `tests/unit/test_evaluate_m3_cli.py`
- Create: `tests/unit/test_m3_evidence_audit.py`

- [ ] Test preflight-only mode, manifest/hash binding, atomic resumable row append, exact duplicate suppression, conflicting duplicate rejection, selected-checkpoint binding, split isolation, and audit failure on any missing/tampered artifact.
- [ ] Run RED.
- [ ] Implement `--preflight`, `--resume`, `--output-dir`, `--selected-checkpoint`, `--pool`, `--m2-baseline`, and `--device`. On resume, validate the complete run identity before accepting prior rows. A bounded retry retains the exact seed, side, opponent, policy, and row identity and increments an explicit retry counter.
- [ ] Audit must recompute all hashes/counts/metrics from raw rows rather than trust summary JSON.
- [ ] Run GREEN, then interrupt/resume a fake 20-row run and compare bytes with uninterrupted output.
- [ ] Commit: `feat: add resumable M3 official evaluation`.

### Task 12: Pass short real-runtime and GPU integration gates

**Files:**

- Create: `tests/integration/test_checkpoint_opponent_duel.py`
- Create: `tests/integration/test_m3_cuda_smoke.py`
- Produce locally: `runs/m3/smoke/`

- [ ] Create a tiny deterministic checkpoint fixture at test runtime; do not commit a binary fixture.
- [ ] Run one real side-swapped seed-pair against it and assert synchronized tics, public observation shapes, recurrent reset count, zero protocol inconsistencies, clean workers, and bounded runtime.
- [ ] Run the 2,048-step CUDA trainer smoke on one GPU and verify finite losses, a reloadable candidate, valid hashes, and non-empty opponent-source counts.
- [ ] Run all integration tests that do not consume official splits.
- [ ] Commit: `test: pass M3 runtime and CUDA smoke gates`.

### Task 13: Generate product-facing evidence without changing raw results

**Files:**

- Create: `src/botcolosseo/evaluation/m3_figures.py`
- Create: `src/botcolosseo/cli/render_m3_evidence.py`
- Create: `scripts/render_m3_evidence.py`
- Create: `tests/unit/test_m3_figures.py`
- Create after real results: `reports/m3/strong-base-summary.json`
- Create after real results: `reports/m3/crossplay.csv`
- Create after real results: `reports/m3/crossplay-matrix.json`
- Create after real results: `docs/assets/m3-crossplay-heatmap.png`
- Create after real results: `docs/assets/m3-pfsp-pool-history.png`
- Create after real results: `docs/milestones/m3.md`
- Create after real results: `models/strong-base/MODEL_CARD.md`

- [ ] Test deterministic labels, matrix ordering, missing-cell failure, figure dimensions, and that renderer output metrics equal raw CSV recomputation.
- [ ] Run RED, implement a pure data-to-figure layer, then run GREEN.
- [ ] Do not create claimed final evidence from smoke or synthetic rows. Documentation templates may exist only with an explicit `status: pending` marker until the real gate passes.
- [ ] Commit code now as `feat: add M3 evidence rendering`; commit real evidence only after the official audit passes.

### Task 14: Run the deterministic preflight and full regression gate

**Files:**

- Modify only if necessary: `script.md`

- [ ] Run:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src tests scripts
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit -q
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/integration -q
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pip check
/home/wencong/miniconda3/envs/botcolosseo/bin/python scripts/build_crystal_run.py --check
git diff --check
git status --short
```

- [ ] Run M3 evaluator preflight against provisional artifacts only; it may validate shape/hashes but must not report the M3 gate as passed.
- [ ] Compare `git diff 42e8ae5 -- configs/m2 assets/scenarios` and require no M3-caused changes.
- [ ] Review the implementation against every numbered design requirement and record any deviation before long training.
- [ ] Commit preflight fixes in focused commits; do not squash evidence into code commits.

### Task 15: Hand off the long league experiment and stop

**Files:**

- Modify: `script.md`

- [ ] First confirm the recovered M2 artifact audit has passed. If not, do not authorize meaningful M3 training.
- [ ] Freeze and record hashes for the M2 base, M3 configs, scenario, manifests, initial pool, and payoff store.
- [ ] Add exact commands to `script.md` for: initial pool bootstrap, 2,000,000-step league training, validation cross-play/admission loop, Strong Base selection, official M3 evaluation, artifact audit, and evidence rendering.
- [ ] Commands must use `nohup`, explicit GPU selection, unbuffered Python, PID file, log file, exit-code file, and an immediately runnable progress command. Use GPU 0 for learner and GPU 1 only where the implementation explicitly supports concurrent evaluation; otherwise run phases serially.
- [ ] Add recovery instructions that never delete valid completed rows and that verify run identity before `--resume`.
- [ ] Run every `--preflight` command and the first short smoke command. Do not start the long experiment from Codex.
- [ ] Mark the active goal blocked and tell the user exactly which single command to run first, expected duration range, log/progress commands, success marker, and next automatic command.

Long-run success is not inferred from process exit alone. M3 completes only when `scripts/audit_m3_evidence.py` returns zero, the frozen Strong Base gate reports PASS, all tracked evidence is derived from raw rows, and the selected checkpoint hash matches the model card and pool manifest.
