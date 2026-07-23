# M5 V2 Teacher-Assisted Style Repair Implementation

**Design:** `docs/superpowers/specs/2026-07-23-m5-v2-teacher-assisted-style-repair-design.md`

**Scope:** Implement the approved V2 mechanism without changing the Strong
Base, public observation, train/validation/test splits, frozen style
evaluators, or V1 artifacts.

## 1. Rollout supervision contract

Files:

- `src/botcolosseo/training/rollout.py`
- `src/botcolosseo/training/ppo.py`
- `tests/unit/test_rollout.py`

Add optional recurrent token tensors for `teacher_actions`, `teacher_mask`,
and `route_modes`. Existing collectors leave all three absent. V2 collection
must provide all three together. Sequence collation must preserve the tensors,
pad masks as false, and pad route modes with `-1`.

Verification:

- an existing PPO rollout produces the unchanged `PPOBatch`;
- a supervised rollout preserves labels and modes through recurrent chunking;
- partial supervision and padded tokens cannot enter the auxiliary loss.

## 2. V2 style models and public route controller

Files:

- `src/botcolosseo/agents/style_model.py`
- `src/botcolosseo/agents/league_opponents.py`
- `tests/unit/test_style_model.py`
- `tests/unit/test_league_opponents.py`

Keep `StyledActorCritic` unchanged for Defensive and V1 compatibility. Add an
Explorer model with three independent residual adapter/policy branches over
one frozen Base actor. Its forward call requires a route-mode tensor and
selects the corresponding branch per token.

The public checkpoint wrapper owns an episode counter and the score at reset:

```text
initial_mode = episode_counter mod 3
active_mode = (initial_mode + own_score - reset_score) mod 3
```

It derives the mode from public `DuelActorObservation` fields before calling
the stateless routed Actor. Checkpoint loading detects and strictly validates
all three branch tensors.

Verification:

- episode starts are `0, 1, 2, 0`;
- own-score progress advances the mode;
- seed, case identity, coordinates, regions, and privileged state are absent
  from the controller interface;
- branches are independent while the Base actor remains frozen;
- missing or malformed branch state fails strict checkpoint loading.

## 3. Training-only supervision

Files:

- `src/botcolosseo/agents/duel_teachers.py`
- `src/botcolosseo/training/style_supervision.py`
- `src/botcolosseo/training/duel_rollout.py`
- `src/botcolosseo/training/league_rollout.py`
- `tests/unit/test_duel_teachers.py`
- `tests/unit/test_style_supervision.py`
- `tests/unit/test_duel_rollout.py`

Add an explicit route-mode action entry point to `RouteExplorerTeacher`.
Create two training-only supervisors:

- Defensive returns the `ProtectiveDefensiveTeacher` action and masks it with
  the frozen `defensive_risk` predicate;
- Explorer derives the approved mode from episode index and public own-score
  progress, then asks the route Teacher for that mode's label.

The collector executes only the sampled policy action. Teacher labels, masks,
and modes are stored after the policy output is computed. The next-value call
uses only the next public score to select an Explorer branch.

Verification:

- Defensive labels are masked outside risk states;
- Explorer preflight episodes expose all modes;
- changing a Teacher label does not change the executed rollout action;
- base/V1 collectors remain byte-contract compatible.

## 4. Masked auxiliary PPO

Files:

- `src/botcolosseo/training/style_ppo.py`
- `tests/unit/test_style_ppo.py`

Add a strict masked categorical cross-entropy primitive and extend
`StylePPOTrainer` with `eta_aux`. The trainer uses:

```text
L = L_PPO + beta_kl * KL(style || base) + eta_aux * CE_teacher
```

For Explorer, the routed forward call receives the stored mode. A complete
rollout with zero supervised tokens is rejected before updates; individual
recurrent minibatches with zero selected supervised tokens contribute a
differentiable zero auxiliary term and report zero tokens.

Verification:

- the primitive rejects an empty mask;
- unsupervised labels cannot change the loss;
- supervised-token count and Teacher agreement are exact;
- finite loss and gradients hold for both style architectures.

## 5. Schema V2, identity, metrics, and checkpoints

Files:

- `src/botcolosseo/cli/train_league.py`
- `src/botcolosseo/training/league_checkpoint.py`
- `configs/m5/defensive_ppo_v2.yaml`
- `configs/m5/explorer_ppo_v2.yaml`
- relevant CLI and checkpoint unit tests

Accept schema V2 only for Defensive/Explorer. Bind architecture, branch order,
Teacher name, auxiliary coefficient, route rule, and warm-start hash through
the existing config/run identity hashes. V1 schema remains accepted.

V2 summaries and JSONL metrics add:

- total and per-mode supervised-token counts;
- auxiliary loss and Teacher agreement;
- per-mode Explorer reward evidence;
- `test_cases_accessed: false`;
- V2 architecture and route-rule identifiers.

The Explorer warm start duplicates the audited single-branch V1 adapter/head
into all three branches before on-policy specialization. Resume remains
strictly hash-bound and never crosses a V1/V2 identity.

Verification:

- schema V1 behavior remains accepted;
- incomplete V2 configuration fails;
- V2 resume rejects config, Teacher, route-rule, or branch drift;
- a saved Explorer candidate loads through the public-only opponent policy.

## 6. Bounded pipeline and evidence audit

Files:

- `scripts/run_m5_defensive_ppo_v2.sh`
- `scripts/run_m5_explorer_ppo_v2.sh`
- `scripts/audit_m5_v2_training.py`
- `script.md`
- audit unit tests

Each isolated run uses:

1. 2,000-step real CUDA preflight;
2. 50,000-step immutable pilot;
3. existing 20-episode validation smoke;
4. at most one owner-approved continuation to 100,000 steps when the frozen
   continuation rule is satisfied;
5. unchanged formal and difficulty evaluators after passing prerequisites.

The audit fails on non-finite metrics, empty Defensive supervision, a missing
Explorer mode, missing checkpoint branches, test access, identity mismatch,
or overwritten evidence.

Verification:

- shell syntax and CLI `--help` checks;
- focused unit suite, full unit suite, and Ruff;
- real 2k ViZDoom/CUDA run on each physical GPU;
- load and one public inference step from each resulting checkpoint.

## 7. Execution and publication order

1. Commit implementation and unit evidence.
2. Run both 2k preflights and commit their immutable reports.
3. Launch Defensive 50k on one A100 and Explorer 50k on the other.
4. Monitor each process at launch and after the first completed rollout.
5. Run the frozen 20-episode smokes on each 50k candidate.
6. Apply the approved stop/continue rule independently per style.
7. Preserve failures; promote only candidates that pass the frozen gates.

