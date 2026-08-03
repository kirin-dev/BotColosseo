# Conservative PPO v2 Design

## Objective

Repair Strong PPO so reinforcement-learning updates cannot silently destroy the
aligned BC capability. The first target is capability preservation, not a claim
of PPO improvement. Reward changes are out of scope until the optimizer passes a
10,000-environment-step preservation gate.

The selected starting artifact is
`runs/extraction-randomized/strong-bc-mask-aware-v2/strong/best.pt`, whose
120-episode validation extraction rate is 82.5% and whose Teacher lineage is
`2d6035fb7d5f5d4c264fdeec17479ffc28dca73a20a798c852a613f2ca62ec75`.

## Failure evidence

The rejected PPO run degraded on both closed-loop and offline measurements:

- the aligned BC achieved 92.26% accuracy on the fixed 20,000-transition BC
  validation set;
- PPO accuracy fell to 76.02% at 50k and 66.72% at 200k;
- the first 512-step PPO batch had approximately 25% Teacher agreement and
  Teacher cross-entropy near 9--10, while the PPO policy loss was around
  `1e-4`;
- `TURN_RIGHT` predictions grew from 28.6% for BC to 47.8% at 200k, while
  `MOVE_FORWARD` fell from 17.2% to 5.0%;
- the 50k, 100k, 150k, and 200k candidates all failed the balanced 32-case
  closed-loop screen.

The privileged critic cannot be the direct source of Actor drift because the
Actor features are detached before the value head. The current online Teacher
cross-entropy dominates Actor updates on the policy's narrow, shifted state
distribution. Per-update PPO KL only constrains adjacent policies and does not
bound cumulative drift from the frozen BC. The 16-layout opening curriculum
amplifies forgetting by restricting early online coverage.

## Selected approach

Use a dual-anchor conservative PPO update:

1. Keep an immutable reference Actor loaded from the aligned BC checkpoint.
2. On every PPO sequence minibatch, add categorical KL from the frozen BC
   reference distribution to the current Actor distribution on the same valid
   public observations.
3. On every PPO update, consume a deterministic batch from the aligned BC
   training manifest and add replay cross-entropy against its Teacher actions.
4. Keep online Teacher supervision, but reduce it to a secondary correction
   rather than the dominant capability anchor.
5. Remove the restricted layout curriculum for this diagnostic run: sample all
   128 training layouts from step zero.
6. Reduce the main learning rate from `1e-5` to `2e-6` and use one update epoch
   per rollout instead of four.

The existing task reward remains unchanged so the experiment isolates the
optimizer and anchoring repair.

## Training components

### Frozen BC reference

The trainer receives a deep-frozen copy of the BC Actor. It is excluded from the
optimizer and checkpoint model state. Its checkpoint SHA-256 and Teacher lineage
are recorded in every summary. Reference inference uses only public Actor inputs;
privileged critic features are never passed to it.

The reference KL is
`KL(reference || current)` over valid PPO loss tokens. This direction penalizes
the current policy for removing actions assigned meaningful probability by BC.
The coefficient is configuration-driven and fixed during the 10k diagnostic.

### Offline BC replay

The CLI loads the aligned BC train manifest through the existing
`ExtractionChunkDataset` and `DeterministicBatchStream`. Each PPO optimizer step
receives one deterministic replay batch. Replay loss is ordinary Teacher-action
cross-entropy over valid tokens and updates only the current Actor.

Replay state is reproducible from the run seed and trainer update count. Resume
must restore the update counter before requesting the next replay batch so an
interrupted run consumes the same sequence of batches.

### Online Teacher term

Online Teacher cross-entropy remains available for recovery on policy-visited
states, but its coefficient is reduced from 1.0 to 0.1 for the diagnostic. It
must be reported separately from replay loss and reference KL. The three Actor
terms must never be combined under one ambiguous `teacher_loss` metric.

### PPO and critic

The clipped PPO objective and privileged critic remain unchanged. The Actor
receives policy, online Teacher, replay, and reference-KL gradients. The critic
receives only value gradients because Actor features remain detached from the
value branch.

## Diagnostic configuration

- environment-step horizon: 1,000,000, to preserve resumable scheduler identity;
- first stop: 10,000 environment steps;
- checkpoint interval: 10,000 steps;
- layouts: all 128 from step zero;
- learning rate: `2e-6`;
- final learning rate: `1e-6`;
- update epochs: 1;
- online Teacher coefficient: 0.1;
- replay coefficient: 1.0;
- reference-KL coefficient: 1.0;
- visual encoder: frozen during the 10k diagnostic;
- test cases: prohibited.

These coefficients are conservative defaults for the short preservation test;
they are not presented as final hyperparameters.

## Gates

Before launch, unit tests must prove:

- the frozen reference receives no gradients and is absent from the optimizer;
- reference KL is zero for identical policies and positive after a policy
  perturbation;
- replay loss uses only valid BC tokens;
- trainer checkpoints resume with the same replay-batch sequence;
- missing or mismatched BC Teacher lineage fails before environment creation;
- the existing PPO loss and critic behavior remain unchanged when both new
  coefficients are zero.

At 10k steps, proceed only when all conditions hold:

- offline aligned-BC validation accuracy is at least 90%;
- the balanced 32-case extraction rate is at least 71.875% (23/32), allowing at
  most two fewer extractions than the 25/32 BC baseline;
- no single predicted action exceeds 40% on the offline validation set;
- protocol inconsistencies and Actor privilege violations equal zero;
- no test case was accessed.

If the gate passes, resume to 20k and repeat it. A 50k run is authorized only
after both 10k and 20k pass. If the gate fails, retain the aligned BC and diagnose
the individual loss metrics; do not compensate by increasing training steps.

## Artifacts and isolation

The repair uses new paths and never overwrites rejected PPO evidence:

- config: `configs/extraction/randomized/aligned-v2/strong-ppo-conservative.yaml`;
- run: `runs/extraction-randomized/strong-ppo-conservative-v2/`;
- reports: `reports/extraction/mask-aware-bc-alignment/ppo-conservative-v2/`.

The rejected `strong-ppo-mask-aware-v2` checkpoints remain local diagnostic
artifacts and are never eligible for the public showcase.
