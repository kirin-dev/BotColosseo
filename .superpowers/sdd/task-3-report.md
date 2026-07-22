# Task 3 Report: Record Auditable Learner Episodes

## Scope

Implemented only Task 3 in `src/botcolosseo/demo/showcase.py` and
`tests/unit/test_showcase_demo.py`.

- Added `ShowcaseEvent` and `RecordedShowcaseEpisode`, whose serialized record
  deliberately excludes in-memory frames.
- Added `CheckpointEvaluationPolicy`, which forwards only the public actor
  observation to its wrapped checkpoint policy.
- Added the single-attempt synchronous episode recorder with learner-only
  frames/events and protocol integrity fields.

## RED

Command:

```bash
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_showcase_demo.py -q
```

Output after record/adapter tests were added:

```text
ImportError: cannot import name 'CheckpointEvaluationPolicy' from 'botcolosseo.demo.showcase'
1 error in 0.27s
```

Output after the capture test was added:

```text
ImportError: cannot import name 'record_showcase_episode' from 'botcolosseo.demo.showcase'
1 error in 1.74s
```

## GREEN and refactor verification

Focused test command:

```bash
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_showcase_demo.py -q
```

Output:

```text
......                                                                   [100%]
6 passed in 1.73s
```

Ruff command:

```bash
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src/botcolosseo/demo/showcase.py tests/unit/test_showcase_demo.py
```

Output:

```text
All checks passed!
```

Full unit-suite command:

```bash
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit -q
```

Output:

```text
304 passed in 12.43s
```

`git diff --check` also completed with no output.

## Post-commit verification

After committing `7ab23ec feat: record auditable showcase episodes`, the same
focused test, Ruff command, and full unit-suite command were rerun against the
committed tree. They reported `6 passed in 1.66s`, `All checks passed!`, and
`304 passed in 12.51s`, respectively. `git status --short` then contained only
the intentionally untracked `.superpowers/` task-material directory.

## Self-review

- Observation fairness: `CheckpointEvaluationPolicy.act` deletes the supplied
  state and calls the wrapped policy with `DuelActorObservation` only. The
  test verifies reset/act calls without privileged-state access.
- Runtime reuse: the recorder directly instantiates only the existing
  `SynchronousDuelEnv` when no injected environment is supplied; it creates no
  model loader or alternate duel session.
- Action/tic consistency: every completed step contributes the existing
  `valid_action_tic_boundary` check; a decision-count overrun raises.
- Score/event consistency: SCORE event counts for both sides are compared with
  reset-to-final score deltas.
- Learner evidence: capture stores only learner-side PICKUP, VALID_HIT, DROP,
  and SCORE events; each normal decision appends one learner overlay frame.
- Retry/provenance: no retry wrapper exists, `environment_attempts` is exactly
  one, and `scenario_hash` comes from reset information.
- Cleanup: the environment is closed from `finally` after success or any
  exception raised after environment creation. The two-step fake-environment
  capture test confirms cleanup on the normal path.
- Record schema: `to_record` includes case/case_id, events, scores, terminal
  and integrity fields, attempt count, and scenario hash; it excludes frames.

## Concerns

None. The recorder's `protocol_inconsistent` remains `False`, matching the
existing M2 single-episode contract; the requested per-step auditable protocol
checks are represented by the action-tic, score-event, peer-lag, case, and
scenario fields.

## Review fixes

Addressed all Critical/Important Task 3 review findings in
`src/botcolosseo/demo/showcase.py` and `tests/unit/test_showcase_demo.py`.

- `protocol_inconsistent` is now derived from nonzero peer-tic lag, invalid
  action-tic boundary, or score/event mismatch. Parameterized regression cases
  cover each source, and the normal capture case asserts it remains false.
- `TeacherEvaluationPolicy` construction and all following recorder setup now
  occur inside the environment's `try`/`finally`. A focused construction-failure
  test proves `close()` is called.
- The learner receives `None` as its state argument; only the teacher opponent
  receives `environment.teacher_state()`. The regression test uses a unique
  privileged object and fails if it reaches the learner.
- The normal two-decision capture test asserts `len(frames) == decisions`.

RED evidence before the implementation:

```text
5 failed, 6 passed in 1.79s
```

Fresh verification after the fix:

```text
PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_showcase_demo.py -q
11 passed in 1.76s

PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit/test_showcase_demo.py -q -k 'marks_protocol_inconsistencies or closes_environment_after_opponent_setup_failure or never_passes_privileged_state_to_learner'
5 passed, 6 deselected in 1.75s

PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src/botcolosseo/demo/showcase.py tests/unit/test_showcase_demo.py
All checks passed!

PYTHONPATH=src /home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit -q
309 passed in 12.83s
```

Self-review: the changes are confined to recorder integrity/cleanup/data-flow
behavior and its focused tests; no model loader, runtime architecture, or
public checkpoint adapter API was added or changed. `git diff --check` is clean.
