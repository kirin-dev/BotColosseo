# M3 Failed-Result Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a complete, integrity-clean M3 capability FAIL independently auditable, reproducibly renderable, and honestly publishable while preserving the frozen FAIL decision and nonzero pipeline exit.

**Architecture:** Keep one evidence recomputation path and split its result into integrity status and capability status. The strict audit remains the default; an explicit integrity-only mode permits rendering a hash-consistent official FAIL. The serial driver captures the evaluator's capability status, performs closure work, then returns the original nonzero status.

**Tech Stack:** Python 3.10, PyTest, Bash, JSON/JSONL, Matplotlib, Ruff, Git/GitHub.

## Global Constraints

- Do not change M3 training, checkpoint selection, test rows, manifests, thresholds, or gate definitions.
- Preserve `summary.passed == false` and final pipeline `exit=1` for the observed result.
- Integrity failures always raise and must never render.
- Reuse the existing 1,340-row official ledger; do not launch ViZDoom, CUDA evaluation, training, or a nohup process.
- Keep `audit_m3_evidence(..., require_capability_pass=True)` strict by default.
- Public text must distinguish engineering/evidence completion from capability success.
- Add only compact, necessary M3 evidence; do not track checkpoints or validation matrices.
- Run Python as `/home/wencong/miniconda3/envs/botcolosseo/bin/python` with `PYTHONPATH="$PWD/src"`.

---

### Task 1: Separate integrity audit from capability status

**Files:**

- Modify: `src/botcolosseo/evaluation/m3_evidence_audit.py`
- Modify: `src/botcolosseo/cli/audit_m3_evidence.py`
- Test: `tests/unit/test_m3_evidence_audit.py`

**Interfaces:**

- Consumes: existing official `run-identity.json`, `episodes.jsonl`, `summary.json`, and `manifest.json`.
- Produces: `audit_m3_evidence(report_dir, *, artifact_root=None, require_capability_pass=True) -> dict[str, object]` and CLI flag `--integrity-only`.

- [ ] **Step 1: Extend the test summary with explicit gate and integrity fields**

Import `field` with `dataclass`, then update `StubSummary` so its `to_dict()`
matches the status fields used by the auditor:

```python
@dataclass(frozen=True)
class StubSummary:
    episodes: int = 1
    expected_episodes: int = 1
    official: bool = True
    complete: bool = True
    passed: bool = True
    pool_size: int = 8
    protocol_inconsistencies: int = 0
    artifact_inconsistencies: int = 0
    gates: dict[str, bool] = field(
        default_factory=lambda: {
            "official": True,
            "complete": True,
            "pool_size": True,
            "protocol_clean": True,
            "artifact_clean": True,
            "heldout_core_strata_complete": True,
            "confidence_intervals_finite": True,
            "historical_worst_case_improved": True,
        }
    )
```

Include those fields in `to_dict()`.

- [ ] **Step 2: Write failing strict-versus-integrity tests**

Add tests that use one hash-consistent recomputed FAIL:

```python
def test_integrity_only_accepts_hash_consistent_capability_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_dir, _ = _evidence(tmp_path)
    failed = StubSummary(
        passed=False,
        gates={
            "official": True,
            "complete": True,
            "pool_size": True,
            "protocol_clean": True,
            "artifact_clean": True,
            "heldout_core_strata_complete": True,
            "confidence_intervals_finite": True,
            "historical_worst_case_improved": False,
        },
    )
    identity = json.loads(
        (report_dir / "run-identity.json").read_text(encoding="utf-8")
    )
    write_m3_evidence(report_dir, summary=failed, run_identity=identity)
    monkeypatch.setattr(
        "botcolosseo.evaluation.m3_evidence_audit.evaluate_m3_records",
        lambda rows, **kwargs: failed,
    )

    with pytest.raises(ValueError, match="did not pass"):
        audit_m3_evidence(report_dir)

    result = audit_m3_evidence(report_dir, require_capability_pass=False)
    assert result == {
        "episodes": 1,
        "official": True,
        "integrity_passed": True,
        "capability_passed": False,
        "passed": False,
        "failed_gates": ["historical_worst_case_improved"],
        "pool_size": 8,
        "selected_checkpoint_sha256": "a" * 64,
    }
```

Add a second test where `protocol_clean` is false and assert integrity-only mode still raises `ValueError` containing `integrity`.

- [ ] **Step 3: Run RED**

Run:

```bash
export PYTHONPATH="$PWD/src"
/home/wencong/miniconda3/envs/botcolosseo/bin/python \
  -m pytest tests/unit/test_m3_evidence_audit.py -q
```

Expected: FAIL because `require_capability_pass` is not accepted.

- [ ] **Step 4: Implement one audit path with two outcomes**

Add these constants and signature:

```python
_INTEGRITY_GATES = {
    "official",
    "complete",
    "pool_size",
    "protocol_clean",
    "artifact_clean",
    "heldout_core_strata_complete",
    "confidence_intervals_finite",
}

def audit_m3_evidence(
    report_dir: Path,
    *,
    artifact_root: Path | None = None,
    require_capability_pass: bool = True,
) -> dict[str, object]:
```

After exact raw-row recomputation and before repository bindings, validate the mode type, require every integrity gate, retain the strict capability check when requested, and return:

```python
if not isinstance(require_capability_pass, bool):
    raise ValueError("M3 capability-pass requirement must be boolean")
failed_gates = sorted(name for name, passed in recomputed.gates.items() if not passed)
failed_integrity = sorted(_INTEGRITY_GATES.intersection(failed_gates))
if failed_integrity:
    raise ValueError(f"Recomputed M3 evidence failed integrity gates: {failed_integrity}")
if require_capability_pass and not recomputed.passed:
    raise ValueError("Recomputed M3 evidence did not pass")
```

Return `integrity_passed`, `capability_passed`, `passed`, and `failed_gates` in addition to the existing identity fields. Do not duplicate recomputation or repository-binding logic.

- [ ] **Step 5: Add the CLI flag**

In `src/botcolosseo/cli/audit_m3_evidence.py` add:

```python
parser.add_argument(
    "--integrity-only",
    action="store_true",
    help="accept a complete integrity-clean result whose capability gate failed",
)
```

Pass `require_capability_pass=not args.integrity_only` to the audit function.

- [ ] **Step 6: Run GREEN and CLI regression checks**

Run:

```bash
export PYTHONPATH="$PWD/src"
/home/wencong/miniconda3/envs/botcolosseo/bin/python \
  -m pytest tests/unit/test_m3_evidence_audit.py -q
/home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/audit_m3_evidence.py --help | rg -- '--integrity-only'
```

Expected: tests PASS and help contains `--integrity-only`.

- [ ] **Step 7: Commit**

```bash
git add \
  src/botcolosseo/evaluation/m3_evidence_audit.py \
  src/botcolosseo/cli/audit_m3_evidence.py \
  tests/unit/test_m3_evidence_audit.py
git commit -m "feat: audit integrity-clean M3 failures"
```

### Task 2: Close the serial pipeline after either capability outcome

**Files:**

- Modify: `src/botcolosseo/cli/render_m3_evidence.py`
- Modify: `scripts/run_m3_pipeline.sh`
- Modify: `tests/unit/test_m3_pipeline_script.py`
- Create: `tests/unit/test_render_m3_evidence_cli.py`

**Interfaces:**

- Consumes: Task 1's `require_capability_pass=False` audit mode.
- Produces: an integrity-gated renderer and two literal terminal markers: `M3 PIPELINE PASS` or `M3 PIPELINE COMPLETE: CAPABILITY FAIL`.

- [ ] **Step 1: Write a failing renderer orchestration test**

Create a test that monkeypatches both dependencies and calls `main` with temporary paths:

```python
def test_renderer_accepts_only_integrity_audited_capability_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = []
    monkeypatch.setattr(
        render_cli,
        "audit_m3_evidence",
        lambda path, **kwargs: calls.append((path, kwargs))
        or {"integrity_passed": True, "capability_passed": False, "passed": False},
    )
    monkeypatch.setattr(
        render_cli,
        "render_evidence_bundle",
        lambda **kwargs: {"executed_rows": 360},
    )

    argv = [
        "--official-report-dir",
        str(tmp_path / "official"),
        "--crossplay-csv",
        str(tmp_path / "crossplay.csv"),
        "--pool-history",
        str(tmp_path / "pool-history.json"),
        "--heatmap-output",
        str(tmp_path / "heatmap.png"),
        "--pool-output",
        str(tmp_path / "pool.png"),
        "--matrix-output",
        str(tmp_path / "matrix.json"),
    ]
    assert render_cli.main(argv) == 0
    assert calls[0][1]["require_capability_pass"] is False
    assert '"capability_passed": false' in capsys.readouterr().out
```

Use real argument strings for `--official-report-dir`, `--crossplay-csv`, `--pool-history`, `--heatmap-output`, `--pool-output`, and `--matrix-output`.

- [ ] **Step 2: Write a failing shell-order test**

Extend `tests/unit/test_m3_pipeline_script.py`:

```python
def test_capability_failure_is_packaged_before_pipeline_returns_nonzero() -> None:
    script = Path("scripts/run_m3_pipeline.sh").read_text(encoding="utf-8")
    official = script.split('OFFICIAL="$REPORT_ROOT/official"', 1)[1]

    capture = official.index("EVALUATION_STATUS=$?")
    audit = official.index("--integrity-only")
    render = official.index("scripts/render_m3_evidence.py")
    failed_marker = official.index('M3 PIPELINE COMPLETE: CAPABILITY FAIL')
    final_exit = official.index('exit "$EVALUATION_STATUS"')

    assert capture < audit < render < failed_marker < final_exit
    assert official.index('M3 PIPELINE PASS') > render
```

- [ ] **Step 3: Run RED**

Run:

```bash
export PYTHONPATH="$PWD/src"
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest \
  tests/unit/test_render_m3_evidence_cli.py \
  tests/unit/test_m3_pipeline_script.py -q
```

Expected: renderer audit-mode assertion and shell-order test FAIL.

- [ ] **Step 4: Make renderer request integrity-only audit**

Change the existing call only:

```python
audit = audit_m3_evidence(
    resolve(args.official_report_dir),
    artifact_root=root,
    require_capability_pass=False,
)
```

Retain the current JSON output containing both `audit` and `matrix`.

- [ ] **Step 5: Capture evaluator status and close deterministically**

Wrap only the final evaluator call:

```bash
set +e
"$PYTHON" -u scripts/evaluate_m3.py \
  --selection-report "$SELECTION" \
  --selected-checkpoint "$SELECTED_CHECKPOINT" \
  --pool "$POOL" \
  --m2-baseline "$BASE_CHECKPOINT" \
  --output-dir "$OFFICIAL" \
  --device cuda:0 \
  "${OFFICIAL_ARGS[@]}"
EVALUATION_STATUS=$?
set -e

"$PYTHON" scripts/audit_m3_evidence.py \
  --report-dir "$OFFICIAL" \
  --integrity-only
"$PYTHON" scripts/render_m3_evidence.py \
  --official-report-dir "$OFFICIAL" \
  --crossplay-csv "$REPORT_ROOT/crossplay.csv" \
  --pool-history "$REPORT_ROOT/pool-history.json"

if (( EVALUATION_STATUS == 0 )); then
  echo "M3 PIPELINE PASS"
else
  echo "M3 PIPELINE COMPLETE: CAPABILITY FAIL"
fi
exit "$EVALUATION_STATUS"
```

Do not disable `set -e` around auditing or rendering.

- [ ] **Step 6: Run GREEN and shell validation**

Run:

```bash
export PYTHONPATH="$PWD/src"
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest \
  tests/unit/test_render_m3_evidence_cli.py \
  tests/unit/test_m3_pipeline_script.py -q
bash -n scripts/run_m3_pipeline.sh
```

Expected: all tests PASS and Bash syntax returns zero.

- [ ] **Step 7: Commit**

```bash
git add \
  src/botcolosseo/cli/render_m3_evidence.py \
  scripts/run_m3_pipeline.sh \
  tests/unit/test_render_m3_evidence_cli.py \
  tests/unit/test_m3_pipeline_script.py
git commit -m "fix: package completed M3 capability failures"
```

### Task 3: Publish the official M3 FAIL honestly

**Files:**

- Create: `docs/milestones/m3.md`
- Modify: `README.md`
- Modify: `Plan.md`
- Modify: `tests/unit/test_public_docs.py`
- Generate: `docs/assets/m3-crossplay-heatmap.png`
- Generate: `docs/assets/m3-pfsp-pool-history.png`
- Generate: `reports/m3/crossplay-matrix.json`
- Track: `reports/m3/official/episodes.jsonl`
- Track: `reports/m3/official/manifest.json`
- Track: `reports/m3/official/run-identity.json`
- Track: `reports/m3/official/summary.json`
- Track: `reports/m3/strong-base-selection.json`
- Track: `reports/m3/final-crossplay/{crossplay.csv,manifest.json,matrix.json}`
- Track: `reports/m3/crossplay.csv`
- Track: `reports/m3/pool-history.json`
- Track: `reports/m3/pools/pool-v7.json`
- Track: `reports/m3/pools/payoffs-v7.json`

**Interfaces:**

- Consumes: Task 2's integrity-capable renderer and the existing completed official ledger.
- Produces: a public, hash-bound, recomputable M3 FAIL record and two generated figures.

- [ ] **Step 1: Add failing public-truth tests**

Add `docs/milestones/m3.md` to `PUBLIC_DOCS` and assert:

```python
def test_public_docs_report_m3_as_complete_but_not_passed() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    milestone = Path("docs/milestones/m3.md").read_text(encoding="utf-8")

    assert "M3 capability gate did **not** pass" in readme
    assert "M3: FAIL" in milestone
    assert "historical_worst_case_improved" in milestone
    assert "12.5%" in milestone
    assert "15.0%" in milestone
    assert "M3 passed" not in readme
    assert "M3 PASS" not in milestone
```

- [ ] **Step 2: Run RED**

Run:

```bash
export PYTHONPATH="$PWD/src"
/home/wencong/miniconda3/envs/botcolosseo/bin/python \
  -m pytest tests/unit/test_public_docs.py -q
```

Expected: FAIL because `docs/milestones/m3.md` does not exist.

- [ ] **Step 3: Audit the real FAIL evidence without changing it**

Run:

```bash
export PYTHONPATH="$PWD/src"
/home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/audit_m3_evidence.py \
  --report-dir reports/m3/official \
  --integrity-only | tee /tmp/m3-integrity-audit.json
```

Expected JSON includes:

```json
{
  "integrity_passed": true,
  "capability_passed": false,
  "passed": false,
  "failed_gates": ["historical_worst_case_improved"],
  "episodes": 1340,
  "pool_size": 8
}
```

- [ ] **Step 4: Render deterministic real evidence**

Run twice and compare hashes:

```bash
/home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/render_m3_evidence.py \
  --official-report-dir reports/m3/official \
  --crossplay-csv reports/m3/crossplay.csv \
  --pool-history reports/m3/pool-history.json
sha256sum \
  docs/assets/m3-crossplay-heatmap.png \
  docs/assets/m3-pfsp-pool-history.png \
  reports/m3/crossplay-matrix.json > /tmp/m3-render-first.sha256
/home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/render_m3_evidence.py \
  --official-report-dir reports/m3/official \
  --crossplay-csv reports/m3/crossplay.csv \
  --pool-history reports/m3/pool-history.json
sha256sum --check /tmp/m3-render-first.sha256
```

Expected: all three hashes verify.

- [ ] **Step 5: Write the milestone record from exact evidence**

Create `docs/milestones/m3.md` with:

- selected validation checkpoint `policy-0200000` and its hash;
- 2,000,000 total league-training environment steps;
- pool size 8 and final cross-play size 360;
- official episode count 1,340;
- protocol/artifact inconsistencies both 0;
- script win rate 90.2%, no-op objective rate 100%, held-out objective rate 99%;
- paired score-difference LCB `+0.10`;
- Strong Base historical worst-case win rate 12.5% versus M2 baseline 15.0%;
- explicit `M3: FAIL` and a note that M4 may reuse engineering artifacts but cannot claim a passed Strong Base.

Link the raw ledger, summary, selection report, cross-play matrix, and figures.

- [ ] **Step 6: Update Plan and README truthfully**

In `Plan.md`, add an M3 status paragraph separating engineering completion from the failed capability gate. In `README.md`, replace “Strong Base result remains pending” with the official result and include the two figures plus a compact gate table. Do not change M4/M5 status.

- [ ] **Step 7: Run GREEN and link checks**

Run:

```bash
export PYTHONPATH="$PWD/src"
/home/wencong/miniconda3/envs/botcolosseo/bin/python \
  -m pytest tests/unit/test_public_docs.py -q
```

Expected: all public documentation tests PASS.

- [ ] **Step 8: Stage only deliberate evidence and commit**

Run explicit staging; never use `git add reports/m3`:

```bash
git add \
  README.md Plan.md docs/milestones/m3.md tests/unit/test_public_docs.py \
  docs/assets/m3-crossplay-heatmap.png \
  docs/assets/m3-pfsp-pool-history.png \
  reports/m3/crossplay.csv \
  reports/m3/crossplay-matrix.json \
  reports/m3/pool-history.json \
  reports/m3/strong-base-selection.json \
  reports/m3/final-crossplay/crossplay.csv \
  reports/m3/final-crossplay/manifest.json \
  reports/m3/final-crossplay/matrix.json \
  reports/m3/official/episodes.jsonl \
  reports/m3/official/manifest.json \
  reports/m3/official/run-identity.json \
  reports/m3/official/summary.json \
  reports/m3/pools/pool-v7.json \
  reports/m3/pools/payoffs-v7.json
git diff --cached --check
git commit -m "docs: publish the official M3 result honestly"
```

### Task 4: Verify and publish the M3 closure branch

**Files:**

- Verify all files changed since `origin/feat/m3-strong-base`.
- Do not modify generated checkpoints, validation matrices, or M2 validation-anchor artifacts.

**Interfaces:**

- Consumes: Tasks 1-3 and the real official evidence.
- Produces: a pushed M3 branch and Draft PR with evidence-backed claims only.

- [ ] **Step 1: Run the fresh full verification gate**

```bash
set -o pipefail
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=''
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m pytest tests/unit -q
/home/wencong/miniconda3/envs/botcolosseo/bin/python -m ruff check src scripts tests
bash -n scripts/run_m3_pipeline.sh
/home/wencong/miniconda3/envs/botcolosseo/bin/python \
  scripts/audit_m3_evidence.py \
  --report-dir reports/m3/official \
  --integrity-only
git diff --check
```

Expected: unit suite and Ruff PASS; integrity audit reports `integrity_passed: true`, `capability_passed: false`; shell and diff checks return zero.

- [ ] **Step 2: Verify public evidence boundaries**

```bash
test "$(wc -l < reports/m3/official/episodes.jsonl)" -eq 1340
jq -e '.complete == true and .passed == false and .protocol_inconsistencies == 0 and .artifact_inconsistencies == 0' \
  reports/m3/official/summary.json
jq -e '.gates.historical_worst_case_improved == false' reports/m3/official/summary.json
test "$(jq '.entries | length' reports/m3/pools/pool-v7.json)" -eq 8
test -z "$(git ls-files reports/m3/validation runs/m3)"
git status --short
```

Expected: all assertions return zero; status shows only intentionally untracked local experiment directories.

- [ ] **Step 3: Push and create a Draft PR**

```bash
git push -u origin feat/m3-strong-base
gh pr create \
  --draft \
  --base main \
  --head feat/m3-strong-base \
  --title "Close M3 with auditable official evidence" \
  --body-file .superpowers/sdd/m3-closure-pr-body.md
```

The PR body must state `M3 FAIL`, list the single failed gate, cite 1,340 complete rows and zero inconsistencies, and explain that no threshold or test-driven checkpoint change was made.

- [ ] **Step 4: Verify GitHub state**

```bash
gh pr view --json url,isDraft,state,mergeable,statusCheckRollup
```

Expected: Draft PR is open; report the actual CI state without claiming success before completion.

---

## Completion Criteria

- Strict audit still exits nonzero for the official capability FAIL.
- Integrity-only audit recomputes all 1,340 rows and returns integrity PASS plus capability FAIL.
- The pipeline renders evidence before returning the evaluator's nonzero status.
- README, Plan, milestone record, figures, and tracked reports agree on the exact official result.
- Full unit/Ruff/shell/public-link checks pass.
- No long experiment, test-row reuse, checkpoint replacement, threshold change, or generated validation directory is committed.
