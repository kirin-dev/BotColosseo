from pathlib import Path


def test_pipeline_finishes_side_swap_inside_one_training_process() -> None:
    script = Path("scripts/run_m3_pipeline.sh").read_text(encoding="utf-8")

    assert "--finish-paired-boundary" in script
    assert "while (( EPISODES % 2 != 0 ))" not in script


def test_final_paired_candidate_is_admitted_before_pipeline_leaves_training_loop() -> None:
    script = Path("scripts/run_m3_pipeline.sh").read_text(encoding="utf-8")
    loop = script.split("while (( BOUNDARY <= 2000000 )); do", 1)[1].split(
        "done", 1
    )[0]

    admission = loop.index('scripts/update_historical_pool.py')
    state_update = loop.index('write_state \\\n    "$POOL" "$PAYOFFS" "$BOUNDARY"')
    final_exit = loop.index('if (( FINAL_BOUNDARY )); then')

    assert admission < state_update < final_exit
