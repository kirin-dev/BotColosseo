from pathlib import Path


def test_pipeline_finishes_side_swap_inside_one_training_process() -> None:
    script = Path("scripts/run_m3_pipeline.sh").read_text(encoding="utf-8")

    assert "--finish-paired-boundary" in script
    assert "while (( EPISODES % 2 != 0 ))" not in script
