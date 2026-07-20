from pathlib import Path

import pytest
import vizdoom as vzd

from botcolosseo.scenarios.build import MAPS, BuildSettings, build_crystal_run

SCENARIO_DIR = Path("assets/scenarios/crystal_run").resolve()
ACC_PATH = Path("/home/wencong/.local/bin/acc")
ACC_INCLUDE = Path("/home/wencong/.local/src/acc-1.60")


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_tracked_scenario_matches_clean_rebuild(tmp_path: Path) -> None:
    manifest = build_crystal_run(
        BuildSettings(
            source_dir=SCENARIO_DIR / "src",
            output_wad=tmp_path / "crystal_run.wad",
            manifest_path=tmp_path / "manifest.json",
            acc_path=ACC_PATH,
            acc_include=ACC_INCLUDE,
        )
    )

    assert (tmp_path / "crystal_run.wad").read_bytes() == (
        SCENARIO_DIR / "crystal_run.wad"
    ).read_bytes()
    assert manifest.acc_version.startswith("This is version 1.60")


@pytest.mark.integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize("map_name", MAPS)
def test_each_crystal_run_map_loads_and_exposes_protocol(map_name: str) -> None:
    game = vzd.DoomGame()
    try:
        assert game.load_config(str(SCENARIO_DIR / "crystal_run.cfg"))
        game.set_doom_map(map_name)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_audio_buffer_enabled(False)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_mode(vzd.Mode.PLAYER)
        game.init()
        game.new_episode()

        state = game.get_state()

        assert state is not None
        assert state.screen_buffer.ndim == 2
        assert game.get_game_variable(vzd.GameVariable.USER1) == 1
    finally:
        game.close()
