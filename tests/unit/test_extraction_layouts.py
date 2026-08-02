from botcolosseo.envs.extraction_layouts import (
    RANDOMIZED_LAYOUT_COUNT,
    RANDOMIZED_LOOT_ANCHORS,
    randomized_layout_variant,
    randomized_loot_layout,
    strong_randomized_route,
)


def test_all_randomized_layouts_are_deterministic_and_collision_free() -> None:
    layouts = [randomized_loot_layout(variant) for variant in range(128)]

    assert len(layouts) == RANDOMIZED_LAYOUT_COUNT
    assert len(set(layouts)) == RANDOMIZED_LAYOUT_COUNT
    assert all(len({item[1:] for item in layout}) == 7 for layout in layouts)
    assert all(
        tuple(item[0] for item in layout) == (10, 10, 10, 10, 25, 25, 50)
        for layout in layouts
    )
    assert all(item[1:] in RANDOMIZED_LOOT_ANCHORS for layout in layouts for item in layout)


def test_case_seed_selects_variant_without_random_global_state() -> None:
    assert randomized_layout_variant(0) == 0
    assert randomized_layout_variant(127) == 127
    assert randomized_layout_variant(128) == 0


def test_randomized_strong_route_visits_50_25_10_loot() -> None:
    layout = randomized_loot_layout(37)
    expected = {layout[index][1:] for index in (6, 4, 0)}

    assert set(strong_randomized_route(side="host", variant=37)) == expected
    assert set(strong_randomized_route(side="opponent", variant=37)) == expected
