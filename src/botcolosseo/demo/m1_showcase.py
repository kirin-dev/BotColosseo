from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

from botcolosseo.agents.teachers import create_teacher
from botcolosseo.envs.single_agent import SingleAgentTaskEnv
from botcolosseo.envs.video import normalize_rgb_frame, write_mp4
from botcolosseo.evaluation.m1 import M1_TEACHERS, load_cases
from botcolosseo.scenarios.regions import RegionGraph
from botcolosseo.scenarios.splits import EpisodeCase, TaskKind

_SHOWCASE_EVENT = {
    TaskKind.NAVIGATION: "task_success",
    TaskKind.PICKUP: "pickup",
    TaskKind.RETURN: "score",
    TaskKind.STATIC_HIT: "valid_hit",
    TaskKind.MOVING_HIT: "valid_hit",
}


def load_m1_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("official") is not True or payload.get("passed") is not True:
        raise ValueError("M1 showcase requires official passing evidence")
    return payload


def arena_labels(graph: RegionGraph) -> dict[str, set[str]]:
    return {
        "regions": {region.name for region in graph.regions},
        "routes": {route.name for route in graph.routes},
    }


def select_frame_indices(frame_count: int, *, cap: int = 40) -> tuple[int, ...]:
    if frame_count < 0 or cap <= 0:
        raise ValueError("frame_count must be nonnegative and cap must be positive")
    if frame_count <= cap:
        return tuple(range(frame_count))
    return tuple(int(index) for index in np.linspace(0, frame_count - 1, cap))


def overlay_text(*, task: str, teacher: str, event: str, success: bool) -> str:
    return f"task={task} | teacher={teacher} | event={event} | success={success}"


def render_arena(graph: RegionGraph, output_path: Path) -> Path:
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    colors = (
        "#6baed6",
        "#9ecae1",
        "#c6dbef",
        "#74c476",
        "#fdae6b",
        "#bcbddc",
        "#9e9ac8",
        "#fb6a4a",
    )
    figure, axis = plt.subplots(figsize=(12, 7))
    for region, color in zip(graph.regions, colors, strict=True):
        bounds = region.bounds
        axis.add_patch(
            Rectangle(
                (bounds.min_x, bounds.min_y),
                bounds.max_x - bounds.min_x,
                bounds.max_y - bounds.min_y,
                facecolor=color,
                edgecolor="#263238",
                alpha=0.42,
                linewidth=1.5,
            )
        )
        axis.text(
            (bounds.min_x + bounds.max_x) / 2,
            (bounds.min_y + bounds.max_y) / 2,
            region.name.replace("_", "\n"),
            ha="center",
            va="center",
            fontsize=9,
        )
    route_styles = (("#08519c", "-"), ("#238b45", "--"), ("#7a0177", ":"))
    for route, (color, style) in zip(graph.routes, route_styles, strict=True):
        points = np.asarray(route.waypoints)
        axis.plot(
            points[:, 0], points[:, 1], style, color=color, linewidth=2.5, label=route.name
        )
    axis.scatter([-640, 0, 384], [0, 0, 0], marker="*", s=180, color="#111111")
    axis.annotate("home", (-640, 0), xytext=(-690, 55))
    axis.annotate("core", (0, 0), xytext=(-35, 55))
    axis.annotate("target", (384, 0), xytext=(340, 55))
    axis.set(
        xlim=(graph.arena.min_x, graph.arena.max_x),
        ylim=(graph.arena.min_y, graph.arena.max_y),
        xlabel="Arena X",
        ylabel="Arena Y",
        title="Crystal Run Arena — regions, objectives, and Teacher routes",
    )
    axis.set_aspect("equal")
    axis.grid(alpha=0.15)
    axis.legend(loc="upper center", ncol=3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=150,
        metadata={"Software": "BotColosseo", "CreationTime": None},
    )
    plt.close(figure)
    return output_path


def _collect_episode(root: Path, graph: RegionGraph, case: EpisodeCase) -> list[np.ndarray]:
    teacher_name = M1_TEACHERS[case.task]
    teacher = create_teacher(teacher_name, graph)
    env = SingleAgentTaskEnv(
        config_path=root / "assets/scenarios/crystal_run/crystal_run.cfg",
        region_graph=graph,
    )
    frames: list[np.ndarray] = []
    event_types: list[str] = []
    success = False
    try:
        observation, _ = env.reset(seed=case.seed, task=case.task)
        teacher.reset(seed=case.seed, task=case.task)
        frames.append(observation.frame.copy())
        truncated = False
        while not success and not truncated:
            step = env.step(teacher.act(env.teacher_state()))
            frames.append(step.observation.frame.copy())
            event_types.extend(item.type.value for item in step.events)
            success = step.terminated
            truncated = step.truncated
    finally:
        env.close()
    if not success:
        raise RuntimeError(f"Held-out showcase episode failed for {case.task.value}")
    event = _SHOWCASE_EVENT[case.task]
    if event not in event_types:
        raise RuntimeError(f"Showcase episode omitted expected event {event}")
    label = overlay_text(
        task=case.task.value,
        teacher=teacher_name,
        event=event,
        success=success,
    )
    selected = [frames[index] for index in select_frame_indices(len(frames))]
    return [_add_overlay(frame, label) for frame in selected]


def _add_overlay(frame: np.ndarray, label: str) -> np.ndarray:
    rgb = normalize_rgb_frame(frame)
    enlarged = cv2.resize(rgb, (672, 672), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((720, 672, 3), dtype=np.uint8)
    canvas[48:] = enlarged
    cv2.putText(
        canvas,
        label,
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def render_montage(root: Path, graph: RegionGraph, output_path: Path) -> Path:
    cases = load_cases(root / "configs/m1/test.json", "test")
    frames: list[np.ndarray] = []
    for task in TaskKind:
        case = next(case for case in cases if case.task is task)
        frames.extend(_collect_episode(root, graph, case))
    return write_mp4(frames, output_path, fps=10)


def render_showcase(root: Path) -> tuple[Path, Path]:
    load_m1_summary(root / "reports/m1/summary.json")
    graph = RegionGraph.from_yaml(root / "assets/scenarios/crystal_run/src/regions.yaml")
    arena_path = root / "docs/assets/m1-arena.png"
    montage_path = root / "docs/assets/m1-teacher-montage.mp4"
    render_arena(graph, arena_path)
    render_montage(root, graph, montage_path)
    return arena_path, montage_path
