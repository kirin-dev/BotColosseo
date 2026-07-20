from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiagramNode:
    x: float
    y: float
    width: float
    height: float
    title: str
    detail: str
    status: str


@dataclass(frozen=True)
class DiagramEdge:
    source: str
    target: str
    label: str
    dashed: bool = False
    source_anchor: str = "right"
    target_anchor: str = "left"


@dataclass(frozen=True)
class SystemDiagramSpec:
    nodes: dict[str, DiagramNode]
    edges: tuple[DiagramEdge, ...]


def diagram_spec() -> SystemDiagramSpec:
    nodes = {
        "legal_observation": DiagramNode(
            0.5,
            5.5,
            3.4,
            1.55,
            "Fair Actor observation",
            "84x84 grayscale + legal scalars\nprevious action + recurrent state",
            "runtime",
        ),
        "privileged_state": DiagramNode(
            0.5,
            2.9,
            3.4,
            1.55,
            "Privileged training state",
            "positions + angles + regions\ncore location + carrier state",
            "training_only",
        ),
        "demonstrations": DiagramNode(
            9.0,
            0.55,
            2.9,
            1.35,
            "Teacher demonstrations",
            "100k train + 20k validation\nzero privileged fields in shards",
            "evidence",
        ),
        "actor": DiagramNode(
            4.8,
            5.25,
            3.6,
            1.8,
            "CNN + MLP -> GRU(256)",
            "13-action categorical Actor\nused unchanged at inference",
            "runtime",
        ),
        "critic": DiagramNode(
            4.8,
            2.75,
            3.6,
            1.6,
            "Asymmetric Critic",
            "Actor features + privileged encoder\ndiscarded at inference",
            "training_only",
        ),
        "training": DiagramNode(
            9.0,
            3.8,
            2.9,
            1.8,
            "Pure BC -> PPO",
            "recurrent updates + KL guard\nvalidation-only selection",
            "training_only",
        ),
        "selected_policy": DiagramNode(
            12.7,
            5.35,
            2.8,
            1.6,
            "Selected PPO Actor",
            "800k-step checkpoint\nfair observations only",
            "evidence",
        ),
        "validation_evidence": DiagramNode(
            12.7,
            2.85,
            2.8,
            1.45,
            "Validation evidence",
            "10 candidates x 30 games\ntracked curves + video",
            "evidence",
        ),
        "official_evaluation": DiagramNode(
            12.7,
            0.55,
            2.8,
            1.35,
            "Official M2 gate",
            "1,500 paired test games\nnot run yet",
            "pending",
        ),
    }
    edges = (
        DiagramEdge("legal_observation", "actor", ""),
        DiagramEdge(
            "actor",
            "critic",
            "",
            dashed=True,
            source_anchor="bottom",
            target_anchor="top",
        ),
        DiagramEdge("privileged_state", "critic", "", dashed=True),
        DiagramEdge("actor", "training", ""),
        DiagramEdge("critic", "training", ""),
        DiagramEdge(
            "demonstrations",
            "training",
            "BC labels",
            source_anchor="top",
            target_anchor="bottom",
        ),
        DiagramEdge("training", "selected_policy", ""),
        DiagramEdge(
            "selected_policy",
            "validation_evidence",
            "frozen cases",
            source_anchor="bottom",
            target_anchor="top",
        ),
        DiagramEdge(
            "validation_evidence",
            "official_evaluation",
            "pending test",
            source_anchor="bottom",
            target_anchor="top",
        ),
    )
    return SystemDiagramSpec(nodes=nodes, edges=edges)


def render_system_diagram(output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    spec = diagram_spec()
    colors = {
        "runtime": ("#DBEAFE", "#2563EB"),
        "training_only": ("#FFEDD5", "#EA580C"),
        "evidence": ("#D1FAE5", "#059669"),
        "pending": ("#E2E8F0", "#64748B"),
    }
    fig, axis = plt.subplots(figsize=(16, 8.5), dpi=150)
    fig.patch.set_facecolor("#F8FAFC")
    axis.set_facecolor("#F8FAFC")
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 8.4)
    axis.axis("off")

    def anchor(node: DiagramNode, name: str) -> tuple[float, float]:
        if name == "left":
            return node.x, node.y + node.height / 2
        if name == "right":
            return node.x + node.width, node.y + node.height / 2
        if name == "top":
            return node.x + node.width / 2, node.y + node.height
        if name == "bottom":
            return node.x + node.width / 2, node.y
        raise ValueError(f"Unsupported diagram anchor: {name}")

    for edge in spec.edges:
        source = spec.nodes[edge.source]
        target = spec.nodes[edge.target]
        start = anchor(source, edge.source_anchor)
        end = anchor(target, edge.target_anchor)
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#64748B",
                "linewidth": 1.5,
                "linestyle": "--" if edge.dashed else "-",
                "shrinkA": 3,
                "shrinkB": 3,
                "connectionstyle": "arc3,rad=0.0",
            },
            zorder=1,
        )
        if edge.label:
            axis.text(
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2 + 0.08,
                edge.label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#64748B",
                zorder=2,
            )

    for node in spec.nodes.values():
        fill, border = colors[node.status]
        box = FancyBboxPatch(
            (node.x, node.y),
            node.width,
            node.height,
            boxstyle="round,pad=0.03,rounding_size=0.09",
            linewidth=1.8,
            edgecolor=border,
            facecolor=fill,
            zorder=3,
        )
        axis.add_patch(box)
        axis.text(
            node.x + 0.18,
            node.y + node.height - 0.34,
            node.title,
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#0F172A",
            zorder=4,
        )
        axis.text(
            node.x + 0.18,
            node.y + node.height - 0.72,
            node.detail,
            ha="left",
            va="top",
            fontsize=8.7,
            linespacing=1.45,
            color="#334155",
            zorder=4,
        )

    axis.text(
        0.5,
        8.05,
        "M2 Learning System: Fair at Inference, Privileged Only During Training",
        fontsize=17,
        fontweight="bold",
        color="#0F172A",
    )
    axis.text(
        0.5,
        7.65,
        "Blue boxes are the deployed path. Orange information never enters policy logits.",
        fontsize=10,
        color="#475569",
    )

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    fig.savefig(temporary, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    temporary.replace(output)
    return output
