from pathlib import Path

from botcolosseo.demo.m2_system_diagram import diagram_spec, render_system_diagram


def test_diagram_spec_keeps_privileged_state_out_of_actor_inputs() -> None:
    spec = diagram_spec()
    actor_inputs = {
        edge.source for edge in spec.edges if edge.target == "actor"
    }

    assert actor_inputs == {"legal_observation"}
    assert any(
        edge.source == "privileged_state" and edge.target == "critic"
        for edge in spec.edges
    )
    assert spec.nodes["official_evaluation"].status == "pending"


def test_system_diagram_renders_nonempty_png(tmp_path: Path) -> None:
    output = render_system_diagram(tmp_path / "m2-system.png")

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 10_000
