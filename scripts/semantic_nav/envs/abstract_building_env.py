from __future__ import annotations

import json
from pathlib import Path

from maps.semantic_graph import Pose2D, SemanticGraph, SemanticNode


DEFAULT_BUILDING_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "single_elevator_building.json"


def load_semantic_graph(config_path: str | Path = DEFAULT_BUILDING_CONFIG) -> SemanticGraph:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        cfg = json.load(file)

    graph = SemanticGraph()
    for item in cfg.get("nodes", []):
        pose_values = item["pose"]
        graph.add_node(
            SemanticNode(
                node_id=item["id"],
                floor=item["floor"],
                kind=item["kind"],
                pose=Pose2D(float(pose_values[0]), float(pose_values[1]), float(pose_values[2])),
                label=item.get("label", ""),
                attrs=item.get("attrs", {}),
            )
        )

    for item in cfg.get("edges", []):
        graph.add_edge(
            src=item["src"],
            dst=item["dst"],
            cost=float(item.get("cost", 1.0)),
            kind=item.get("kind", "walk"),
            bidirectional=bool(item.get("bidirectional", True)),
            attrs=item.get("attrs", {}),
        )
    return graph


def build_two_floor_elevator_graph() -> SemanticGraph:
    """Build a small oracle semantic graph for the first elevator-search experiment."""

    graph = SemanticGraph()
    nodes = [
        SemanticNode("start_f1", "F1", "start", Pose2D(0.0, 0.0), "start"),
        SemanticNode("corridor_f1", "F1", "corridor", Pose2D(4.0, 0.0), "main corridor"),
        SemanticNode("room_a_f1", "F1", "room", Pose2D(4.0, 3.0), "room A"),
        SemanticNode("elevator_a_f1", "F1", "elevator_lobby", Pose2D(8.0, 1.2), "elevator A lobby"),
        SemanticNode("elevator_b_f1", "F1", "elevator_lobby", Pose2D(12.0, -2.5), "elevator B lobby"),
        SemanticNode("elevator_a_b1", "B1", "elevator_lobby", Pose2D(8.0, 1.2), "elevator A lobby"),
        SemanticNode("elevator_b_b1", "B1", "elevator_lobby", Pose2D(12.0, -2.5), "elevator B lobby"),
        SemanticNode("corridor_b1", "B1", "corridor", Pose2D(6.0, 0.0), "basement corridor"),
        SemanticNode("target_room_b1", "B1", "room", Pose2D(2.0, -3.0), "target room"),
    ]
    for node in nodes:
        graph.add_node(node)

    graph.add_edge("start_f1", "corridor_f1", cost=4.0)
    graph.add_edge("corridor_f1", "room_a_f1", cost=3.0)
    graph.add_edge("corridor_f1", "elevator_a_f1", cost=4.2)
    graph.add_edge("corridor_f1", "elevator_b_f1", cost=8.4)
    graph.add_edge("elevator_a_f1", "elevator_a_b1", cost=1.0, kind="elevator_transition")
    graph.add_edge("elevator_b_f1", "elevator_b_b1", cost=1.0, kind="elevator_transition")
    graph.add_edge("elevator_a_b1", "corridor_b1", cost=2.4)
    graph.add_edge("elevator_b_b1", "corridor_b1", cost=6.2)
    graph.add_edge("corridor_b1", "target_room_b1", cost=5.0)
    return graph
