from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Callable


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class SemanticNode:
    node_id: str
    floor: str
    kind: str
    pose: Pose2D
    label: str = ""
    attrs: dict[str, str | float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticEdge:
    src: str
    dst: str
    cost: float
    kind: str = "walk"
    attrs: dict[str, str | float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphPath:
    node_ids: list[str]
    edges: list[SemanticEdge]
    cost: float

    @property
    def is_empty(self) -> bool:
        return not self.node_ids


class SemanticGraph:
    """Small directed semantic graph used by the long-horizon nav prototypes."""

    def __init__(self) -> None:
        self.nodes: dict[str, SemanticNode] = {}
        self._edges: dict[str, list[SemanticEdge]] = defaultdict(list)

    def add_node(self, node: SemanticNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node id: {node.node_id}")
        self.nodes[node.node_id] = node

    def update_node(self, node: SemanticNode) -> None:
        if node.node_id not in self.nodes:
            raise KeyError(f"Unknown node id: {node.node_id}")
        self.nodes[node.node_id] = node

    def add_edge(
        self,
        src: str,
        dst: str,
        cost: float,
        kind: str = "walk",
        bidirectional: bool = True,
        attrs: dict[str, str | float | bool] | None = None,
    ) -> None:
        if src not in self.nodes:
            raise KeyError(f"Unknown source node: {src}")
        if dst not in self.nodes:
            raise KeyError(f"Unknown destination node: {dst}")
        edge_attrs = attrs or {}
        self._edges[src].append(SemanticEdge(src=src, dst=dst, cost=cost, kind=kind, attrs=edge_attrs))
        if bidirectional:
            self._edges[dst].append(SemanticEdge(src=dst, dst=src, cost=cost, kind=kind, attrs=edge_attrs))

    def outgoing_edges(self, node_id: str) -> list[SemanticEdge]:
        return list(self._edges.get(node_id, []))

    def shortest_path(
        self,
        start: str,
        goal_fn: Callable[[SemanticNode], bool],
        edge_filter: Callable[[SemanticEdge], bool] | None = None,
    ) -> GraphPath:
        if start not in self.nodes:
            raise KeyError(f"Unknown start node: {start}")

        queue: list[tuple[float, str]] = [(0.0, start)]
        best_cost: dict[str, float] = {start: 0.0}
        prev_node: dict[str, str] = {}
        prev_edge: dict[str, SemanticEdge] = {}
        visited: set[str] = set()

        while queue:
            curr_cost, curr = heappop(queue)
            if curr in visited:
                continue
            visited.add(curr)

            if goal_fn(self.nodes[curr]):
                return self._reconstruct_path(start, curr, curr_cost, prev_node, prev_edge)

            for edge in self.outgoing_edges(curr):
                if edge_filter is not None and not edge_filter(edge):
                    continue
                next_cost = curr_cost + edge.cost
                if next_cost < best_cost.get(edge.dst, float("inf")):
                    best_cost[edge.dst] = next_cost
                    prev_node[edge.dst] = curr
                    prev_edge[edge.dst] = edge
                    heappush(queue, (next_cost, edge.dst))

        return GraphPath(node_ids=[], edges=[], cost=float("inf"))

    def _reconstruct_path(
        self,
        start: str,
        goal: str,
        cost: float,
        prev_node: dict[str, str],
        prev_edge: dict[str, SemanticEdge],
    ) -> GraphPath:
        nodes = [goal]
        edges: list[SemanticEdge] = []
        curr = goal
        while curr != start:
            edge = prev_edge[curr]
            edges.append(edge)
            curr = prev_node[curr]
            nodes.append(curr)
        nodes.reverse()
        edges.reverse()
        return GraphPath(node_ids=nodes, edges=edges, cost=cost)

    def describe_path(self, path: GraphPath) -> list[str]:
        if path.is_empty:
            return []
        lines: list[str] = []
        for node_id in path.node_ids:
            node = self.nodes[node_id]
            label = node.label or node.node_id
            lines.append(f"{node.node_id} [{node.kind}, floor={node.floor}, label={label}]")
        return lines
