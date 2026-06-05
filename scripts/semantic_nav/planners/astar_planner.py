from __future__ import annotations

from heapq import heappop, heappush
from math import hypot
from typing import Callable

from maps.semantic_graph import GraphPath, SemanticEdge, SemanticGraph, SemanticNode


class SemanticAStarPlanner:
    """A* over the semantic/topological graph."""

    def __init__(self, graph: SemanticGraph) -> None:
        self.graph = graph

    def plan_to_any(
        self,
        start: str,
        goal_node_ids: list[str],
        edge_filter: Callable[[SemanticEdge], bool] | None = None,
    ) -> GraphPath:
        if start not in self.graph.nodes:
            raise KeyError(f"Unknown start node: {start}")
        goals = [node_id for node_id in goal_node_ids if node_id in self.graph.nodes]
        if not goals:
            return GraphPath(node_ids=[], edges=[], cost=float("inf"))

        goal_set = set(goals)
        queue: list[tuple[float, float, str]] = [(self._heuristic_to_any(start, goals), 0.0, start)]
        best_cost: dict[str, float] = {start: 0.0}
        prev_node: dict[str, str] = {}
        prev_edge: dict[str, SemanticEdge] = {}
        visited: set[str] = set()

        while queue:
            _, curr_cost, curr = heappop(queue)
            if curr in visited:
                continue
            visited.add(curr)

            if curr in goal_set:
                return self._reconstruct_path(start, curr, curr_cost, prev_node, prev_edge)

            for edge in self.graph.outgoing_edges(curr):
                if edge_filter is not None and not edge_filter(edge):
                    continue
                next_cost = curr_cost + edge.cost
                if next_cost >= best_cost.get(edge.dst, float("inf")):
                    continue
                best_cost[edge.dst] = next_cost
                prev_node[edge.dst] = curr
                prev_edge[edge.dst] = edge
                priority = next_cost + self._heuristic_to_any(edge.dst, goals)
                heappush(queue, (priority, next_cost, edge.dst))

        return GraphPath(node_ids=[], edges=[], cost=float("inf"))

    def _heuristic_to_any(self, node_id: str, goal_node_ids: list[str]) -> float:
        node = self.graph.nodes[node_id]
        return min(_node_distance(node, self.graph.nodes[goal_id]) for goal_id in goal_node_ids)

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


def _node_distance(a: SemanticNode, b: SemanticNode) -> float:
    if a.floor != b.floor:
        return 0.0
    return hypot(a.pose.x - b.pose.x, a.pose.y - b.pose.y)

