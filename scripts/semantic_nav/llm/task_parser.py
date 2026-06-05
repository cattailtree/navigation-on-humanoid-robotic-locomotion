from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse
from urllib import request as url_request

from maps.semantic_graph import SemanticGraph
from planners.goal_parser import GoalParser, NavigationGoal


@dataclass(frozen=True)
class NavigationTaskParse:
    goal: NavigationGoal
    target_node_id: str | None = None
    rationale: str | None = None
    source: str = "rule"


class NavigationTaskParser(Protocol):
    def parse(
        self,
        text: str,
        current_floor: str,
        *,
        graph: SemanticGraph,
        start_node_id: str,
    ) -> NavigationTaskParse:
        ...


class RuleBasedTaskParser:
    """Compatibility wrapper around the original tiny rule-based parser."""

    def __init__(self) -> None:
        self._parser = GoalParser()

    def parse(
        self,
        text: str,
        current_floor: str,
        *,
        graph: SemanticGraph,
        start_node_id: str,
    ) -> NavigationTaskParse:
        del graph, start_node_id
        return NavigationTaskParse(goal=self._parser.parse(text, current_floor=current_floor), source="rule")


class HttpNavigationTaskParser:
    """External LLM task parser, kept out-of-process like ApexNav's VLM services."""

    VALID_INTENTS = {"floor_transition", "find_elevator", "local_goal"}

    def __init__(
        self,
        endpoint: str,
        *,
        mode: str = "task_endpoint",
        model: str | None = None,
        api_key_env: str | None = None,
        timeout_s: float = 20.0,
        log_raw: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.mode = mode
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.log_raw = log_raw

    def parse(
        self,
        text: str,
        current_floor: str,
        *,
        graph: SemanticGraph,
        start_node_id: str,
    ) -> NavigationTaskParse:
        context = _graph_context(graph)
        if self.mode == "openai_chat":
            payload = _chat_completion_payload(
                goal_text=text,
                current_floor=current_floor,
                start_node_id=start_node_id,
                graph_context=context,
                model=self.model or "local-llm",
            )
        else:
            payload = {
                "goal_text": text,
                "current_floor": current_floor,
                "start_node_id": start_node_id,
                "graph": context,
                "schema": _response_schema(),
            }

        try:
            response = self._post_json(payload)
            if self.log_raw:
                print("[semantic_nav:llm] raw response:", json.dumps(response, ensure_ascii=False))

            data = _extract_task_json(response) if self.mode == "openai_chat" else response
            return self._parse_response(data, text=text, current_floor=current_floor, graph=graph)
        finally:
            if self.mode == "openai_chat":
                unload_ollama_model(self.endpoint, self.model, timeout_s=2.0, log_raw=self.log_raw)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        http_request = url_request.Request(self.endpoint, data=body, headers=headers, method="POST")
        with url_request.urlopen(http_request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _parse_response(
        self,
        data: dict[str, Any],
        *,
        text: str,
        current_floor: str,
        graph: SemanticGraph,
    ) -> NavigationTaskParse:
        intent = str(data.get("intent", "local_goal")).strip()
        if intent not in self.VALID_INTENTS:
            raise ValueError(f"LLM returned unsupported navigation intent: {intent}")

        target_floor = _normalize_floor_id(_optional_str(data.get("target_floor")), current_floor=current_floor, graph=graph)
        target_label = _optional_str(data.get("target_label"))
        target_node_id = _resolve_target_node_id(
            graph,
            node_id=_optional_str(data.get("target_node_id")),
            label=target_label,
            floor=target_floor,
        )
        target_node_id = _repair_target_node_id(
            graph,
            text=text,
            target_floor=target_floor,
            target_label=target_label,
            target_node_id=target_node_id,
        )
        intent = _repair_intent(
            intent,
            text=text,
            current_floor=current_floor,
            target_floor=target_floor,
            target_node_id=target_node_id,
            graph=graph,
        )

        if target_node_id is not None and target_floor is None:
            target_floor = graph.nodes[target_node_id].floor
        if intent == "find_elevator" and target_floor is None:
            target_floor = current_floor
        if intent == "local_goal" and target_floor is None:
            target_floor = current_floor

        return NavigationTaskParse(
            goal=NavigationGoal(
                raw_text=text,
                intent=intent,
                target_floor=target_floor,
                target_label=target_label,
            ),
            target_node_id=target_node_id,
            rationale=_optional_str(data.get("rationale")),
            source=self.mode,
        )


def _response_schema() -> dict[str, Any]:
    return {
        "intent": "floor_transition | find_elevator | local_goal",
        "target_floor": "floor id such as F1 or B1, or null",
        "target_label": "human-readable target object/place, or null",
        "target_node_id": "one graph node id if the graph contains the requested target, or null",
        "rationale": "short reason, optional",
    }


def unload_ollama_model(endpoint: str | None, model: str | None, *, timeout_s: float = 2.0, log_raw: bool = False) -> bool:
    if endpoint is None or model is None:
        return False
    parsed = urlparse(endpoint)
    if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port != 11434:
        return False
    unload_url = f"{parsed.scheme or 'http'}://{parsed.netloc}/api/generate"
    body = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
    http_request = url_request.Request(
        unload_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with url_request.urlopen(http_request, timeout=timeout_s) as response:
            response.read()
        if log_raw:
            print(f"[semantic_nav:llm] unloaded Ollama model: {model}")
        return True
    except Exception as exc:
        if log_raw:
            print(f"[semantic_nav:llm] Ollama unload skipped: {exc}")
        return False


def _chat_completion_payload(
    *,
    goal_text: str,
    current_floor: str,
    start_node_id: str,
    graph_context: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    user_payload = {
        "goal_text": goal_text,
        "current_floor": current_floor,
        "start_node_id": start_node_id,
        "graph": graph_context,
        "response_schema": _response_schema(),
    }
    system_prompt = (
        "You parse natural-language navigation requests for a humanoid robot. "
        "Return only one JSON object matching the schema. Choose target_node_id only from the provided graph. "
        "Use intent=floor_transition whenever the request asks the robot to use/take an elevator to reach another floor or another-floor destination. "
        "Use intent=find_elevator only when the elevator itself is the final requested target and there is no destination beyond the elevator. "
        "Use intent=local_goal for same-floor targets. Prefer room/place target nodes over elevator nodes when the user names a room/place destination."
    )
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }


def _graph_context(graph: SemanticGraph) -> dict[str, Any]:
    nodes = []
    for node in graph.nodes.values():
        nodes.append(
            {
                "id": node.node_id,
                "floor": node.floor,
                "kind": node.kind,
                "label": node.label,
                "pose": [node.pose.x, node.pose.y, node.pose.yaw],
                "attrs": node.attrs,
            }
        )

    edges = []
    for node_id in graph.nodes:
        for edge in graph.outgoing_edges(node_id):
            edges.append(
                {
                    "src": edge.src,
                    "dst": edge.dst,
                    "kind": edge.kind,
                    "cost": edge.cost,
                    "attrs": edge.attrs,
                }
            )
    return {"nodes": nodes, "edges": edges}


def _extract_task_json(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("LLM chat response did not include choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(str(item.get("text", item)) for item in content)
    return _loads_json_object(str(content))


def _loads_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        data = json.loads(stripped[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM task parse must be a JSON object")
    return data


def _resolve_target_node_id(
    graph: SemanticGraph,
    *,
    node_id: str | None,
    label: str | None,
    floor: str | None,
) -> str | None:
    if node_id:
        if node_id in graph.nodes:
            return node_id
        matched = _match_node_by_text(graph, node_id, floor=floor)
        if matched is not None:
            return matched
        raise ValueError(f"LLM returned unknown target_node_id: {node_id}")
    if label:
        return _match_node_by_text(graph, label, floor=floor)
    return None


def _repair_intent(
    intent: str,
    *,
    text: str,
    current_floor: str,
    target_floor: str | None,
    target_node_id: str | None,
    graph: SemanticGraph,
) -> str:
    if intent != "find_elevator":
        return intent
    text_norm = text.lower()
    asks_only_for_elevator = any(token in text_norm for token in ("find elevator", "find the elevator", "locate elevator", "locate the elevator"))
    has_destination_words = any(token in text_norm for token in ("room", "target", "destination", "basement", "b1", "地下", "地下室", "房间", "目标"))
    if target_node_id is not None:
        node = graph.nodes[target_node_id]
        if node.floor != current_floor or node.kind not in {"elevator", "elevator_lobby"}:
            return "floor_transition" if node.floor != current_floor else "local_goal"
    if target_floor is not None and target_floor != current_floor and (has_destination_words or not asks_only_for_elevator):
        return "floor_transition"
    return intent


def _repair_target_node_id(
    graph: SemanticGraph,
    *,
    text: str,
    target_floor: str | None,
    target_label: str | None,
    target_node_id: str | None,
) -> str | None:
    text_norm = text.lower()
    label_norm = (target_label or "").lower()
    wants_room = any(token in text_norm or token in label_norm for token in ("room", "target", "房间", "目标"))
    if not wants_room:
        return target_node_id
    if target_node_id is not None:
        node = graph.nodes[target_node_id]
        if node.kind == "room":
            return target_node_id
    floor = target_floor or (graph.nodes[target_node_id].floor if target_node_id is not None else None)
    room_candidates = [
        node
        for node in graph.nodes.values()
        if node.kind == "room" and (floor is None or node.floor == floor)
    ]
    if not room_candidates:
        return target_node_id
    for node in room_candidates:
        label = node.label.lower()
        node_id = node.node_id.lower()
        if "target" in text_norm and ("target" in label or "target" in node_id):
            return node.node_id
    return room_candidates[0].node_id


def _normalize_floor_id(floor: str | None, *, current_floor: str, graph: SemanticGraph) -> str | None:
    if floor is None:
        return None
    floors = {node.floor for node in graph.nodes.values()}
    if floor in floors:
        return floor

    text = floor.strip().lower()
    aliases = {
        "current": current_floor,
        "current floor": current_floor,
        "same floor": current_floor,
        "basement": "B1",
        "basement floor": "B1",
        "lower floor": "B1",
        "downstairs": "B1",
        "地下": "B1",
        "地下室": "B1",
        "楼下": "B1",
        "一层": "F1",
        "1f": "F1",
        "first floor": "F1",
        "ground floor": "F1",
    }
    alias = aliases.get(text)
    if alias in floors:
        return alias
    upper = text.upper()
    if upper in floors:
        return upper
    return floor


def _match_node_by_text(graph: SemanticGraph, text: str, *, floor: str | None) -> str | None:
    query = text.strip().lower().replace("_", " ")
    if not query:
        return None

    candidates = [node for node in graph.nodes.values() if floor is None or node.floor == floor]
    for node in candidates:
        node_id = node.node_id.lower().replace("_", " ")
        label = node.label.lower()
        if query == node_id or query == label:
            return node.node_id
    for node in candidates:
        node_id = node.node_id.lower().replace("_", " ")
        label = node.label.lower()
        kind = node.kind.lower().replace("_", " ")
        if query in node_id or query in label or query in kind or node_id in query or label in query:
            return node.node_id
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nil"}:
        return None
    return text
