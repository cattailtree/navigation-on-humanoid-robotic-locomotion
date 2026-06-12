from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavigationGoal:
    raw_text: str
    intent: str
    target_floor: str | None = None
    target_label: str | None = None


class GoalParser:
    """Tiny rule-based parser for the first semantic-nav prototype."""

    DOWNSTAIRS_TOKENS = ("downstairs", "go down", "lower floor", "basement", "下楼", "楼下", "地下")
    ELEVATOR_TOKENS = ("elevator", "lift", "电梯")

    def parse(self, text: str, current_floor: str) -> NavigationGoal:
        text_norm = text.strip().lower()
        target_floor = self._infer_target_floor(text_norm, current_floor)

        if any(token in text_norm for token in self.DOWNSTAIRS_TOKENS):
            return NavigationGoal(raw_text=text, intent="floor_transition", target_floor=target_floor)
        if any(token in text_norm for token in self.ELEVATOR_TOKENS):
            return NavigationGoal(raw_text=text, intent="find_elevator", target_floor=current_floor)
        if self._infer_search_label(text_norm):
            return NavigationGoal(
                raw_text=text,
                intent="open_set_object_search",
                target_floor=current_floor,
                target_label=self._infer_search_label(text_norm),
            )
        return NavigationGoal(raw_text=text, intent="local_goal", target_floor=current_floor, target_label=text.strip())

    def _infer_target_floor(self, text: str, current_floor: str) -> str | None:
        if "b1" in text or "地下" in text or "basement" in text:
            return "B1"
        if current_floor == "F1" and any(token in text for token in self.DOWNSTAIRS_TOKENS):
            return "B1"
        return None

    def _infer_search_label(self, text: str) -> str | None:
        prefixes = (
            "find the ",
            "find a ",
            "find an ",
            "find ",
            "locate the ",
            "locate a ",
            "locate an ",
            "locate ",
            "search for the ",
            "search for a ",
            "search for an ",
            "search for ",
            "look for the ",
            "look for a ",
            "look for an ",
            "look for ",
            "找",
            "寻找",
        )
        for prefix in prefixes:
            if text.startswith(prefix):
                label = text[len(prefix) :].strip(" .,!?:;，。！？：；")
                return label or None
        return None
