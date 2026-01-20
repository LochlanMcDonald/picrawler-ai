from __future__ import annotations

from behaviors.base_behavior import BaseBehavior


class ObjectDetectionBehavior(BaseBehavior):
    name = "detect"

    def context(self) -> str:
        target = self.target or "the target object"
        return (
            f"Search for {target}. If you see it centered and close, stop. "
            "If not visible, move/turn to search. Avoid obstacles."
        )
