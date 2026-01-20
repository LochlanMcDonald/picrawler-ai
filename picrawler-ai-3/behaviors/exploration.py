from __future__ import annotations

from behaviors.base_behavior import BaseBehavior


class ExplorationBehavior(BaseBehavior):
    name = "explore"

    def context(self) -> str:
        return "Explore the area safely. Prefer forward motion when clear. Avoid obstacles."
