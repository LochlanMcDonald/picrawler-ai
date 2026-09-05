from __future__ import annotations

from behaviors.base_behavior import BaseBehavior


class AvoidanceBehavior(BaseBehavior):
    name = "avoid"

    def context(self) -> str:
        avoid = self.target or "people/pets"
        return (
            f"Explore while avoiding {avoid}. If {avoid} is near or centered, stop or turn away. "
            "Prefer turning over moving forward when uncertain. Always avoid obstacles."
        )
