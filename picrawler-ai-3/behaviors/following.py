from __future__ import annotations

from behaviors.base_behavior import BaseBehavior


class FollowingBehavior(BaseBehavior):
    name = "follow"

    def context(self) -> str:
        target = self.target or "person"
        return (
            f"Follow the {target} at a safe distance. If the target is centered and not too close, move forward slowly. "
            "If the target is left/right, turn to center it. If lost, rotate to reacquire. Avoid obstacles."
        )
