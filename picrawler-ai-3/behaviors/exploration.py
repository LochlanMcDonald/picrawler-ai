from __future__ import annotations

from behaviors.base_behavior import BaseBehavior


class ExplorationBehavior(BaseBehavior):
    name = "explore"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_action: str | None = None

    def context(self) -> str:
        return "Explore the area safely. Prefer forward motion when clear. Avoid obstacles."

    def postprocess_action(self, action: str) -> str:
        """
        Anti-loop / curiosity bias.

        Prevents immediate oscillation like:
          turn_left -> turn_right -> turn_left

        If detected, bias toward forward motion instead.
        """
        if self._last_action is None:
            self._last_action = action
            return action

        opposite = {
            "turn_left": "turn_right",
            "turn_right": "turn_left",
        }

        if opposite.get(self._last_action) == action:
            # Bias toward curiosity: try moving forward instead of oscillating
            biased_action = "forward"
            self._last_action = biased_action
            return biased_action

        self._last_action = action
        return action