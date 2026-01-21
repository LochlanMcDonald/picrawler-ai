from __future__ import annotations

from typing import Optional

from behaviors.base_behavior import BaseBehavior, ActionChoice


class ExplorationBehavior(BaseBehavior):
    name = "explore"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_action: Optional[str] = None

        # Optional: tweak explore-mode anti-loop aggressiveness without touching config.
        # (Config still wins if you add behavior_settings.anti_loop.* later.)
        # self._max_consecutive_turns = max(self._max_consecutive_turns, 5)

    def context(self) -> str:
        return "Explore the area safely. Prefer forward motion when clear. Avoid obstacles."

    def postprocess_action(self, action: str, *, analysis=None, **kwargs) -> ActionChoice:
        """
        Exploration adds only LIGHT bias on top of BaseBehavior:
          1) Run BaseBehavior anti-loop first (scene stagnation, bans, escape ladder, etc.).
          2) If it detects immediate L<->R oscillation, try forward once (short).
          3) If the model keeps choosing the same turn direction, we *optionally* clamp it here too.
        """

        # 1) Let BaseBehavior handle the heavy lifting first
        choice = super().postprocess_action(action, analysis=analysis, **kwargs)

        # Unpack for local biasing
        if isinstance(choice, tuple):
            base_action, base_dur = choice
        else:
            base_action, base_dur = choice, None

        # If base already decided to stop, do not override
        if base_action == "stop":
            self._last_action = base_action
            return choice

        # 2) Immediate oscillation bias: L then R (or R then L) -> forward once
        opposite = {"turn_left": "turn_right", "turn_right": "turn_left"}
        if self._last_action and opposite.get(self._last_action) == base_action:
            biased = "forward"
            self._last_action = biased

            # Keep duration override if base provided one; otherwise a small nudge forward
            if base_dur is not None:
                return (biased, base_dur)
            return (biased, 0.9)

        # 3) Explore-mode “turn_right loop” clamp:
        # If the model keeps asking for the SAME turn direction repeatedly, force a scene-change move.
        # (BaseBehavior already handles repeats, but this adds a quicker nudge specifically for explore.)
        if self._last_action == base_action and base_action in ("turn_left", "turn_right"):
            # If we have analysis and it's clearly not changing, back up a bit
            # (This complements BaseBehavior's scene signature / stagnation checks.)
            self._last_action = "backward"
            return ("backward", 1.0)

        self._last_action = base_action
        return choice