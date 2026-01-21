from __future__ import annotations

from typing import Optional

from behaviors.base_behavior import BaseBehavior, ActionChoice


class ExplorationBehavior(BaseBehavior):
    name = "explore"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_action: Optional[str] = None

    def context(self) -> str:
        return "Explore the area safely. Prefer forward motion when clear. Avoid obstacles."

    # IMPORTANT: match BaseBehavior signature (keyword-only analysis + **kwargs)
    def postprocess_action(self, action: str, *, analysis=None, **kwargs) -> ActionChoice:
        """
        Adds a small exploration-specific bias on top of BaseBehavior anti-loop logic.
        """
        # First let the BaseBehavior anti-loop system run (scene stagnation, bans, escape ladder, etc.)
        choice = super().postprocess_action(action, analysis=analysis, **kwargs)

        # Unpack choice to a plain action string so we can apply our bias safely,
        # then re-pack if needed.
        if isinstance(choice, tuple):
            base_action, base_dur = choice
        else:
            base_action, base_dur = choice, None

        # If base already decided to stop, don't fight it.
        if base_action == "stop":
            self._last_action = base_action
            return choice

        # Small extra bias: if we detect immediate L<->R oscillation, try forward once.
        opposite = {"turn_left": "turn_right", "turn_right": "turn_left"}
        if self._last_action and opposite.get(self._last_action) == base_action:
            biased = "forward"
            self._last_action = biased
            # keep duration override if one existed (otherwise let caller use model duration)
            return (biased, base_dur) if base_dur is not None else biased

        self._last_action = base_action
        return choice