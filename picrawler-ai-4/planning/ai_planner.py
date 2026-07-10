"""
AI planner for high-level strategic decision making.

The AI's role is to provide strategy and context, not low-level motor control.
It sees the full world model and suggests multi-step plans.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI, APIError


@dataclass
class Plan:
    """Multi-step plan with fallbacks."""
    primary: List[str]
    fallback: List[str]
    recovery: List[str]
    reasoning: str
    confidence: float


class AIPlanner:
    """High-level strategic planner using AI."""

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)

        api_key = config.get("openai_api_key")
        self.client = OpenAI(api_key=None if api_key in (None, "", "your-api-key-here") else api_key)

        ai = config.get("ai_settings", {})
        self.model = ai.get("model", "gpt-4o-mini")
        self.temperature = float(ai.get("temperature", 0.4))
        self.timeout_s = int(ai.get("request_timeout_s", 30))

        self.last_plan_time = 0.0
        self.replan_interval = 5.0  # Replan every 5 seconds minimum

    def should_replan(self, consecutive_failures: int = 0) -> bool:
        """Decide if we should generate a new plan."""
        time_since_plan = time.time() - self.last_plan_time

        # Replan if been a while OR having failures
        if time_since_plan > self.replan_interval:
            return True

        if consecutive_failures >= 2:
            return True

        return False

    def plan_exploration(self, world_model, memory) -> Optional[Plan]:
        """Generate exploration strategy based on current state.

        Args:
            world_model: Current WorldModel with sensor fusion
            memory: SpatialMemory with action history

        Returns:
            Plan with primary/fallback/recovery sequences, or None if AI fails
        """
        # Build rich context for AI
        context = {
            'world_state': world_model.to_dict(),
            'memory': memory.to_dict(),
            'timestamp': time.time()
        }

        prompt = f"""You are a robot navigation strategist. Based on sensor data and action history, create a multi-step exploration plan.

**Current Situation:**
Obstacles:
- Front: {context['world_state']['obstacles']['front']}
- Left: {context['world_state']['obstacles']['left']}
- Right: {context['world_state']['obstacles']['right']}
- Rear: {context['world_state']['obstacles']['rear']}

Vision: {context['world_state']['vision']['description']}
Hazards seen: {context['world_state']['vision']['hazards']}

Free space score: {context['world_state']['free_space_score']:.2f} (0=trapped, 1=open)
Best direction: {context['world_state']['best_direction']}

**Recent History:**
Recent actions: {context['memory']['recent_actions']}
Failures: {context['memory']['failure_counts']}
Stuck status: {context['memory']['is_stuck']}
Best turn direction: {context['memory']['best_turn']}

**Action Scores (higher=better):**
{json.dumps(context['memory']['action_scores'], indent=2)}

**Instructions:**
Create a 3-5 step plan to explore effectively. Consider:
1. Sensor data shows PHYSICAL obstacles - respect ultrasonic readings!
2. Recent failures mean that approach isn't working
3. If stuck, need significant change in strategy
4. Plan should have primary path + fallback if blocked

Return STRICT JSON:
{{
    "primary": ["action1", "action2", "action3"],
    "fallback": ["alt_action1", "alt_action2"],
    "recovery": ["recovery_action"],
    "reasoning": "why this plan makes sense",
    "confidence": 0.8
}}

Valid actions: forward, backward, turn_left, turn_right, stop
"""

        try:
            self.logger.debug("Requesting plan from AI...")
            t0 = time.time()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                timeout=self.timeout_s
            )

            text = response.choices[0].message.content
            dt = time.time() - t0

            self.logger.debug(f"AI response received in {dt:.1f}s")

            # Parse JSON
            plan_data = self._parse_plan(text)

            if plan_data:
                plan = Plan(
                    primary=plan_data.get('primary', ['turn_left', 'forward']),
                    fallback=plan_data.get('fallback', ['turn_right', 'forward']),
                    recovery=plan_data.get('recovery', ['backward', 'turn_left']),
                    reasoning=plan_data.get('reasoning', ''),
                    confidence=float(plan_data.get('confidence', 0.5))
                )

                self.logger.info(f"Generated plan: {plan.primary} (confidence: {plan.confidence:.2f})")
                self.last_plan_time = time.time()
                return plan
            else:
                self.logger.error("Failed to parse AI plan")
                return None

        except (APIError, Exception) as e:
            self.logger.error(f"AI planner failed: {e}")
            return None

    def _parse_plan(self, text: str) -> Optional[Dict]:
        """Parse AI response into plan dictionary."""
        # Try to extract JSON
        try:
            # Handle code fences
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        return json.loads(part)

            # Try direct parse
            if text.strip().startswith("{"):
                return json.loads(text.strip())

        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parse error: {e}")

        return None

    def get_simple_action(self, world_model, memory) -> str:
        """Get single action suggestion (faster, for reactive behavior).

        This is used when we don't need a full plan, just next action.
        """
        # Use behavior tree logic instead of AI for fast decisions
        scores = memory.get_action_scores()

        # Filter by safety
        safe_actions = {}
        for action, score in scores.items():
            if action == 'forward' and not world_model.is_safe_to_move('forward'):
                continue  # Skip unsafe forward
            safe_actions[action] = score

        if not safe_actions:
            return 'backward'  # Last resort

        # Return highest scoring safe action
        return max(safe_actions, key=safe_actions.get)
