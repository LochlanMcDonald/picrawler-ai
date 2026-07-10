"""
Vision-Language Control System for Embodied AI.

Enables natural language commands that combine visual understanding
with task execution. Similar to Google RT-2 and other vision-language-action models.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available - language control disabled")


@dataclass
class LanguageCommand:
    """A natural language command with execution plan."""
    text: str  # Original command
    intent: str  # Classified intent (navigate, find, explore, etc.)
    target: Optional[str]  # Target object/location if specified
    actions: List[str]  # Sequence of robot actions to execute
    reasoning: str  # LLM's explanation of the plan
    confidence: float  # Confidence in the plan (0-1)
    timestamp: float
    navigation_goal: Optional[Tuple[float, float]] = None  # (x, y) coordinates for navigation


@dataclass
class SceneDescription:
    """Rich description of what the robot sees."""
    objects: List[str]  # Detected objects
    spatial_layout: str  # Description of object positions
    navigable_areas: List[str]  # Where robot can go
    obstacles: List[str]  # What's blocking
    suggested_actions: List[str]  # What robot could do next


class LanguageController:
    """Embodied AI controller using vision-language models."""

    # Valid robot actions for grounding
    VALID_ACTIONS = [
        "forward", "backward", "left", "right",
        "turn_left", "turn_right", "stop",
        "scan", "wait", "reverse"
    ]

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dict with API keys
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        # Get API key from config or environment variable
        api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            self.logger.error("OPENAI_API_KEY not set in config or environment")
            api_key = None

        self.api_key = api_key
        self.client = None

        if OPENAI_AVAILABLE and api_key:
            self.client = OpenAI(api_key=api_key)
            self.logger.info("Language controller initialized with OpenAI GPT-4 Vision")
        else:
            self.logger.warning("Language controller unavailable - missing dependencies or API key")

        # Command history
        self.command_history: List[LanguageCommand] = []
        self.current_command: Optional[LanguageCommand] = None

    def understand_scene(self, image_b64: str, depth_info: Optional[Dict] = None) -> Optional[SceneDescription]:
        """Analyze scene using vision-language model.

        Args:
            image_b64: Base64 encoded image
            depth_info: Optional depth information from depth estimator

        Returns:
            SceneDescription with detailed understanding
        """
        if not self.client:
            return None

        try:
            # Build prompt with depth context if available
            depth_context = ""
            if depth_info:
                depth_context = f"\n\nDepth information:\n- Front: {depth_info.get('front', 'unknown')}\n- Left: {depth_info.get('left', 'unknown')}\n- Right: {depth_info.get('right', 'unknown')}"

            prompt = f"""You are the vision system for a small mobile robot. Analyze this image and provide:

1. Objects visible in the scene (list specific items)
2. Spatial layout (where objects are positioned - left, right, center, near, far)
3. Navigable areas (where the robot could move - clear paths, open spaces)
4. Obstacles (what would block the robot - walls, furniture, objects)
5. Suggested actions (what the robot could do next - explore left, move forward, turn around, etc.)
{depth_context}

Respond in this exact JSON format:
{{
    "objects": ["object1", "object2"],
    "spatial_layout": "description of layout",
    "navigable_areas": ["area1", "area2"],
    "obstacles": ["obstacle1", "obstacle2"],
    "suggested_actions": ["action1", "action2"]
}}"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )

            # Parse response
            import json
            result = json.loads(response.choices[0].message.content)

            return SceneDescription(
                objects=result.get("objects", []),
                spatial_layout=result.get("spatial_layout", ""),
                navigable_areas=result.get("navigable_areas", []),
                obstacles=result.get("obstacles", []),
                suggested_actions=result.get("suggested_actions", [])
            )

        except Exception as e:
            self.logger.error(f"Scene understanding failed: {e}")
            return None

    def _extract_coordinates(self, command_text: str) -> Optional[Tuple[float, float]]:
        """Extract (x, y) coordinates from command text.

        Args:
            command_text: Natural language command

        Returns:
            (x, y) tuple or None if no coordinates found
        """
        # Look for patterns like: "coordinates (1.5, 0.8)", "(1.5, 0.8)", "x=1.5 y=0.8", etc.
        patterns = [
            r'coordinates?\s*\(([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\)',  # coordinates (x, y)
            r'\(([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\)',  # (x, y)
            r'x\s*=?\s*([+-]?\d+\.?\d*)\s*,?\s*y\s*=?\s*([+-]?\d+\.?\d*)',  # x=1.5, y=0.8
            r'position\s*\(([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\)'  # position (x, y)
        ]

        for pattern in patterns:
            match = re.search(pattern, command_text.lower())
            if match:
                try:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    return (x, y)
                except (ValueError, IndexError):
                    continue

        return None

    def parse_command(self, command_text: str, scene: Optional[SceneDescription] = None,
                     world_state: Optional[Dict] = None) -> Optional[LanguageCommand]:
        """Parse natural language command into executable actions.

        Args:
            command_text: Natural language command from user
            scene: Current scene understanding
            world_state: Current world model state

        Returns:
            LanguageCommand with action plan
        """
        if not self.client:
            return None

        try:
            # Check for navigation coordinates
            nav_coords = self._extract_coordinates(command_text)

            # Build context
            scene_context = ""
            if scene:
                scene_context = f"""
Current scene:
- Objects visible: {', '.join(scene.objects)}
- Spatial layout: {scene.spatial_layout}
- Navigable areas: {', '.join(scene.navigable_areas)}
- Obstacles: {', '.join(scene.obstacles)}
"""

            world_context = ""
            if world_state:
                world_context = f"""
Current sensors:
- Front distance: {world_state.get('front_distance', 'unknown')}
- Free space score: {world_state.get('free_space', 'unknown')}
- Best direction: {world_state.get('best_direction', 'unknown')}
"""

            prompt = f"""You are controlling a small mobile robot. The user gave you this command:

"{command_text}"
{scene_context}{world_context}
Available robot actions: {', '.join(self.VALID_ACTIONS)}

Parse this command and create an action plan. Respond in this exact JSON format:
{{
    "intent": "navigate|find|explore|avoid|stop|scan",
    "target": "target object or location if specified, else null",
    "actions": ["action1", "action2", "action3"],
    "reasoning": "brief explanation of your plan",
    "confidence": 0.0-1.0
}}

Rules:
- Only use actions from the available list
- Keep action sequences short (3-5 steps max)
- Consider the current scene and obstacles
- If you can't fulfill the command safely, return confidence < 0.5"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Parse response
            import json
            result = json.loads(response.choices[0].message.content)

            language_cmd = LanguageCommand(
                text=command_text,
                intent=result.get("intent", "unknown"),
                target=result.get("target"),
                actions=result.get("actions", []),
                reasoning=result.get("reasoning", ""),
                confidence=result.get("confidence", 0.5),
                timestamp=time.time(),
                navigation_goal=nav_coords
            )

            self.command_history.append(language_cmd)
            self.current_command = language_cmd

            self.logger.info(f"Command parsed: {command_text}")
            self.logger.info(f"Intent: {language_cmd.intent}, Actions: {language_cmd.actions}")
            if nav_coords:
                self.logger.info(f"Navigation goal: ({nav_coords[0]:.2f}, {nav_coords[1]:.2f})")
            self.logger.info(f"Reasoning: {language_cmd.reasoning}")

            return language_cmd

        except Exception as e:
            self.logger.error(f"Command parsing failed: {e}")
            return None

    def execute_command(self, command: LanguageCommand, robot, logger) -> bool:
        """Execute a language command using robot controller.

        Args:
            command: LanguageCommand to execute
            robot: RobotController instance
            logger: Logger for execution tracking

        Returns:
            True if successful, False otherwise
        """
        if not command or command.confidence < 0.5:
            logger.warning(f"Command confidence too low: {command.confidence}")
            return False

        logger.info(f"Executing command: {command.text}")
        logger.info(f"Plan: {' -> '.join(command.actions)}")

        success_count = 0
        for i, action in enumerate(command.actions):
            # Validate action
            if action not in self.VALID_ACTIONS:
                logger.warning(f"Invalid action skipped: {action}")
                continue

            # Map to robot execution
            try:
                if action == "forward":
                    robot.execute("forward", 1.0)
                elif action == "backward" or action == "reverse":
                    robot.execute("backward", 1.0)
                elif action in ["left", "turn_left"]:
                    robot.execute("left", 0.8)
                elif action in ["right", "turn_right"]:
                    robot.execute("right", 0.8)
                elif action == "stop":
                    robot.execute("stop", 0.5)
                elif action == "scan":
                    # Scan by turning in place
                    robot.execute("left", 0.5)
                    time.sleep(0.3)
                    robot.execute("right", 0.5)
                elif action == "wait":
                    time.sleep(1.0)

                success_count += 1
                logger.info(f"Action {i+1}/{len(command.actions)}: {action} completed")

                # Small delay between actions
                time.sleep(0.2)

            except Exception as e:
                logger.error(f"Action {action} failed: {e}")
                robot.execute("stop", 0.2)
                return False

        success_rate = success_count / len(command.actions) if command.actions else 0
        logger.info(f"Command execution complete: {success_count}/{len(command.actions)} actions succeeded")

        return success_rate > 0.5

    def get_command_history(self) -> List[LanguageCommand]:
        """Get history of executed commands."""
        return self.command_history

    def clear_history(self) -> None:
        """Clear command history."""
        self.command_history.clear()
        self.current_command = None
