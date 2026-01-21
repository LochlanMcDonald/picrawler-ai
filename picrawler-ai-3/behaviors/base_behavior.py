from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple, Union

from ai.vision_ai import AIVisionSystem
from core.robot_controller import RobotController
from vision.camera import CameraSystem
from voice.voice_system import VoiceSystem


ActionChoice = Union[str, Tuple[str, float]]  # ("action", duration_override_s)


class BaseBehavior:
    name: str = "base"

    def __init__(
        self,
        config: dict,
        robot: RobotController,
        camera: CameraSystem,
        ai: AIVisionSystem,
        duration_minutes: float,
        target: Optional[str] = None,
        verbose: bool = False,
    ):
        self.config = config
        self.robot = robot
        self.camera = camera
        self.ai = ai
        self.duration_minutes = duration_minutes
        self.target = target
        self.verbose = verbose

        self.logger = logging.getLogger(self.__class__.__name__)
        self.decisions_path = Path("logs") / "decisions.jsonl"

        # Voice system (safe: no-op if disabled)
        self.voice = VoiceSystem(config)

        # Narration throttling
        vs = config.get("voice_settings", {}) if isinstance(config, dict) else {}
        self._narration_enabled = bool(vs.get("enabled", True)) and bool(vs.get("narration_enabled", True))
        self._narration_min_interval_s = float(vs.get("narration_min_interval_s", 4.0))
        self._last_narration_at = 0.0

        # --- AI Dialogue settings ---
        self._dialogue_enabled = bool(vs.get("enabled", True)) and bool(vs.get("dialogue_enabled", True))
        self._dialogue_min_interval_s = float(vs.get("dialogue_min_interval_s", 10.0))
        self._last_dialogue_at = 0.0
        self._last_dialogue_text: Optional[str] = None
        self._dialogue_dedupe_window_s = float(vs.get("dialogue_dedupe_window_s", 30.0))
        self._last_dialogue_text_at = 0.0
        # ----------------------------

        # --- Curiosity / anti-loop state ---
        bl = (config.get("behavior_settings") or {}) if isinstance(config, dict) else {}
        self._recent_actions: Deque[str] = deque(maxlen=int(bl.get("anti_loop_history", 8)))
        self._repeat_threshold = int(bl.get("anti_loop_repeat_threshold", 3))
        self._oscillation_window = int(bl.get("anti_loop_oscillation_window", 4))

        # Escape ladder (action, duration_override)
        self._escape_plan: List[Tuple[str, float]] = [
            ("stop", float(bl.get("escape_stop_s", 0.3))),
            ("backward", float(bl.get("escape_back_s", 0.9))),
            ("turn_left", float(bl.get("escape_turn_s", 0.9))),  # may swap to right dynamically
            ("forward", float(bl.get("escape_forward_s", 1.1))),
        ]
        self._escape_stage = 0
        self._last_escape_at = 0.0
        self._escape_cooldown_s = float(bl.get("escape_cooldown_s", 3.0))

        # If we keep triggering escapes rapidly, increase severity.
        self._escape_strikes = 0
        self._max_strikes_before_reset = int(bl.get("escape_strikes_reset", 6))
        # ----------------------------------

    # -----------------------------------------------------

    def available_actions(self) -> List[str]:
        return ["forward", "turn_left", "turn_right", "backward", "stop"]

    def context(self) -> str:
        return f"mode={self.name} target={self.target}".strip()

    # -----------------------------------------------------
    # 🧠 Anti-loop / curiosity logic
    # -----------------------------------------------------

    def postprocess_action(self, action: str) -> ActionChoice:
        """
        Bias actions to avoid getting stuck in loops.
        Returns either:
          - "action"
          - ("action", duration_override_s)

        Never overrides 'stop' (unless it's part of the escape ladder we choose).
        """
        # Always record what the model *wanted* to do.
        if action:
            self._recent_actions.append(action)

        # If model wants stop, honor it.
        if action == "stop":
            self._reset_escape_if_safe()
            return action

        # Not enough history yet.
        if len(self._recent_actions) < self._repeat_threshold:
            self._reset_escape_if_safe()
            return action

        # Detect same-action repetition (e.g., turn_right x3).
        if self._is_repeating(action):
            return self._escape(action, reason=f"repeat:{action}")

        # Detect oscillation left-right-left-right.
        if self._is_oscillating():
            return self._escape(action, reason="oscillation")

        # If no loop detected, slowly relax escape stage.
        self._reset_escape_if_safe()
        return action

    def _is_repeating(self, action: str) -> bool:
        last_actions = list(self._recent_actions)[-self._repeat_threshold :]
        return len(last_actions) == self._repeat_threshold and all(a == action for a in last_actions)

    def _is_oscillating(self) -> bool:
        if len(self._recent_actions) < self._oscillation_window:
            return False
        a, b, c, d = list(self._recent_actions)[-4:]
        # basic ABAB pattern where A != B
        return a == c and b == d and a != b

    def _escape(self, stuck_action: str, *, reason: str) -> ActionChoice:
        """
        Execute an escape ladder to break out of loops.
        We also dynamically pick turn direction opposite the stuck turn.
        """
        now = time.time()

        # Cooldown: if we just escaped, don't keep hammering changes every frame.
        if (now - self._last_escape_at) < self._escape_cooldown_s:
            # During cooldown, prefer a gentle forward/back instead of endless turning.
            if stuck_action in ("turn_left", "turn_right"):
                return ("backward", self._escape_plan[1][1])
            return ("forward", self._escape_plan[-1][1])

        self._last_escape_at = now
        self._escape_strikes += 1
        if self._escape_strikes >= self._max_strikes_before_reset:
            # Hard reset so we don't spiral forever.
            self._escape_strikes = 0
            self._escape_stage = 0

        # Decide which stage to use.
        stage = self._escape_stage % len(self._escape_plan)
        action, dur = self._escape_plan[stage]

        # Dynamic “turn opposite” if we are stuck turning.
        if action in ("turn_left", "turn_right"):
            if stuck_action == "turn_right":
                action = "turn_left"
            elif stuck_action == "turn_left":
                action = "turn_right"
            # If stuck in forward/backward loops, alternate turns by stage parity.
            else:
                action = "turn_left" if (stage % 2 == 0) else "turn_right"

        # Escalate for next time.
        self._escape_stage += 1

        # Clear recent action history so we don't re-trigger immediately.
        self._recent_actions.clear()
        self._recent_actions.append(action)

        # Speak once per escape event (short).
        self.logger.debug(f"Anti-loop escape triggered ({reason}) -> {action} ({dur:.2f}s)")
        self._narrate("I’m stuck—breaking the loop.", level="normal")

        return (action, dur)

    def _reset_escape_if_safe(self) -> None:
        """
        If we're not looping, gently decay escape stage/strikes over time.
        """
        now = time.time()
        if (now - self._last_escape_at) > (self._escape_cooldown_s * 2):
            if self._escape_stage > 0:
                self._escape_stage -= 1
            if self._escape_strikes > 0:
                self._escape_strikes -= 1

    # -----------------------------------------------------
    # Voice helpers
    # -----------------------------------------------------

    def _narrate(self, text: str, *, level: str = "normal", force: bool = False) -> None:
        if not self._narration_enabled:
            return
        now = time.time()
        if not force and (now - self._last_narration_at) < self._narration_min_interval_s:
            return
        self.voice.say(text, level=level, force=force)
        self._last_narration_at = now

    def _speak_dialogue(self, analysis, decision: dict, executed_action: str) -> None:
        """
        AI-generated dialogue (short, personality-driven).
        Throttled + deduped. Safe: non-fatal if unavailable.
        """
        if not self._dialogue_enabled:
            return

        now = time.time()
        if (now - self._last_dialogue_at) < self._dialogue_min_interval_s:
            return

        # Only attempt if the AI system exposes the method (ai/vision_ai.py)
        if not hasattr(self.ai, "generate_dialogue"):
            return

        try:
            line = self.ai.generate_dialogue(  # type: ignore[attr-defined]
                mode=self.name,
                target=self.target,
                analysis=analysis,
                decision=decision,
                executed_action=executed_action,
            )
            line = (line or "").strip()
            if not line:
                return

            if (
                self._last_dialogue_text == line
                and (now - self._last_dialogue_text_at) < self._dialogue_dedupe_window_s
            ):
                return

            self.voice.say(line, level="normal", force=False)

            self._last_dialogue_at = now
            self._last_dialogue_text = line
            self._last_dialogue_text_at = now

        except Exception as e:
            self.logger.debug(f"Dialogue generation failed (non-fatal): {e}")
            self._last_dialogue_at = now  # prevent hammering on repeated failures

    # -----------------------------------------------------

    def run(self) -> None:
        end_t = time.time() + (self.duration_minutes * 60)
        self.logger.info(f"Starting behavior '{self.name}' for {self.duration_minutes} min")

        # Narrate start
        start_line = f"Starting {self.name} mode."
        if self.target:
            start_line = f"Starting {self.name} mode. Target: {self.target}."
        self._narrate(start_line, level="normal", force=True)

        capture_interval = float(self.config.get("camera_settings", {}).get("capture_interval_s", 2.5))
        last_capture = 0.0

        while time.time() < end_t:
            now = time.time()
            if now - last_capture < capture_interval:
                time.sleep(0.05)
                continue

            last_capture = now
            pil, b64, path = self.camera.capture(
                save=bool(self.config.get("logging_settings", {}).get("save_images", True))
            )
            if b64 is None:
                self.logger.warning("No camera frame; stopping")
                self._narrate("Stopping.", level="normal")
                self.robot.execute("stop", 0.3)
                time.sleep(0.5)
                continue

            analysis = self.ai.analyze_scene(b64, context=self.context())

            if getattr(analysis, "description", "") in {"AI unavailable", "AI error"}:
                self._narrate("I can't reach my AI right now. Stopping.", level="normal", force=True)

            decision = self.ai.decide_action(
                analysis=analysis,
                mode=self.name,
                available_actions=self.available_actions(),
                target=self.target,
            )

            raw_action = decision.get("action", "stop")
            raw_duration_s = float(decision.get("duration_s", 0.6))

            choice = self.postprocess_action(raw_action)
            if isinstance(choice, tuple):
                action, duration_s = choice
            else:
                action, duration_s = choice, raw_duration_s

            record = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode": self.name,
                "target": self.target,
                "image_path": path,
                "analysis": {
                    "description": getattr(analysis, "description", ""),
                    "objects": getattr(analysis, "objects", []),
                    "hazards": getattr(analysis, "hazards", []),
                    "suggested_actions": getattr(analysis, "suggested_actions", []),
                    "processing_time_s": getattr(analysis, "processing_time_s", 0.0),
                },
                "decision": decision,
                "executed_action": action,
                "executed_duration_s": duration_s,
            }
            if bool(self.config.get("logging_settings", {}).get("save_decisions", True)):
                self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
                with self.decisions_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

            if self.verbose:
                self.logger.info(f"Seen: {getattr(analysis, 'description', '')}")
                self.logger.info(f"Decision: {decision}")
                self.logger.info(f"Executing: {action} ({duration_s:.2f}s)")

            # Simple narration (short)
            if action == "forward":
                self._narrate("Moving forward.", level="normal")
            elif action == "turn_left":
                self._narrate("Turning left.", level="normal")
            elif action == "turn_right":
                self._narrate("Turning right.", level="normal")
            elif action == "backward":
                self._narrate("Backing up.", level="normal")
            elif action == "stop":
                self._narrate("Stopping.", level="normal")

            # AI-generated dialogue line (throttled)
            self._speak_dialogue(analysis, decision, action)

            self.robot.execute(action, duration_s)

        self.logger.info(f"Behavior '{self.name}' complete")
        self._narrate(f"{self.name} mode complete.", level="normal", force=True)