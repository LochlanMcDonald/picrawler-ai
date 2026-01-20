from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

from ai.vision_ai import AIVisionSystem
from core.robot_controller import RobotController
from vision.camera import CameraSystem
from voice.voice_system import VoiceSystem


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

        # --- AI Dialogue settings (new) ---
        # Separate from "narration" so you can have either/both.
        self._dialogue_enabled = bool(vs.get("enabled", True)) and bool(vs.get("dialogue_enabled", True))
        self._dialogue_min_interval_s = float(vs.get("dialogue_min_interval_s", 10.0))
        self._last_dialogue_at = 0.0
        self._last_dialogue_text: Optional[str] = None
        self._dialogue_dedupe_window_s = float(vs.get("dialogue_dedupe_window_s", 30.0))
        self._last_dialogue_text_at = 0.0
        # ---------------------------------

        # --- Curiosity / anti-loop state ---
        self._recent_actions: Deque[str] = deque(maxlen=5)
        self._repeat_threshold = 3
        # ----------------------------------

    # -----------------------------------------------------

    def available_actions(self) -> List[str]:
        return ["forward", "turn_left", "turn_right", "backward", "stop"]

    def context(self) -> str:
        return f"mode={self.name} target={self.target}".strip()

    # -----------------------------------------------------
    # 🧠 Curiosity / anti-loop logic lives here
    # -----------------------------------------------------

    def postprocess_action(self, action: str) -> str:
        """
        Bias actions to avoid getting stuck in loops.
        Never overrides 'stop'.
        """
        if action == "stop":
            return action

        self._recent_actions.append(action)

        # Not enough history yet
        if len(self._recent_actions) < self._repeat_threshold:
            return action

        # Detect same-action repetition
        last_actions = list(self._recent_actions)[-self._repeat_threshold :]
        if all(a == action for a in last_actions):
            return self._curiosity_override(action)

        # Detect oscillation: left-right-left-right
        if len(self._recent_actions) >= 4:
            a, b, c, d = list(self._recent_actions)[-4:]
            if a == c and b == d and a != b:
                return self._curiosity_override(action)

        return action

    def _curiosity_override(self, stuck_action: str) -> str:
        """
        Choose a different action when stuck.
        """
        self.logger.debug(f"Curiosity triggered (stuck on '{stuck_action}')")

        # Prefer forward if we weren't already trying it
        if stuck_action != "forward":
            new_action = "forward"
        else:
            # Alternate turns
            new_action = "turn_left"

        # Narrate curiosity intervention
        self._narrate("I seem to be stuck. Trying something different.", level="normal")

        # Reset history slightly so we don't immediately re-trigger
        self._recent_actions.clear()
        self._recent_actions.append(new_action)

        return new_action

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
        This is throttled and deduped separately from narration.
        Safe: non-fatal if unavailable.
        """
        if not self._dialogue_enabled:
            return

        now = time.time()
        if (now - self._last_dialogue_at) < self._dialogue_min_interval_s:
            return

        # Dedupe identical line within window
        if (
            self._last_dialogue_text
            and (now - self._last_dialogue_text_at) < self._dialogue_dedupe_window_s
        ):
            # We'll still allow new dialogue; this block only prevents repeating the exact same line.
            pass

        # Only attempt if the AI system exposes the method (we’ll add it in ai/vision_ai.py next)
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

            if self._last_dialogue_text == line and (now - self._last_dialogue_text_at) < self._dialogue_dedupe_window_s:
                return

            # Speak it (force=False so VoiceSystem cooldown/dedupe still apply)
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

            if analysis.description in {"AI unavailable", "AI error"}:
                self._narrate("I can't reach my AI right now. Stopping.", level="normal", force=True)

            decision = self.ai.decide_action(
                analysis=analysis,
                mode=self.name,
                available_actions=self.available_actions(),
                target=self.target,
            )

            action = decision.get("action", "stop")
            action = self.postprocess_action(action)
            duration_s = float(decision.get("duration_s", 0.6))

            record = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode": self.name,
                "target": self.target,
                "image_path": path,
                "analysis": {
                    "description": analysis.description,
                    "objects": analysis.objects,
                    "hazards": analysis.hazards,
                    "suggested_actions": analysis.suggested_actions,
                    "processing_time_s": analysis.processing_time_s,
                },
                "decision": decision,
                "executed_action": action,
            }
            if bool(self.config.get("logging_settings", {}).get("save_decisions", True)):
                self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
                with self.decisions_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

            if self.verbose:
                self.logger.info(f"Seen: {analysis.description}")
                self.logger.info(f"Decision: {decision}")
                self.logger.info(f"Executing: {action} ({duration_s:.2f}s)")

            # Simple narration (short)
            if action == "forward":
                self._narrate("Clear. Moving forward.", level="normal")
            elif action == "turn_left":
                self._narrate("Turning left.", level="normal")
            elif action == "turn_right":
                self._narrate("Turning right.", level="normal")
            elif action == "backward":
                self._narrate("Backing up.", level="normal")
            elif action == "stop":
                self._narrate("Stopping.", level="normal")

            # NEW: AI-generated dialogue line (throttled)
            self._speak_dialogue(analysis, decision, action)

            self.robot.execute(action, duration_s)

        self.logger.info(f"Behavior '{self.name}' complete")
        self._narrate(f"{self.name} mode complete.", level="normal", force=True)