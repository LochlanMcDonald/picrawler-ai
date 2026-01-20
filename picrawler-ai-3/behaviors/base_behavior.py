from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from ai.vision_ai import AIVisionSystem, SceneAnalysis
from core.robot_controller import RobotController
from vision.camera import CameraSystem

# Voice (OpenAI TTS) — new module you’ll add at voice/voice_system.py
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

        # Voice system (safe: will no-op if disabled or if TTS fails)
        self.voice = VoiceSystem(config)

        # Narration throttling (separate from VoiceSystem cooldown)
        vs = config.get("voice_settings", {}) if isinstance(config, dict) else {}
        self._narration_enabled = bool(vs.get("enabled", True)) and bool(vs.get("narration_enabled", True))
        self._narration_min_interval_s = float(vs.get("narration_min_interval_s", 4.0))
        self._last_narration_at = 0.0

    def available_actions(self) -> List[str]:
        return ["forward", "turn_left", "turn_right", "backward", "stop"]

    def context(self) -> str:
        return f"mode={self.name} target={self.target}".strip()

    def postprocess_action(self, action: str) -> str:
        """
        Optional hook for behaviors to bias or modify actions
        (e.g., anti-loop, curiosity bias, safety filters).
        """
        return action

    def _narrate(self, text: str, *, level: str = "normal", force: bool = False) -> None:
        """Narration helper with extra rate-limit on top of VoiceSystem."""
        if not self._narration_enabled:
            return
        now = time.time()
        if not force and (now - self._last_narration_at) < self._narration_min_interval_s:
            return
        self.voice.say(text, level=level, force=force)
        self._last_narration_at = now

    def run(self) -> None:
        end_t = time.time() + (self.duration_minutes * 60)
        self.logger.info(f"Starting behavior '{self.name}' for {self.duration_minutes} min")

        # Narrate start (force so you always hear mode changes)
        mode_line = f"Starting {self.name} mode."
        if self.target:
            mode_line = f"Starting {self.name} mode. Target: {self.target}."
        self._narrate(mode_line, level="normal", force=True)

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

            # If AI fell back, it will set description like "AI unavailable" / "AI error"
            if analysis.description in {"AI unavailable", "AI error"}:
                self._narrate("I can't reach my AI right now. Stopping.", level="normal", force=True)

            decision = self.ai.decide_action(
                analysis=analysis,
                mode=self.name,
                available_actions=self.available_actions(),
                target=self.target,
            )

            # --- behavior-level action postprocessing ---
            action = decision.get("action", "stop")
            action = self.postprocess_action(action)
            duration_s = float(decision.get("duration_s", 0.6))
            # -----------------------------------------

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

            # Narration: action + a tiny bit of “thinking” (throttled)
            # Keep it short and non-annoying.
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

            self.robot.execute(action, duration_s)

        self.logger.info(f"Behavior '{self.name}' complete")
        self._narrate(f"{self.name} mode complete.", level="normal", force=True)
