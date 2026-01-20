from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from ai.vision_ai import AIVisionSystem, SceneAnalysis
from core.robot_controller import RobotController
from vision.camera import CameraSystem


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

    def run(self) -> None:
        end_t = time.time() + (self.duration_minutes * 60)
        self.logger.info(f"Starting behavior '{self.name}' for {self.duration_minutes} min")

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
                self.robot.execute("stop", 0.3)
                time.sleep(0.5)
                continue

            analysis = self.ai.analyze_scene(b64, context=self.context())
            decision = self.ai.decide_action(
                analysis=analysis,
                mode=self.name,
                available_actions=self.available_actions(),
                target=self.target,
            )

            # --- NEW: behavior-level action postprocessing ---
            action = decision.get("action", "stop")
            action = self.postprocess_action(action)
            duration_s = float(decision.get("duration_s", 0.6))
            # -------------------------------------------------

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

            self.robot.execute(action, duration_s)

        self.logger.info(f"Behavior '{self.name}' complete")