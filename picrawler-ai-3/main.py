#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import sys

from core.config_loader import load_config
from core.logger import setup_logging
from core.robot_controller import RobotController
from vision.camera import CameraSystem
from ai.vision_ai import AIVisionSystem
from behaviors.exploration import ExplorationBehavior
from behaviors.object_detection import ObjectDetectionBehavior
from behaviors.following import FollowingBehavior
from behaviors.avoidance import AvoidanceBehavior


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PiCrawler-AI (OpenAI vision)")
    p.add_argument("--mode", required=True, choices=["explore", "detect", "follow", "avoid", "test"], help="Operation mode")
    p.add_argument("--target", default=None, help="Target object/person for detect/follow/avoid")
    p.add_argument("--duration", type=float, default=5, help="Run duration in minutes")
    p.add_argument("--config", default="config/config.json", help="Path to config.json")
    p.add_argument("--verbose", action="store_true", help="Verbose console logging")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    config = load_config(args.config)

    log_level = config.get("logging_settings", {}).get("log_level", "INFO")
    if args.verbose:
        log_level = "DEBUG"
    setup_logging(log_level=log_level)
    logger = logging.getLogger("main")

    # Warn if key missing (but allow dry-run test)
    api_key = config.get("openai_api_key")
    if (not api_key or api_key == "your-api-key-here") and not os.getenv("OPENAI_API_KEY"):
        logger.warning("No OpenAI API key set. Set OPENAI_API_KEY env var or config/openai_api_key. AI calls will fail.")

    # --- Quick improvement #1: Throttle AI calls to reduce cost/jitter ---
    # Default to 3 seconds, overridable via config:
    # "ai_settings": { "min_seconds_between_calls": 3.0 }
    ai_settings = config.get("ai_settings", {})
    min_seconds_between_calls = float(ai_settings.get("min_seconds_between_calls", 3.0))
    AIVisionSystem.MIN_SECONDS_BETWEEN_CALLS = min_seconds_between_calls  # type: ignore[attr-defined]
    # -------------------------------------------------------------------

    robot = RobotController(config)
    camera = CameraSystem(config)
    ai = AIVisionSystem(config)

    try:
        if args.mode == "test":
            # Simple self-test: capture one frame and run one analysis
            _, b64, path = camera.capture(save=True)
            if not b64:
                logger.error("Camera test failed (no frame)")
                return 2
            analysis = ai.analyze_scene(b64, context="self-test")
            logger.info(f"Saved frame: {path}")
            logger.info(f"Analysis: {analysis.description} | objects={analysis.objects} | hazards={analysis.hazards}")
            return 0

        behaviors = {
            "explore": ExplorationBehavior,
            "detect": ObjectDetectionBehavior,
            "follow": FollowingBehavior,
            "avoid": AvoidanceBehavior,
        }
        behavior_cls = behaviors[args.mode]
        behavior = behavior_cls(
            config=config,
            robot=robot,
            camera=camera,
            ai=ai,
            duration_minutes=args.duration,
            target=args.target,
            verbose=args.verbose,
        )
        behavior.run()
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted; stopping")
        try:
            robot.execute("stop", 0.2)
        except Exception:
            pass
        return 130

    finally:
        try:
            camera.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
