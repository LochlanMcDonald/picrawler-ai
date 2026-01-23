#!/usr/bin/env python3
"""
PiCrawler-AI v4 - Robust Layered Architecture

This version uses:
- Sensor fusion (WorldModel)
- Spatial memory (learning from history)
- Behavior trees (structured decision making)
- AI planner (high-level strategy)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.world_model import WorldModel
from core.spatial_memory import SpatialMemory
from core.robot_controller import RobotController
from perception.camera import CameraSystem, CameraPanicException
from perception.vision_ai import VisionAI
from planning.behavior_tree import (
    build_exploration_tree,
    build_cautious_exploration_tree,
    BehaviorContext,
    Status
)
from planning.ai_planner import AIPlanner


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging."""
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/operation_v4.log')
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        return {
            "robot_settings": {"obstacle_distance_threshold_cm": 20},
            "ai_settings": {"model": "gpt-4o-mini"}
        }

    with open(path) as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PiCrawler-AI v4 - Robust Architecture")
    p.add_argument("--mode", default="explore", choices=["explore", "cautious", "test"],
                  help="Operation mode")
    p.add_argument("--duration", type=float, default=5,
                  help="Run duration in minutes")
    p.add_argument("--config", default="config/config.json",
                  help="Path to config.json")
    p.add_argument("--verbose", action="store_true",
                  help="Verbose logging")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = logging.getLogger("main")

    logger.info("=" * 70)
    logger.info("PiCrawler-AI v4 - Robust Layered Architecture")
    logger.info("=" * 70)

    # Load configuration
    config = load_config(args.config)

    # Initialize core systems
    logger.info("Initializing systems...")

    try:
        # Hardware
        robot = RobotController(config)

        # Perception
        camera = CameraSystem(config)
        vision_ai = VisionAI(config)

        # Planning
        ai_planner = AIPlanner(config)

        # World model (sensor fusion)
        threshold = config.get("robot_settings", {}).get("obstacle_distance_threshold_cm", 20.0)
        world_model = WorldModel(obstacle_threshold_cm=threshold)

        # Spatial memory (learning)
        memory = SpatialMemory()

        logger.info("All systems initialized successfully")

        # Test mode: Just capture and analyze one frame
        if args.mode == "test":
            logger.info("Running test mode...")

            _, b64, path = camera.capture(save=True)
            if not b64:
                logger.error("Camera test failed")
                return 2

            # Update world model with sensor
            distance = robot.get_distance()
            world_model.update_ultrasonic(distance)

            # Get vision analysis
            analysis = vision_ai.analyze_scene(b64)

            if analysis:
                world_model.update_vision(
                    analysis.objects,
                    analysis.hazards,
                    analysis.description
                )

            logger.info(f"Image saved: {path}")
            logger.info(f"World model: {world_model}")
            logger.info(f"Analysis: {analysis.description if analysis else 'N/A'}")

            return 0

        # Build behavior tree based on mode
        if args.mode == "cautious":
            behavior_tree = build_cautious_exploration_tree()
            logger.info("Using cautious exploration behavior")
        else:
            behavior_tree = build_exploration_tree()
            logger.info("Using standard exploration behavior")

        # Create behavior context
        context = BehaviorContext(world_model, memory, robot, logger)

        # Main control loop
        end_time = time.time() + (args.duration * 60)
        capture_interval = 2.5
        last_capture = 0.0
        consecutive_failures = 0

        logger.info(f"Starting {args.duration} minute exploration...")

        while time.time() < end_time:
            loop_start = time.time()

            # === PERCEPTION LAYER (Fast) ===
            # Update sensors
            distance = robot.get_distance()
            world_model.update_ultrasonic(distance)

            # Camera capture (throttled)
            if loop_start - last_capture >= capture_interval:
                try:
                    _, b64, _ = camera.capture(save=args.verbose)

                    if b64:
                        # Vision analysis
                        analysis = vision_ai.analyze_scene(b64)

                        if analysis:
                            world_model.update_vision(
                                analysis.objects,
                                analysis.hazards,
                                analysis.description
                            )

                        last_capture = loop_start

                except CameraPanicException as e:
                    logger.warning(f"Camera panic: {e}")
                    robot.execute("stop", 0.5)
                    time.sleep(2.0)
                    continue

            # === BEHAVIOR LAYER (Medium) ===
            # Execute behavior tree
            logger.debug(f"World: {world_model}")
            logger.debug(f"Memory: {memory}")

            status = behavior_tree.execute(context)

            if status == Status.FAILURE:
                consecutive_failures += 1
                logger.warning(f"Behavior failed (consecutive: {consecutive_failures})")
            else:
                consecutive_failures = 0

            # === PLANNING LAYER (Slow, only when needed) ===
            # AI planner can override if stuck
            if ai_planner.should_replan(consecutive_failures):
                logger.info("Requesting new plan from AI...")

                plan = ai_planner.plan_exploration(world_model, memory)

                if plan:
                    logger.info(f"AI suggests: {plan.primary}")
                    logger.info(f"Reasoning: {plan.reasoning}")

                    # Could execute AI plan here, but for now just log it
                    # (behavior tree handles execution)

            # Small sleep to prevent tight loop
            time.sleep(0.1)

        logger.info("Exploration complete!")
        return 0

    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt - stopping")
        return 130

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    finally:
        # Cleanup
        logger.info("Shutting down...")
        try:
            robot.execute("stop", 0.2)
        except:
            pass

        try:
            camera.close()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
