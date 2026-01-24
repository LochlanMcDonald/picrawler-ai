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
from perception.depth_estimator import DepthEstimator
from planning.behavior_tree import (
    build_exploration_tree,
    build_cautious_exploration_tree,
    BehaviorContext,
    Status
)
from planning.ai_planner import AIPlanner
from planning.language_controller import LanguageController
from mapping.slam_controller import SLAMController


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
    p.add_argument("--mode", default="explore",
                  choices=["explore", "cautious", "test", "language", "interactive", "slam", "slam_explore", "navigate"],
                  help="Operation mode")
    p.add_argument("--duration", type=float, default=5,
                  help="Run duration in minutes")
    p.add_argument("--config", default="config/config.json",
                  help="Path to config.json")
    p.add_argument("--verbose", action="store_true",
                  help="Verbose logging")
    p.add_argument("--command", type=str,
                  help="Natural language command for language mode")
    p.add_argument("--goal", type=str,
                  help="Navigation goal coordinates as 'x,y' (e.g., '1.5,0.8')")
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
        depth_estimator = DepthEstimator(
            model_type="MiDaS_small",  # Pi-optimized
            input_size=256,  # Lower resolution for speed
            cache_duration_s=1.0  # Update depth every second
        )

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

            image, b64, path = camera.capture(save=True)
            if not b64:
                logger.error("Camera test failed")
                return 2

            # Update world model with sensors
            distance = robot.get_distance()
            world_model.update_ultrasonic(distance)

            # Estimate depth from image
            logger.info("Estimating depth...")
            depth_map = depth_estimator.estimate_depth(image, force_refresh=True)
            if depth_map:
                world_model.update_depth(depth_map)
                logger.info(f"Depth inference: {depth_map.inference_time_ms:.1f}ms")
                logger.info(f"Directional depths: {depth_map.get_directional_depths()}")

                # Save depth visualization
                import cv2
                depth_viz = depth_estimator.visualize_depth(depth_map)
                depth_path = path.replace('.jpg', '_depth.jpg')
                cv2.imwrite(depth_path, depth_viz)
                logger.info(f"Depth map saved: {depth_path}")

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

        # Language mode: Execute natural language commands
        if args.mode in ["language", "interactive"]:
            logger.info("=" * 70)
            logger.info("Language Control Mode - Vision-Language-Action System")
            logger.info("=" * 70)

            language_controller = LanguageController(config)

            if args.mode == "language":
                # Single command mode
                if not args.command:
                    logger.error("--command required for language mode")
                    logger.info("Example: python main.py --mode language --command 'turn left and explore'")
                    return 1

                logger.info(f"Command: {args.command}")

                # Capture scene
                logger.info("Capturing scene...")
                image, b64, path = camera.capture(save=True)
                if not b64:
                    logger.error("Camera capture failed")
                    return 2

                # Update world model
                distance = robot.get_distance()
                world_model.update_ultrasonic(distance)

                # Get depth
                depth_map = depth_estimator.estimate_depth(image)
                depth_info = None
                if depth_map:
                    world_model.update_depth(depth_map)
                    depth_info = depth_map.get_directional_depths()

                # Understand scene
                logger.info("Understanding scene...")
                scene = language_controller.understand_scene(b64, depth_info)
                if scene:
                    logger.info(f"Scene: {scene.spatial_layout}")
                    logger.info(f"Objects: {', '.join(scene.objects)}")
                    logger.info(f"Navigable: {', '.join(scene.navigable_areas)}")

                # Parse command
                logger.info("Parsing command...")
                world_state = {
                    'front_distance': distance,
                    'free_space': world_model.free_space_score,
                    'best_direction': world_model.get_best_direction()
                }
                command = language_controller.parse_command(args.command, scene, world_state)

                if not command:
                    logger.error("Failed to parse command")
                    return 1

                if command.confidence < 0.5:
                    logger.warning(f"Low confidence ({command.confidence:.2f}) - may not be safe")
                    logger.warning(f"Reasoning: {command.reasoning}")
                    return 1

                # Execute command
                logger.info("Executing command...")
                success = language_controller.execute_command(command, robot, logger)

                if success:
                    logger.info("Command completed successfully!")
                    return 0
                else:
                    logger.error("Command execution failed")
                    return 1

            elif args.mode == "interactive":
                # Interactive mode - keep asking for commands
                logger.info("Interactive mode - type 'quit' to exit")
                logger.info("Example commands:")
                logger.info("  - 'turn left and move forward'")
                logger.info("  - 'find a clear path'")
                logger.info("  - 'explore to the right'")
                logger.info("  - 'back up and turn around'")
                logger.info("")

                while True:
                    try:
                        # Get command from user
                        user_input = input("Command> ").strip()

                        if user_input.lower() in ['quit', 'exit', 'q']:
                            logger.info("Exiting interactive mode")
                            break

                        if not user_input:
                            continue

                        # Capture fresh scene
                        logger.info("Capturing scene...")
                        image, b64, path = camera.capture(save=args.verbose)
                        if not b64:
                            logger.error("Camera capture failed")
                            continue

                        # Update sensors
                        distance = robot.get_distance()
                        world_model.update_ultrasonic(distance)

                        # Get depth
                        depth_map = depth_estimator.estimate_depth(image)
                        depth_info = None
                        if depth_map:
                            world_model.update_depth(depth_map)
                            depth_info = depth_map.get_directional_depths()

                        # Understand scene
                        scene = language_controller.understand_scene(b64, depth_info)
                        if scene:
                            logger.info(f"I see: {', '.join(scene.objects[:3])}")

                        # Parse and execute
                        world_state = {
                            'front_distance': distance,
                            'free_space': world_model.free_space_score,
                            'best_direction': world_model.get_best_direction()
                        }
                        command = language_controller.parse_command(user_input, scene, world_state)

                        if command and command.confidence >= 0.5:
                            logger.info(f"Plan: {' -> '.join(command.actions)}")
                            language_controller.execute_command(command, robot, logger)
                        else:
                            logger.warning("Could not understand or safely execute that command")
                            if command:
                                logger.warning(f"Reason: {command.reasoning}")

                    except KeyboardInterrupt:
                        logger.info("\nExiting interactive mode")
                        break
                    except EOFError:
                        logger.info("\nExiting interactive mode")
                        break

                return 0

        # SLAM mode: Build map while exploring
        if args.mode in ["slam", "slam_explore"]:
            logger.info("=" * 70)
            logger.info("SLAM Mode - Simultaneous Localization and Mapping")
            logger.info("=" * 70)

            slam_controller = SLAMController(map_size_m=10.0, resolution_m=0.05)

            # Create behavior context for exploration (if slam_explore)
            if args.mode == "slam_explore":
                behavior_tree = build_exploration_tree()
                context = BehaviorContext(world_model, memory, robot, logger)

            end_time = time.time() + (args.duration * 60)
            capture_interval = 1.0  # SLAM needs frequent updates
            last_capture = 0.0
            last_action = None
            frame_count = 0

            logger.info(f"Starting {args.duration} minute SLAM session...")
            logger.info(f"Map will be saved to logs/slam_map_final.jpg")

            while time.time() < end_time:
                loop_start = time.time()

                # Capture frame
                if loop_start - last_capture >= capture_interval:
                    try:
                        image, b64, img_path = camera.capture(save=False)

                        if image is not None:
                            # Get depth estimation
                            depth_map = depth_estimator.estimate_depth(image)

                            # Process with SLAM
                            pose, map_vis = slam_controller.process_frame(
                                image, depth_map, action_hint=last_action
                            )

                            frame_count += 1

                            # Log pose every 10 frames
                            if frame_count % 10 == 0:
                                logger.info(f"Pose: {pose}")
                                stats = slam_controller.get_statistics()
                                logger.info(f"Map: {stats['map']['explored_percent']:.1f}% explored")

                            # Save map visualization periodically
                            if frame_count % 50 == 0:
                                slam_controller.save_map(f'logs/slam_map_{frame_count}.jpg')

                            last_capture = loop_start

                    except Exception as e:
                        logger.error(f"SLAM frame processing failed: {e}")

                # Execute behavior if in slam_explore mode
                if args.mode == "slam_explore":
                    # Update sensors
                    distance = robot.get_distance()
                    world_model.update_ultrasonic(distance)

                    # Execute behavior tree
                    status = behavior_tree.execute(context)

                    # Track last action for odometry hint
                    # This is a simplification - in real code, track actual executed action
                    last_action = "forward"  # Placeholder

                else:
                    # Pure SLAM mode - just capture and map, no movement
                    time.sleep(0.5)

                # Small sleep
                time.sleep(0.1)

            # Save final map
            logger.info("SLAM session complete!")
            slam_controller.save_map('logs/slam_map_final.jpg')

            stats = slam_controller.get_statistics()
            logger.info("Final statistics:")
            logger.info(f"  Total distance: {stats['odometry']['total_distance_m']:.2f}m")
            logger.info(f"  Total rotation: {stats['odometry']['total_rotation_deg']:.1f}°")
            logger.info(f"  Map explored: {stats['map']['explored_percent']:.1f}%")
            logger.info(f"  Frames processed: {frame_count}")

            return 0

        # Navigate mode: Waypoint navigation using SLAM
        if args.mode == "navigate":
            logger.info("=" * 70)
            logger.info("Navigate Mode - Waypoint Navigation with SLAM")
            logger.info("=" * 70)

            if not args.goal:
                logger.error("--goal required for navigate mode")
                logger.info("Example: python main.py --mode navigate --goal '1.5,0.8'")
                return 1

            # Parse goal coordinates
            try:
                goal_parts = args.goal.split(',')
                goal_x = float(goal_parts[0])
                goal_y = float(goal_parts[1])
            except (ValueError, IndexError):
                logger.error(f"Invalid goal format: {args.goal}. Use 'x,y' format (e.g., '1.5,0.8')")
                return 1

            logger.info(f"Navigation goal: ({goal_x:.2f}, {goal_y:.2f})")

            slam_controller = SLAMController(map_size_m=10.0, resolution_m=0.05)

            # Build initial map first
            logger.info("Building initial map (exploring for 30 seconds)...")
            behavior_tree = build_exploration_tree()
            context = BehaviorContext(world_model, memory, robot, logger)

            explore_end_time = time.time() + 30  # 30 seconds of exploration
            last_capture = 0.0

            while time.time() < explore_end_time:
                loop_start = time.time()

                # Capture and process with SLAM
                if loop_start - last_capture >= 1.0:
                    image, b64, _ = camera.capture(save=False)
                    if image is not None:
                        depth_map = depth_estimator.estimate_depth(image)
                        pose, map_vis = slam_controller.process_frame(image, depth_map, action_hint="forward")
                        last_capture = loop_start

                # Update sensors and explore
                distance = robot.get_distance()
                world_model.update_ultrasonic(distance)
                behavior_tree.execute(context)

                time.sleep(0.1)

            # Plan path to goal
            logger.info("Planning path to goal...")
            path = slam_controller.plan_path_to_goal(goal_x, goal_y)

            if not path:
                logger.error("Could not find path to goal")
                logger.info("Map may not be sufficiently explored or goal is unreachable")
                slam_controller.save_map('logs/navigate_failed_map.jpg')
                return 1

            logger.info(f"Path planned with {len(path)} waypoints")
            slam_controller.save_map('logs/navigate_planned_path.jpg')

            # Follow the path
            logger.info("Following path...")
            nav_start_time = time.time()
            max_nav_time = 300  # 5 minutes max

            while slam_controller.is_navigating() and (time.time() - nav_start_time) < max_nav_time:
                loop_start = time.time()

                # Update SLAM
                if loop_start - last_capture >= 1.0:
                    image, b64, _ = camera.capture(save=False)
                    if image is not None:
                        depth_map = depth_estimator.estimate_depth(image)
                        pose, map_vis = slam_controller.process_frame(image, depth_map)
                        last_capture = loop_start

                # Check for obstacles
                distance = robot.get_distance()
                obstacle_detected = distance is not None and distance < 20.0

                # Get navigation command
                nav_cmd = slam_controller.get_navigation_command(obstacle_detected)

                if nav_cmd:
                    logger.info(f"Nav command: {nav_cmd.action} for {nav_cmd.duration:.1f}s ({nav_cmd.reason})")

                    if nav_cmd.action == "stop":
                        robot.execute("stop", nav_cmd.duration)
                        if "goal_reached" in nav_cmd.reason:
                            logger.info("Navigation complete - goal reached!")
                            break
                        elif "obstacle" in nav_cmd.reason or "blocked" in nav_cmd.reason:
                            logger.warning("Navigation blocked by obstacle")
                            # Try to replan
                            logger.info("Attempting to replan...")
                            new_path = slam_controller.plan_path_to_goal(goal_x, goal_y)
                            if new_path:
                                logger.info(f"Replanned with {len(new_path)} waypoints")
                            else:
                                logger.error("Replan failed - cannot reach goal")
                                break
                        else:
                            logger.error(f"Navigation failed: {nav_cmd.reason}")
                            break
                    else:
                        robot.execute(nav_cmd.action, nav_cmd.duration)

                # Log progress
                progress = slam_controller.get_navigation_progress()
                if loop_start % 5 < 0.2:  # Every ~5 seconds
                    logger.info(f"Progress: {progress['progress_percent']:.1f}% ({progress['waypoints_reached']}/{progress['waypoints_total']} waypoints)")

                time.sleep(0.1)

            # Save final map
            slam_controller.save_map('logs/navigate_final_map.jpg')

            if slam_controller.waypoint_navigator.is_complete():
                logger.info("Navigation succeeded!")
                return 0
            else:
                logger.error("Navigation failed or timed out")
                return 1

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
                    image, b64, img_path = camera.capture(save=args.verbose)

                    if b64:
                        # Depth estimation (runs every capture due to cache)
                        depth_map = depth_estimator.estimate_depth(image)
                        if depth_map:
                            world_model.update_depth(depth_map)
                            logger.debug(f"Depth: {depth_map.inference_time_ms:.1f}ms")

                            # Save depth visualization in verbose mode
                            if args.verbose and img_path:
                                import cv2
                                depth_viz = depth_estimator.visualize_depth(depth_map)
                                depth_path = img_path.replace('.jpg', '_depth.jpg')
                                cv2.imwrite(depth_path, depth_viz)

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
