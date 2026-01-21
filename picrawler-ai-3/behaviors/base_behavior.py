from __future__ import annotations

import inspect
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union

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

        # AI Dialogue settings
        self._dialogue_enabled = bool(vs.get("enabled", True)) and bool(vs.get("dialogue_enabled", True))
        self._dialogue_min_interval_s = float(vs.get("dialogue_min_interval_s", 10.0))
        self._last_dialogue_at = 0.0
        self._last_dialogue_text: Optional[str] = None
        self._dialogue_dedupe_window_s = float(vs.get("dialogue_dedupe_window_s", 30.0))
        self._last_dialogue_text_at = 0.0

        # --- NEW: "thoughts" narration (user requested: narrate all thoughts) ---
        self._thoughts_enabled = bool(vs.get("enabled", True)) and bool(vs.get("thoughts_enabled", True))
        # Set to 0.0 to speak every loop (can be loud/spammy; but it’s what you asked for)
        self._thoughts_min_interval_s = float(vs.get("thoughts_min_interval_s", 0.0))
        self._last_thoughts_at = 0.0
        self._thoughts_force_every_tick = bool(vs.get("thoughts_force_every_tick", True))
        # ----------------------------------------------------------------------

        # --- Behavior / anti-loop settings ---
        bs = (config.get("behavior_settings") or {}) if isinstance(config, dict) else {}
        # IMPORTANT: don't treat your entire behavior_settings blob as anti-loop settings
        # unless you explicitly provided behavior_settings.anti_loop
        bl = (bs.get("anti_loop") or {}) if isinstance(bs, dict) else {}

        # Action-loop tracking
        self._recent_actions: Deque[str] = deque(maxlen=int(bl.get("history", bl.get("anti_loop_history", 8))))
        self._repeat_threshold = int(bl.get("repeat_threshold", bl.get("anti_loop_repeat_threshold", 3)))
        self._oscillation_window = int(bl.get("oscillation_window", bl.get("anti_loop_oscillation_window", 4)))

        # Scene stagnation tracking (prevents "turn_right forever" in same view)
        self._recent_scene_sigs: Deque[str] = deque(maxlen=int(bl.get("scene_history", 10)))
        self._scene_repeat_threshold = int(bl.get("scene_repeat_threshold", 4))

        # Short “ban” on actions that keep failing (prevents immediate re-pick)
        self._banned_until: Dict[str, float] = {}
        self._ban_seconds = float(bl.get("ban_seconds", 8.0))

        # Turn-run limiter (helps when vision slightly changes but bot is still just rotating)
        self._consecutive_turns = 0
        self._max_consecutive_turns = int(bl.get("max_consecutive_turns", 5))

        # Escape ladder (escalates)
        self._escape_plan: List[Tuple[str, float]] = [
            ("stop", float(bl.get("escape_stop_s", 0.25))),
            ("backward", float(bl.get("escape_back_s", 1.25))),
            ("turn_left", float(bl.get("escape_turn_s", 1.10))),  # direction may flip dynamically
            ("forward", float(bl.get("escape_forward_s", 1.35))),
        ]
        self._escape_stage = 0
        self._last_escape_at = 0.0
        self._escape_cooldown_s = float(bl.get("escape_cooldown_s", 3.0))

        self._escape_strikes = 0
        self._max_strikes_before_reset = int(bl.get("escape_strikes_reset", 6))

    # -----------------------------------------------------

    def available_actions(self) -> List[str]:
        return ["forward", "turn_left", "turn_right", "backward", "stop"]

    def context(self) -> str:
        return f"mode={self.name} target={self.target}".strip()

    # -----------------------------------------------------
    # Anti-loop helpers
    # -----------------------------------------------------

    def _scene_signature(self, analysis) -> str:
        """
        Create a lightweight “same scene” signature.
        If this repeats, we’re not making progress.
        """
        desc = (getattr(analysis, "description", "") or "").strip().lower()
        hazards = getattr(analysis, "hazards", []) or []
        objs = getattr(analysis, "objects", []) or []

        h = ",".join(sorted([str(x).lower() for x in hazards])[:4])
        o = ",".join(sorted([str(x).lower() for x in objs])[:4])
        return f"{desc[:90]}|h:{h}|o:{o}"

    def _is_scene_stagnant(self) -> bool:
        if len(self._recent_scene_sigs) < self._scene_repeat_threshold:
            return False
        last = list(self._recent_scene_sigs)[-self._scene_repeat_threshold :]
        return len(set(last)) == 1

    def _is_repeating(self, action: str) -> bool:
        last_actions = list(self._recent_actions)[-self._repeat_threshold :]
        return len(last_actions) == self._repeat_threshold and all(a == action for a in last_actions)

    def _is_oscillating(self) -> bool:
        if len(self._recent_actions) < self._oscillation_window:
            return False
        a, b, c, d = list(self._recent_actions)[-4:]
        return a == c and b == d and a != b  # ABAB pattern

    def _ban(self, action: str, seconds: Optional[float] = None) -> None:
        self._banned_until[action] = time.time() + float(seconds if seconds is not None else self._ban_seconds)

    def _is_banned(self, action: str) -> bool:
        until = self._banned_until.get(action)
        return bool(until and time.time() < until)

    def _pick_non_banned_fallback(self, preferred: str) -> str:
        """
        Choose a safe alternative when the model picked a banned action.
        """
        order = {
            "turn_right": ["backward", "turn_left", "forward", "stop"],
            "turn_left": ["backward", "turn_right", "forward", "stop"],
            "forward": ["backward", "turn_left", "turn_right", "stop"],
            "backward": ["turn_left", "turn_right", "stop"],
        }
        for a in order.get(preferred, ["stop"]):
            if not self._is_banned(a):
                return a
        return "stop"

    # IMPORTANT: keyword-only analysis to avoid subclass signature crashes
    def postprocess_action(self, action: str, *, analysis=None, **_ignored) -> ActionChoice:
        """
        Returns either:
          - "action"
          - ("action", duration_override_s)

        `analysis` is keyword-only so subclasses can ignore it safely.
        `**_ignored` prevents crashes if callers add more keyword args later.
        """
        if action:
            self._recent_actions.append(action)

        # If banned, override immediately
        if action and self._is_banned(action):
            new_action = self._pick_non_banned_fallback(action)
            self.logger.debug(f"Action '{action}' is banned; overriding -> '{new_action}'")
            return (new_action, 1.0 if new_action in ("backward", "forward") else 0.9)

        # Honor stop
        if action == "stop":
            self._consecutive_turns = 0
            self._reset_escape_if_safe()
            return action

        # Track "turning too long" even if vision signature is changing slightly
        if action in ("turn_left", "turn_right"):
            self._consecutive_turns += 1
        else:
            self._consecutive_turns = 0

        if self._consecutive_turns >= self._max_consecutive_turns:
            # ban the current turn direction briefly; force a scene change
            self._ban(action, seconds=max(self._ban_seconds, 8.0))
            self._ban("turn_left", seconds=6.0)
            self._ban("turn_right", seconds=6.0)
            return self._escape(action, reason=f"turn_run>{self._max_consecutive_turns}")

        # Detect stagnation (same scene) if we have analysis
        if analysis is not None:
            self._recent_scene_sigs.append(self._scene_signature(analysis))
            if self._is_scene_stagnant():
                if action in ("turn_left", "turn_right"):
                    self._ban(action)
                return self._escape(action, reason="scene_stagnation")

        # Not enough history yet
        if len(self._recent_actions) < self._repeat_threshold:
            self._reset_escape_if_safe()
            return action

        # Repeating action
        if self._is_repeating(action):
            if action in ("turn_left", "turn_right"):
                self._ban(action)
            return self._escape(action, reason=f"repeat:{action}")

        # Oscillation
        if self._is_oscillating():
            self._ban("turn_left")
            self._ban("turn_right")
            return self._escape(action, reason="oscillation")

        self._reset_escape_if_safe()
        return action

    def _escape(self, stuck_action: str, *, reason: str) -> ActionChoice:
        now = time.time()

        # Cooldown: don't trigger escapes every single loop tick
        if (now - self._last_escape_at) < self._escape_cooldown_s:
            # During cooldown, force a move that changes the scene
            if stuck_action in ("turn_left", "turn_right"):
                return ("backward", 1.2)
            return ("forward", 1.2)

        self._last_escape_at = now
        self._escape_strikes += 1
        if self._escape_strikes >= self._max_strikes_before_reset:
            self._escape_strikes = 0
            self._escape_stage = 0

        stage = self._escape_stage % len(self._escape_plan)
        action, dur = self._escape_plan[stage]

        # Turn opposite if stuck turning
        if action in ("turn_left", "turn_right"):
            if stuck_action == "turn_right":
                action = "turn_left"
            elif stuck_action == "turn_left":
                action = "turn_right"
            else:
                action = "turn_left" if (stage % 2 == 0) else "turn_right"

        self._escape_stage += 1

        # Clear histories so we don’t re-trigger instantly
        self._recent_actions.clear()
        self._recent_actions.append(action)
        self._recent_scene_sigs.clear()
        self._consecutive_turns = 0

        self.logger.debug(f"Anti-loop escape ({reason}) -> {action} ({dur:.2f}s)")
        self._narrate("I’m stuck—backing out and trying a new angle.", level="normal")

        return (action, dur)

    def _reset_escape_if_safe(self) -> None:
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

    def _think(self, text: str, *, force: bool = False) -> None:
        """
        "Narrate all thoughts" channel.
        Separate throttle from _narrate so you can hear the brain every tick if desired.
        """
        if not self._thoughts_enabled:
            return
        now = time.time()
        if not force and not self._thoughts_force_every_tick:
            if (now - self._last_thoughts_at) < self._thoughts_min_interval_s:
                return
        if not force and self._thoughts_force_every_tick is False:
            # handled above
            pass
        # If thoughts_force_every_tick is True, we do not throttle unless you set thoughts_min_interval_s > 0
        if not force and self._thoughts_min_interval_s > 0.0:
            if (now - self._last_thoughts_at) < self._thoughts_min_interval_s:
                return

        try:
            self.voice.say(text, level="normal", force=False)
        except Exception:
            pass
        self._last_thoughts_at = now

    def _speak_dialogue(self, analysis, decision: dict, executed_action: str) -> None:
        if not self._dialogue_enabled:
            return

        now = time.time()
        if (now - self._last_dialogue_at) < self._dialogue_min_interval_s:
            return
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
            self._last_dialogue_at = now

    # -----------------------------------------------------
    # Safe postprocess dispatcher (prevents subclass signature crashes)
    # -----------------------------------------------------

    def _postprocess_action_safe(self, action: str, *, analysis=None) -> ActionChoice:
        """
        Calls self.postprocess_action in a way that will NOT crash if a subclass overrides
        postprocess_action(action: str) without accepting analysis / kwargs.
        """
        fn = getattr(self, "postprocess_action")
        try:
            sig = inspect.signature(fn)
            params = sig.parameters
            # if it accepts **kwargs, we're safe
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            accepts_analysis = ("analysis" in params) or has_varkw

            if accepts_analysis:
                return fn(action, analysis=analysis)  # type: ignore[misc]
            else:
                return fn(action)  # type: ignore[misc]
        except Exception:
            # fallback to BaseBehavior implementation
            return BaseBehavior.postprocess_action(self, action, analysis=analysis)

    # -----------------------------------------------------

    def run(self) -> None:
        end_t = time.time() + (self.duration_minutes * 60)
        self.logger.info(f"Starting behavior '{self.name}' for {self.duration_minutes} min")

        start_line = f"Starting {self.name} mode."
        if self.target:
            start_line = f"Starting {self.name} mode. Target: {self.target}."
        self._narrate(start_line, level="normal", force=True)
        self._think(f"Booting {self.name}. I will narrate my reasoning so you can tell if I'm stuck.", force=True)

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
                self._think("No camera frame. I will stop for safety.", force=True)
                self._narrate("Stopping.", level="normal")
                self.robot.execute("stop", 0.3)
                time.sleep(0.5)
                continue

            analysis = self.ai.analyze_scene(b64, context=self.context())

            if getattr(analysis, "description", "") in {"AI unavailable", "AI error"}:
                self._think("AI perception failed. I will stop and wait.", force=True)
                self._narrate("I can't reach my AI right now. Stopping.", level="normal", force=True)

            decision = self.ai.decide_action(
                analysis=analysis,
                mode=self.name,
                available_actions=self.available_actions(),
                target=self.target,
            )

            raw_action = decision.get("action", "stop")
            raw_duration_s = float(decision.get("duration_s", 0.6))
            reasoning = str(decision.get("reasoning", "") or "").strip()

            # Thought narration BEFORE anti-loop (so you can see what the model wanted)
            desc = (getattr(analysis, "description", "") or "").strip()
            hazards = getattr(analysis, "hazards", []) or []
            objects = getattr(analysis, "objects", []) or []
            hz = ", ".join([str(x) for x in hazards[:3]]) if hazards else "none"
            obj = ", ".join([str(x) for x in objects[:3]]) if objects else "none"
            self._think(
                f"I see: {desc[:120] or 'no description'}. Hazards: {hz}. Objects: {obj}. "
                f"Model wants: {raw_action} for {raw_duration_s:.1f}s."
                + (f" Reason: {reasoning[:140]}" if reasoning else "")
            )

            # Safe postprocess (won't crash even if a subclass has old signature)
            choice = self._postprocess_action_safe(raw_action, analysis=analysis)
            if isinstance(choice, tuple):
                action, duration_s = choice
            else:
                action, duration_s = choice, raw_duration_s

            # Thought narration AFTER anti-loop (so you can see overrides)
            if action != raw_action:
                banned_left = self._banned_until.get("turn_left", 0.0) - time.time()
                banned_right = self._banned_until.get("turn_right", 0.0) - time.time()
                bans = []
                if banned_left > 0:
                    bans.append(f"turn_left {banned_left:.0f}s")
                if banned_right > 0:
                    bans.append(f"turn_right {banned_right:.0f}s")
                bans_txt = ("; bans: " + ", ".join(bans)) if bans else ""
                self._think(f"Override: {raw_action} → {action} to prevent looping. Executing {action} for {duration_s:.1f}s.{bans_txt}")

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
                "raw_action": raw_action,
                "raw_duration_s": raw_duration_s,
                "executed_action": action,
                "executed_duration_s": duration_s,
                "banned": {k: round(v - time.time(), 2) for k, v in self._banned_until.items() if v > time.time()},
                "consecutive_turns": self._consecutive_turns,
            }
            if bool(self.config.get("logging_settings", {}).get("save_decisions", True)):
                self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
                with self.decisions_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

            if self.verbose:
                self.logger.info(f"Seen: {getattr(analysis, 'description', '')}")
                self.logger.info(f"Decision: {decision}")
                self.logger.info(f"Executing: {action} ({duration_s:.2f}s)")

            # Short narration (you can keep this ON, thoughts narration is separate)
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

            self._speak_dialogue(analysis, decision, action)

            self.robot.execute(action, duration_s)

        self.logger.info(f"Behavior '{self.name}' complete")
        self._think(f"{self.name} finished. Stopping motors.", force=True)
        self._narrate(f"{self.name} mode complete.", level="normal", force=True)