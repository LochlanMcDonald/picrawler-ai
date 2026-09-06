"""
Guided exploration — the robot looks, explains, proposes, asks, then acts.

One cycle:
  1. Look:     capture a frame, run depth + AI vision
  2. Explain:  say out loud what it sees (and any hazards)
  3. Decide:   ask the AI for ONE next action with a short reason
  4. Ask:      say "I'd like to … because …. Is that a good choice?"
  5. Listen:   wait for yes / no / an instruction / quit
  6. Act:      execute the approved action (with a physical safety check),
               or take the user's instruction, or propose an alternative

Everything the explorer touches is injected so the whole loop is unit-testable
without hardware, a camera, a speaker or the OpenAI API.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

VALID_ACTIONS = ("forward", "backward", "turn_left", "turn_right", "stop")

_PHRASES = {
    "forward": "walk forward",
    "backward": "back up",
    "turn_left": "turn left",
    "turn_right": "turn right",
    "stop": "stay here and look around",
}


@dataclass
class Decision:
    action: str
    reasoning: str
    confidence: float = 0.5
    source: str = "ai"          # "ai" | "rule" | "user"

    def phrase(self) -> str:
        return _PHRASES.get(self.action, self.action.replace("_", " "))


# ---------------------------------------------------------------------------
# AI decider
# ---------------------------------------------------------------------------

class GuidedDecider:
    """Asks the LLM for a single next action. Falls back to sensor rules."""

    def __init__(self, config: dict, client=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        ai = (config or {}).get("ai_settings", {})
        self.model = ai.get("model", "gpt-4o-mini")
        self.temperature = float(ai.get("temperature", 0.4))
        self.timeout_s = int(ai.get("request_timeout_s", 30))

        self.client = client
        if self.client is None:
            try:
                from openai import OpenAI
                api_key = (config or {}).get("openai_api_key")
                self.client = OpenAI(
                    api_key=None if api_key in (None, "", "your-api-key-here") else api_key
                )
            except Exception as e:
                self.logger.warning(f"No OpenAI client — rule-based decisions only ({e})")
                self.client = None

    def decide(self,
               world_state: Dict,
               scene_description: str,
               hazards: List[str],
               recent_actions: List[str],
               rejected: List[str]) -> Decision:
        if self.client is not None:
            d = self._ask_ai(world_state, scene_description, hazards, recent_actions, rejected)
            if d is not None:
                return d
        return self.rule_based(world_state, rejected)

    # -- fallback ---------------------------------------------------------

    @staticmethod
    def rule_based(world_state: Dict, rejected: List[str]) -> Decision:
        obstacles = world_state.get("obstacles", {}) if isinstance(world_state, dict) else {}
        best = world_state.get("best_direction", "forward") if isinstance(world_state, dict) else "forward"
        front_blocked = bool(obstacles.get("front"))

        order: List[str] = []
        if not front_blocked:
            order.append("forward")
        order += ["turn_left", "turn_right", "backward", "stop"]
        if best in ("left", "turn_left"):
            order.remove("turn_left"); order.insert(0 if front_blocked else 1, "turn_left")
        elif best in ("right", "turn_right"):
            order.remove("turn_right"); order.insert(0 if front_blocked else 1, "turn_right")

        for a in order:
            if a not in rejected:
                reason = ("the way ahead looks clear" if a == "forward"
                          else "something is close in front" if front_blocked
                          else "that looks like the most open direction")
                return Decision(a, reason, 0.4, source="rule")
        return Decision("stop", "every option has been ruled out", 0.2, source="rule")

    # -- LLM --------------------------------------------------------------

    def _ask_ai(self, world_state, scene_description, hazards, recent_actions, rejected) -> Optional[Decision]:
        obstacles = world_state.get("obstacles", {}) if isinstance(world_state, dict) else {}
        prompt = f"""You are a small legged robot exploring a room with a human supervisor.
Choose ONE next action and explain it in one short, friendly sentence that will be spoken aloud.

What the camera sees: {scene_description or "unknown"}
Hazards noticed: {hazards or "none"}
Physical sensors — obstacle front: {obstacles.get("front")}, left: {obstacles.get("left")}, right: {obstacles.get("right")}
Most open direction: {world_state.get("best_direction", "unknown") if isinstance(world_state, dict) else "unknown"}
Recent actions: {recent_actions[-6:] if recent_actions else "none"}
Actions the human already said NO to this turn: {rejected or "none"}

Rules: never choose forward if the front is blocked. Prefer exploring new directions
over repeating the last action. Never choose an action in the rejected list.

Return STRICT JSON only:
{{"action": "forward|backward|turn_left|turn_right|stop", "reasoning": "one short sentence", "confidence": 0.0-1.0}}
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                timeout=self.timeout_s,
            )
            text = resp.choices[0].message.content or ""
            data = _parse_json(text)
            if not data:
                self.logger.warning(f"AI returned non-JSON: {text[:120]!r}")
                return None
            action = str(data.get("action", "")).strip().lower().replace(" ", "_")
            if action == "left": action = "turn_left"
            if action == "right": action = "turn_right"
            if action not in VALID_ACTIONS or action in rejected:
                self.logger.warning(f"AI chose invalid/rejected action {action!r}")
                return None
            return Decision(
                action=action,
                reasoning=str(data.get("reasoning", "")).strip() or "it seems like a good idea",
                confidence=float(data.get("confidence", 0.5)),
                source="ai",
            )
        except Exception as e:
            self.logger.error(f"AI decision failed: {e}")
            return None


def _parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Answer interpretation
# ---------------------------------------------------------------------------

_YES = {"y", "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "go", "do it",
        "go ahead", "fine", "good", "yes please", "affirmative", "correct"}
_NO = {"n", "no", "nope", "nah", "don't", "dont", "negative", "stop", "wait"}
_QUIT = {"q", "quit", "exit", "bye", "goodbye", "shut down", "shutdown", "finish"}

_INSTRUCTION_WORDS = [
    (("backward", "back up", "reverse", "back"), "backward"),
    (("turn left", "left"), "turn_left"),
    (("turn right", "right"), "turn_right"),
    (("forward", "ahead", "straight", "go on", "keep going"), "forward"),
    (("stay", "hold", "look around", "don't move", "dont move"), "stop"),
]


@dataclass
class Answer:
    kind: str                       # "yes" | "no" | "quit" | "instruction" | "unknown" | "timeout"
    action: Optional[str] = None    # for "instruction"
    raw: str = ""


def interpret_answer(text: Optional[str]) -> Answer:
    if text is None:
        return Answer("timeout")
    raw = text.strip()
    t = re.sub(r"[^\w\s']", " ", raw.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    if not t:
        return Answer("timeout", raw=raw)
    if t in _QUIT:
        return Answer("quit", raw=raw)
    if t in _YES:
        return Answer("yes", raw=raw)
    if t in _NO:
        return Answer("no", raw=raw)
    # Instruction like "no, turn left instead" or just "left"
    for words, action in _INSTRUCTION_WORDS:
        if any(re.search(rf"\b{re.escape(w)}\b", t) for w in words):
            return Answer("instruction", action=action, raw=raw)
    first = t.split()[0]
    if first in _YES:
        return Answer("yes", raw=raw)
    if first in _NO:
        return Answer("no", raw=raw)
    return Answer("unknown", raw=raw)


# ---------------------------------------------------------------------------
# The explorer
# ---------------------------------------------------------------------------

class GuidedExplorer:
    def __init__(self, *,
                 camera, vision_ai, decider, voice, robot, world_model, memory, listener,
                 depth_estimator=None, config: Optional[dict] = None,
                 logger: Optional[logging.Logger] = None,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.camera = camera
        self.vision_ai = vision_ai
        self.decider = decider
        self.voice = voice
        self.robot = robot
        self.world_model = world_model
        self.memory = memory
        self.listener = listener
        self.depth_estimator = depth_estimator
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.clock = clock
        self.sleep = sleep

        g = (config or {}).get("guided_settings", {})
        bt = (config or {}).get("behavior_tree_settings", {})
        self.answer_timeout_s = float(g.get("answer_timeout_s", 30.0))
        self.max_reproposals = int(g.get("max_reproposals", 2))
        self.durations = {
            "forward": float(g.get("forward_duration_s", bt.get("move_forward_duration_s", 1.2))),
            "backward": float(g.get("backup_duration_s", bt.get("backup_duration_s", 1.0))),
            "turn_left": float(g.get("turn_duration_s", bt.get("turn_duration_s", 0.8))),
            "turn_right": float(g.get("turn_duration_s", bt.get("turn_duration_s", 0.8))),
            "stop": 0.2,
        }

        self.recent_actions: List[str] = []
        self.cycles = 0
        self.executed = 0

    # ------------------------------------------------------------------

    def say(self, text: str) -> None:
        self.logger.info(f"🗣  {text}")
        try:
            self.voice.say(text, level="normal", force=True)
        except Exception as e:
            self.logger.warning(f"Speech failed: {e}")

    def run(self, duration_min: float) -> int:
        self.say("Hello. I'll look around, tell you what I see, and ask before I move.")
        end = self.clock() + duration_min * 60.0
        try:
            while self.clock() < end:
                if not self.step():
                    break
        except KeyboardInterrupt:
            self.logger.info("Interrupted")
        self.say("Okay, I'm done for now.")
        try:
            self.robot.execute("stop", 0.2)
        except Exception:
            pass
        return 0

    # ------------------------------------------------------------------

    def step(self) -> bool:
        """One look → explain → propose → ask → act cycle. False = user quit."""
        self.cycles += 1

        # 1. Look
        description, hazards = self._look()

        # 2. Explain
        if description:
            self.say(f"I see {description}")
        else:
            self.say("I can't make out much right now.")
        if hazards:
            self.say(f"I should be careful of {self._join(hazards)}.")

        # 3-6. Propose / ask / act, with a few re-proposals if the user says no
        rejected: List[str] = []
        for attempt in range(self.max_reproposals + 1):
            decision = self.decider.decide(
                self.world_model.to_dict(), description, hazards, self.recent_actions, rejected
            )
            if attempt == 0:
                self.say(f"I'd like to {decision.phrase()} because {decision.reasoning}. Is that a good choice?")
            else:
                self.say(f"How about I {decision.phrase()} instead, because {decision.reasoning}? Is that okay?")

            answer = interpret_answer(self.listener.ask("Your answer (yes / no / left / right / forward / back / quit): ",
                                                        self.answer_timeout_s))
            self.logger.info(f"Answer: {answer.kind} {answer.action or ''} ({answer.raw!r})")

            if answer.kind == "quit":
                return False
            if answer.kind == "yes":
                self._act(decision)
                return True
            if answer.kind == "instruction":
                self.say(f"Okay, I'll {_PHRASES.get(answer.action, answer.action)}.")
                self._act(Decision(answer.action, "you asked me to", 1.0, source="user"))
                return True
            if answer.kind == "timeout":
                self.say("I didn't hear an answer, so I'll stay put and look again.")
                return True
            if answer.kind == "unknown":
                self.say("I didn't catch that. I'll take it as a no.")
            # "no" or "unknown": mark rejected and re-propose
            rejected.append(decision.action)

        self.say("Alright, I'll stay here for now. What would you like me to do?")
        answer = interpret_answer(self.listener.ask("Instruction (forward / back / left / right / stay / quit): ",
                                                    self.answer_timeout_s))
        if answer.kind == "quit":
            return False
        if answer.kind == "instruction":
            self.say(f"Okay, I'll {_PHRASES.get(answer.action, answer.action)}.")
            self._act(Decision(answer.action, "you asked me to", 1.0, source="user"))
        else:
            self.say("Okay, I'll have another look.")
        return True

    # ------------------------------------------------------------------

    def _look(self):
        description, hazards = "", []
        try:
            distance = self.robot.get_distance()
            self.world_model.update_ultrasonic(distance)
        except Exception as e:
            self.logger.debug(f"Ultrasonic read failed: {e}")

        try:
            image, b64, _ = self.camera.capture(save=False)
        except Exception as e:
            self.logger.warning(f"Camera capture failed: {e}")
            return description, hazards
        if image is None:
            return description, hazards

        if self.depth_estimator is not None:
            try:
                depth = self.depth_estimator.estimate_depth(image)
                if depth:
                    self.world_model.update_depth(depth)
            except Exception as e:
                self.logger.debug(f"Depth failed: {e}")

        if b64 and self.vision_ai is not None:
            analysis = self.vision_ai.analyze_scene(b64)
            if analysis:
                description = (analysis.description or "").strip().rstrip(".")
                hazards = [h for h in (analysis.hazards or []) if h and h != "unknown"]
                try:
                    self.world_model.update_vision(analysis.objects, analysis.hazards, analysis.description)
                except Exception as e:
                    self.logger.debug(f"World model vision update failed: {e}")
                if description.lower().startswith("vision unavailable"):
                    description = ""
        return description, hazards

    def _act(self, decision: Decision) -> None:
        action = decision.action
        duration = self.durations.get(action, 0.5)

        # Physical safety check — the AI's opinion never overrides the sensors
        if action == "forward":
            blocked = False
            try:
                blocked = bool(self.robot.has_obstacle())
            except Exception:
                pass
            try:
                blocked = blocked or not self.world_model.is_safe_to_move("forward")
            except Exception:
                pass
            if blocked:
                self.say("Actually, something is too close in front, so I'll stop instead.")
                action, duration = "stop", 0.2

        before = self._distance()
        ok = True
        try:
            self.robot.execute(action, duration)
        except Exception as e:
            ok = False
            self.logger.error(f"Action {action} failed: {e}")
            self.say("Hmm, that didn't work.")
        after = self._distance()

        self.recent_actions.append(action)
        self.recent_actions = self.recent_actions[-20:]
        self.executed += 1
        try:
            self.memory.record_action(action, ok, before, after, reason=f"guided:{decision.source}")
        except Exception:
            pass

    def _distance(self) -> Optional[float]:
        try:
            return self.robot.get_distance()
        except Exception:
            return None

    @staticmethod
    def _join(items: List[str]) -> str:
        items = [str(i) for i in items if i]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " and " + items[-1]
