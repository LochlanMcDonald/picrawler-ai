"""OpenAI vision + decision helper.

Uses the OpenAI Responses API (text + image inputs).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from voice.voice_system import VoiceSystem


@dataclass
class SceneAnalysis:
    description: str
    objects: List[str]
    hazards: List[str]
    suggested_actions: List[str]
    raw: str
    processing_time_s: float


class AIVisionSystem:
    # Default throttle (seconds); can be overridden via config or main.py
    MIN_SECONDS_BETWEEN_CALLS: float = 3.0

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config  # keep for dialogue/personality access

        api_key = config.get("openai_api_key") or None
        # Also allow OPENAI_API_KEY env var automatically via SDK
        self.client = OpenAI(api_key=None if (api_key in (None, "", "your-api-key-here")) else api_key)

        ai = config.get("ai_settings", {})
        self.model = ai.get("model", "gpt-4.1-mini")
        self.max_output_tokens = int(ai.get("max_output_tokens", 800))
        self.temperature = float(ai.get("temperature", 0.4))
        self.timeout_s = int(ai.get("request_timeout_s", 30))

        # Throttling state
        self._last_call_time: float = 0.0
        self._last_analysis: Optional[SceneAnalysis] = None

        # Allow config override
        self.MIN_SECONDS_BETWEEN_CALLS = float(
            ai.get("min_seconds_between_calls", self.MIN_SECONDS_BETWEEN_CALLS)
        )

        # Voice system (safe: will no-op if disabled or if TTS fails)
        self.voice = VoiceSystem(config)

        # Failure narration suppression
        vs = config.get("voice_settings", {}) if isinstance(config, dict) else {}
        self._ai_fail_speak_cooldown_s = float(vs.get("ai_failure_speak_cooldown_s", 25.0))
        self._last_ai_fail_spoken_at: float = 0.0

        # ---- AI dialogue throttle/dedupe state (new) ----
        self._last_dialogue_at: float = 0.0
        self._last_dialogue_text: Optional[str] = None
        self._last_dialogue_text_at: float = 0.0
        # -----------------------------------------------

    # -----------------------------------------------------

    def _retry_api_call(self, func, max_retries: int = 2, delay_s: float = 1.0):
        """Retry API call with exponential backoff for transient failures."""
        for attempt in range(max_retries + 1):
            try:
                return func()
            except (APITimeoutError, RateLimitError) as e:
                if attempt < max_retries:
                    self.logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    time.sleep(delay_s * (2 ** attempt))  # Exponential backoff
                else:
                    raise  # Re-raise on final attempt
            except Exception:
                # Don't retry non-transient errors
                raise

    # -----------------------------------------------------

    def _speak_ai_failure_once(self, line: str) -> None:
        """Speak an AI-failure line, but suppress repeats for a cooldown window."""
        now = time.time()
        if now - self._last_ai_fail_spoken_at < self._ai_fail_speak_cooldown_s:
            return
        try:
            self.voice.say(line, level="normal")
        except Exception:
            pass
        self._last_ai_fail_spoken_at = now

    # -----------------------------------------------------
    # Vision analysis
    # -----------------------------------------------------

    def analyze_scene(self, image_base64: str, context: str) -> SceneAnalysis:
        """Return a structured summary of what the robot sees.

        Throttles calls to the OpenAI API and reuses the last analysis if called too soon.
        Falls back safely on API failure.
        """
        now = time.time()
        elapsed = now - self._last_call_time

        if self._last_analysis is not None and elapsed < self.MIN_SECONDS_BETWEEN_CALLS:
            self.logger.debug(
                f"AI throttled: reusing last analysis ({elapsed:.2f}s < {self.MIN_SECONDS_BETWEEN_CALLS:.2f}s)"
            )
            return self._last_analysis

        prompt = (
            "You are a robotics perception module. Summarize what you see for navigation. "
            "Be concise, grounded, and action-oriented.\n\n"
            f"Task context: {context}\n\n"
            "Return STRICT JSON with keys: description (string), objects (array of strings), "
            "hazards (array of strings), suggested_actions (array of strings)."
        )

        t0 = time.time()
        try:
            # Retry API call for transient failures
            def api_call():
                return self.client.responses.create(
                    model=self.model,
                    max_output_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{image_base64}",
                                },
                            ],
                        }
                    ],
                    timeout=self.timeout_s,
                )

            resp = self._retry_api_call(api_call, max_retries=2)
            text = getattr(resp, "output_text", "") or ""

        except (RateLimitError, APITimeoutError, APIError) as e:
            self.logger.error(f"OpenAI API failure: {e}")
            self._speak_ai_failure_once("I can't reach my AI right now. Going safe.")
            analysis = SceneAnalysis(
                description="AI unavailable",
                objects=[],
                hazards=[],
                suggested_actions=["stop"],
                raw=str(e),
                processing_time_s=time.time() - t0,
            )
            self._last_call_time = now
            self._last_analysis = analysis
            return analysis

        except Exception as e:
            self.logger.exception("Unexpected error during AI analysis")
            self._speak_ai_failure_once("My perception failed. Stopping to be safe.")
            analysis = SceneAnalysis(
                description="AI error",
                objects=[],
                hazards=[],
                suggested_actions=["stop"],
                raw=str(e),
                processing_time_s=time.time() - t0,
            )
            self._last_call_time = now
            self._last_analysis = analysis
            return analysis

        dt = time.time() - t0
        data = self._safe_json(text)

        analysis = SceneAnalysis(
            description=str(data.get("description", "")) if isinstance(data, dict) else "",
            objects=list(data.get("objects", [])) if isinstance(data, dict) else [],
            hazards=list(data.get("hazards", [])) if isinstance(data, dict) else [],
            suggested_actions=list(data.get("suggested_actions", [])) if isinstance(data, dict) else [],
            raw=text,
            processing_time_s=dt,
        )

        # Update throttle cache
        self._last_call_time = now
        self._last_analysis = analysis

        return analysis

    # -----------------------------------------------------
    # Decision
    # -----------------------------------------------------

    def decide_action(
        self,
        analysis: SceneAnalysis,
        mode: str,
        available_actions: List[str],
        target: Optional[str] = None,
        obstacle_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ask the model to choose one action from available_actions."""
        prompt = {
            "mode": mode,
            "target": target,
            "available_actions": available_actions,
            "analysis": {
                "description": analysis.description,
                "objects": analysis.objects,
                "hazards": analysis.hazards,
                "suggested_actions": analysis.suggested_actions,
            },
            "instruction": (
                "Choose the single best action from available_actions. "
                "Return STRICT JSON with keys: action (string), duration_s (number), reasoning (string). "
                "If unsafe/uncertain, pick 'stop'."
            ),
        }

        # Add ultrasonic sensor data if available
        if obstacle_info and obstacle_info.get("sensor_available"):
            distance = obstacle_info.get("distance_cm")
            has_obstacle = obstacle_info.get("has_obstacle")
            threshold = obstacle_info.get("threshold_cm")

            prompt["ultrasonic_sensor"] = {
                "distance_cm": distance,
                "obstacle_detected": has_obstacle,
                "threshold_cm": threshold,
                "note": (
                    f"Physical obstacle at {distance:.1f}cm (threshold: {threshold}cm). "
                    "DO NOT choose 'forward' if obstacle_detected is true!"
                ) if has_obstacle else f"Clear path ahead ({distance:.1f}cm)"
            }

        t0 = time.time()
        try:
            # Retry API call for transient failures
            def api_call():
                return self.client.responses.create(
                    model=self.model,
                    max_output_tokens=min(self.max_output_tokens, 500),
                    temperature=self.temperature,
                    input=[{"role": "user", "content": [{"type": "input_text", "text": json.dumps(prompt)}]}],
                    timeout=self.timeout_s,
                )

            resp = self._retry_api_call(api_call, max_retries=2)
            text = getattr(resp, "output_text", "") or ""
        except Exception as e:
            self.logger.error(f"Decision API error: {e}")
            self._speak_ai_failure_once("Decision system is unstable. Stopping.")
            return {"action": "stop", "duration_s": 0.5, "reasoning": "AI failure", "_latency_s": time.time() - t0}

        dt = time.time() - t0
        data = self._safe_json(text)

        if not isinstance(data, dict):
            return {"action": "stop", "duration_s": 0.5, "reasoning": text, "_latency_s": dt}

        action = str(data.get("action", "stop"))
        if action not in available_actions:
            action = "stop"
        try:
            duration_s = float(data.get("duration_s", 0.6))
        except Exception:
            duration_s = 0.6

        return {
            "action": action,
            "duration_s": max(0.2, min(duration_s, 3.0)),
            "reasoning": str(data.get("reasoning", "")),
            "_latency_s": dt,
        }

    # -----------------------------------------------------
    # 🗣️ AI-generated dialogue (new)
    # -----------------------------------------------------

    def generate_dialogue(
        self,
        *,
        mode: str,
        target: Optional[str],
        analysis: SceneAnalysis,
        decision: Dict[str, Any],
        executed_action: str,
    ) -> str:
        """
        Generate a short, personality-driven line of dialogue suitable for TTS.

        Safety + constraints:
          - short (<= ~20 words)
          - grounded in the analysis
          - no unsafe instructions
          - no mention of policies/system prompts
        Throttled + deduped internally.
        """
        vs = self.config.get("voice_settings", {}) if isinstance(self.config, dict) else {}
        enabled = bool(vs.get("enabled", True)) and bool(vs.get("dialogue_enabled", True))
        if not enabled:
            return ""

        # Throttle (separate from BaseBehavior throttling)
        now = time.time()
        min_interval_s = float(vs.get("dialogue_min_interval_s", 10.0))
        if (now - self._last_dialogue_at) < min_interval_s:
            return ""

        dedupe_window_s = float(vs.get("dialogue_dedupe_window_s", 30.0))
        if (
            self._last_dialogue_text
            and (now - self._last_dialogue_text_at) < dedupe_window_s
            and self._last_dialogue_text.strip()
        ):
            # we still allow new dialogue; dedupe check below prevents exact repeats
            pass

        # Personality (simple, explicit, editable)
        personality = self.config.get("personality_settings", {}) if isinstance(self.config, dict) else {}
        style = str(personality.get("style", "curious, analytical"))
        name = str(personality.get("name", "Nimue"))
        humor = str(personality.get("humor", "dry")).strip()

        # Keep it tight
        max_words = int(personality.get("max_words", 20))

        # Build a compact context object for the model
        ctx = {
            "name": name,
            "style": style,
            "humor": humor,
            "mode": mode,
            "target": target,
            "what_i_see": analysis.description,
            "hazards": analysis.hazards[:4],
            "objects": analysis.objects[:6],
            "suggested": analysis.suggested_actions[:4],
            "decision": {
                "action": decision.get("action"),
                "duration_s": decision.get("duration_s"),
                "reasoning": decision.get("reasoning", "")[:220],
            },
            "executed_action": executed_action,
        }

        prompt = (
            "You are the robot's inner voice. Produce ONE short spoken line for text-to-speech.\n"
            f"- Persona: {style}\n"
            f"- Name: {name}\n"
            f"- Max words: {max_words}\n"
            "- Requirements: grounded in what you see; mention a relevant object or hazard; "
            "reflect intent; keep it safe.\n"
            "- Do NOT output JSON. Output only the spoken sentence.\n\n"
            f"Context:\n{json.dumps(ctx)}"
        )

        t0 = time.time()
        try:
            resp = self.client.responses.create(
                model=self.model,
                max_output_tokens=80,
                temperature=min(self.temperature, 0.6),
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                timeout=self.timeout_s,
            )
            text = (getattr(resp, "output_text", "") or "").strip()
        except (RateLimitError, APITimeoutError, APIError) as e:
            self.logger.debug(f"Dialogue API failure (non-fatal): {e}")
            self._last_dialogue_at = now
            return ""
        except Exception as e:
            self.logger.debug(f"Dialogue error (non-fatal): {e}")
            self._last_dialogue_at = now
            return ""

        # Clean up: keep first line, strip quotes, enforce length
        line = text.splitlines()[0].strip().strip('"').strip("'")
        if not line:
            self._last_dialogue_at = now
            return ""

        # Enforce word cap (hard)
        words = line.split()
        if len(words) > max_words:
            line = " ".join(words[:max_words]).rstrip(" ,.;:") + "."

        # Dedupe exact repeats
        if self._last_dialogue_text == line and (now - self._last_dialogue_text_at) < dedupe_window_s:
            self._last_dialogue_at = now
            return ""

        # Update state
        self._last_dialogue_at = now
        self._last_dialogue_text = line
        self._last_dialogue_text_at = now

        self.logger.debug(f"Dialogue generated in {time.time() - t0:.2f}s: {line}")
        return line

    # -----------------------------------------------------

    def _safe_json(self, text: str) -> Any:
        """Best-effort JSON parse (handles code fences)."""
        s = text.strip()
        if "```" in s:
            parts = s.split("```")
            if len(parts) >= 2:
                s = parts[1].strip()
                if s.lower().startswith("json"):
                    s = s[4:].strip()
        try:
            parsed = json.loads(s)
            # Validate it's a dictionary (expected structure)
            if not isinstance(parsed, dict):
                self.logger.warning(f"JSON parsed but not a dict: {type(parsed)}")
                return {"description": "", "objects": [], "hazards": [], "suggested_actions": []}
            return parsed
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON decode error at position {e.pos}: {e.msg}")
            self.logger.debug(f"Invalid JSON text: {text[:200]}")
            return {"description": "", "objects": [], "hazards": [], "suggested_actions": []}
        except Exception as e:
            self.logger.warning(f"Unexpected error parsing JSON: {e}")
            return {"description": "", "objects": [], "hazards": [], "suggested_actions": []}
