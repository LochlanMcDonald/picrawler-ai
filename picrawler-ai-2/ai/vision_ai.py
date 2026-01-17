"""OpenAI vision + decision helper.

Uses the OpenAI Responses API (text + image inputs).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class SceneAnalysis:
    description: str
    objects: List[str]
    hazards: List[str]
    suggested_actions: List[str]
    raw: str
    processing_time_s: float


class AIVisionSystem:
    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        api_key = config.get("openai_api_key") or None
        # Also allow OPENAI_API_KEY env var automatically via SDK
        self.client = OpenAI(api_key=None if (api_key in (None, "", "your-api-key-here")) else api_key)

        ai = config.get("ai_settings", {})
        self.model = ai.get("model", "gpt-4.1-mini")
        self.max_output_tokens = int(ai.get("max_output_tokens", 800))
        self.temperature = float(ai.get("temperature", 0.4))
        self.timeout_s = int(ai.get("request_timeout_s", 30))

    def analyze_scene(self, image_base64: str, context: str) -> SceneAnalysis:
        """Return a structured summary of what the robot sees."""
        prompt = (
            "You are a robotics perception module. Summarize what you see for navigation. "
            "Be concise, grounded, and action-oriented.\n\n"
            f"Task context: {context}\n\n"
            "Return STRICT JSON with keys: description (string), objects (array of strings), "
            "hazards (array of strings), suggested_actions (array of strings)."
        )

        t0 = time.time()
        resp = self.client.responses.create(
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
        dt = time.time() - t0
        text = getattr(resp, "output_text", "") or ""

        data = self._safe_json(text)
        return SceneAnalysis(
            description=str(data.get("description", "")) if isinstance(data, dict) else "",
            objects=list(data.get("objects", [])) if isinstance(data, dict) else [],
            hazards=list(data.get("hazards", [])) if isinstance(data, dict) else [],
            suggested_actions=list(data.get("suggested_actions", [])) if isinstance(data, dict) else [],
            raw=text,
            processing_time_s=dt,
        )

    def decide_action(
        self,
        analysis: SceneAnalysis,
        mode: str,
        available_actions: List[str],
        target: Optional[str] = None,
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

        t0 = time.time()
        resp = self.client.responses.create(
            model=self.model,
            max_output_tokens=min(self.max_output_tokens, 500),
            temperature=self.temperature,
            input=[{"role": "user", "content": [{"type": "input_text", "text": json.dumps(prompt)}]}],
            timeout=self.timeout_s,
        )
        dt = time.time() - t0
        text = getattr(resp, "output_text", "") or ""
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

    def _safe_json(self, text: str) -> Any:
        """Best-effort JSON parse (handles code fences)."""
        s = text.strip()
        if "```" in s:
            # pick first fenced block
            parts = s.split("```")
            if len(parts) >= 2:
                s = parts[1].strip()
                # strip 'json' label
                if s.lower().startswith("json"):
                    s = s[4:].strip()
        try:
            return json.loads(s)
        except Exception:
            self.logger.warning("Model output was not valid JSON; returning raw text")
            return {"description": "", "objects": [], "hazards": [], "suggested_actions": [], "raw": text}
