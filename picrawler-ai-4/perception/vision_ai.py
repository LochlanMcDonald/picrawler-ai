"""
Vision AI for scene understanding.

This analyzes camera images and returns structured information
that feeds into the world model.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI, APIError


@dataclass
class SceneAnalysis:
    """Structured scene understanding from vision AI."""
    description: str
    objects: List[str]
    hazards: List[str]
    suggestions: List[str]
    processing_time: float


class VisionAI:
    """Vision analysis using OpenAI."""

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)

        api_key = config.get("openai_api_key")
        self.client = OpenAI(api_key=None if api_key in (None, "", "your-api-key-here") else api_key)

        ai = config.get("ai_settings", {})
        self.model = ai.get("model", "gpt-4o-mini")
        self.timeout_s = int(ai.get("request_timeout_s", 30))

        # Throttling
        self.last_call_time = 0.0
        self.min_interval = 3.0
        self.last_analysis: Optional[SceneAnalysis] = None

    def analyze_scene(self, image_base64: str) -> Optional[SceneAnalysis]:
        """Analyze camera image and return structured data."""

        # Throttle API calls
        now = time.time()
        if self.last_analysis and (now - self.last_call_time) < self.min_interval:
            self.logger.debug("Throttling: reusing last analysis")
            return self.last_analysis

        prompt = """You are a robot vision system. Analyze what you see for navigation.

Be concise and grounded. Focus on obstacles and navigation hazards.

Return STRICT JSON:
{
    "description": "brief description of scene",
    "objects": ["object1", "object2"],
    "hazards": ["hazard1", "hazard2"],
    "suggestions": ["suggested_action1"]
}

Hazards include: walls, furniture, obstacles, stairs, etc.
"""

        try:
            t0 = time.time()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }],
                timeout=self.timeout_s
            )

            text = response.choices[0].message.content
            dt = time.time() - t0

            data = self._parse_json(text)

            if data:
                analysis = SceneAnalysis(
                    description=str(data.get('description', '')),
                    objects=list(data.get('objects', [])),
                    hazards=list(data.get('hazards', [])),
                    suggestions=list(data.get('suggestions', [])),
                    processing_time=dt
                )

                self.last_analysis = analysis
                self.last_call_time = now

                self.logger.info(f"Scene: {analysis.description}")
                return analysis

        except (APIError, Exception) as e:
            self.logger.error(f"Vision AI failed: {e}")

        # Return safe fallback
        return SceneAnalysis(
            description="Vision unavailable",
            objects=[],
            hazards=["unknown"],  # Assume hazard for safety
            suggestions=["stop"],
            processing_time=0.0
        )

    def _parse_json(self, text: str) -> Optional[dict]:
        """Parse JSON from AI response."""
        try:
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        return json.loads(part)

            if text.strip().startswith("{"):
                return json.loads(text.strip())

        except json.JSONDecodeError:
            pass

        return None
