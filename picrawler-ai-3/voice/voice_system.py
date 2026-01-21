from __future__ import annotations

import hashlib
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import OpenAI


@dataclass
class VoiceSettings:
    # Master enable
    enabled: bool = True

    # OpenAI TTS
    model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"

    # Output format (mp3 or wav). Used for file extension AND (when supported) response_format.
    format: str = "mp3"

    # Some SDK versions support speed for TTS. We'll try and fallback safely.
    speed: float = 1.0

    # Standard cooldown/dedupe (used for normal narration/dialogue)
    cooldown_s: float = 2.5
    dedupe_window_s: float = 15.0

    # Verbosity gate for normal use
    verbosity: str = "normal"  # low | normal | high

    # --- NEW: "thoughts" channel settings ---
    # If enabled, level="thought" lines can be spoken even when normal narration is throttled.
    thoughts_enabled: bool = False

    # If True, thought lines ignore normal cooldown and use thoughts_* controls instead.
    thoughts_bypass_cooldown: bool = True

    # Separate cooldown/dedupe for thought lines (set to 0 to speak everything).
    thoughts_cooldown_s: float = 0.0
    thoughts_dedupe_window_s: float = 0.0

    # If True, do not suppress identical thought lines (even if dedupe window > 0)
    thoughts_allow_repeat: bool = True
    # ----------------------------------------


class VoiceSystem:
    """
    Text-to-speech with:
      - OpenAI Audio API
      - on-disk cache
      - cooldown + dedupe (separate for "thoughts" if enabled)
      - local playback
      - non-fatal failure

    Levels supported:
      - "low", "normal", "high" : gated by settings.verbosity
      - "thought"               : controlled by settings.thoughts_*
    """

    def __init__(self, config: dict, cache_dir: str = "logs/tts_cache"):
        self.logger = logging.getLogger(self.__class__.__name__)

        vs = (config.get("voice_settings") or {}) if isinstance(config, dict) else {}
        api_key = config.get("openai_api_key") if isinstance(config, dict) else None

        # NOTE: Config keys are intentionally forgiving.
        self.settings = VoiceSettings(
            enabled=bool(vs.get("enabled", True)),
            model=str(vs.get("model", "gpt-4o-mini-tts")),
            voice=str(vs.get("voice", "alloy")),
            format=str(vs.get("format", "mp3")),
            speed=float(vs.get("speed", 1.0)),
            cooldown_s=float(vs.get("cooldown_s", 2.5)),
            dedupe_window_s=float(vs.get("dedupe_window_s", 15.0)),
            verbosity=str(vs.get("verbosity", "normal")),
            thoughts_enabled=bool(vs.get("thoughts_enabled", False)),
            thoughts_bypass_cooldown=bool(vs.get("thoughts_bypass_cooldown", True)),
            thoughts_cooldown_s=float(vs.get("thoughts_cooldown_s", 0.0)),
            thoughts_dedupe_window_s=float(vs.get("thoughts_dedupe_window_s", 0.0)),
            thoughts_allow_repeat=bool(vs.get("thoughts_allow_repeat", True)),
        )

        # OpenAI client (requires API key via env or config)
        self.client = OpenAI(
            api_key=None if api_key in (None, "", "your-api-key-here") else api_key
        )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Separate tracking for normal vs thought
        self._last_spoken_at_normal: float = 0.0
        self._last_text_normal: Optional[str] = None
        self._last_text_at_normal: float = 0.0

        self._last_spoken_at_thought: float = 0.0
        self._last_text_thought: Optional[str] = None
        self._last_text_at_thought: float = 0.0

        self._player = self._detect_player()

    # -----------------------------------------------------

    def _detect_player(self) -> list[str]:
        from shutil import which

        if which("ffplay"):
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
        if which("mpg123"):
            return ["mpg123", "-q"]
        if which("aplay"):
            return ["aplay", "-q"]
        return []

    # -----------------------------------------------------

    def say(self, text: str, *, level: str = "normal", force: bool = False) -> None:
        """
        Speak text.

        - level in {"low","normal","high","thought"}
        - force=True bypasses *all* gating/dedupe/cooldown.
        """
        if not self.settings.enabled:
            return

        text = (text or "").strip()
        if not text:
            return

        is_thought = (level == "thought")

        # Gate by verbosity for normal channels
        if not force and not is_thought and not self._verbosity_allows(level):
            return

        # Gate thoughts channel
        if is_thought and not force and not self.settings.thoughts_enabled:
            return

        now = time.time()

        # Pick which timing rules apply
        if is_thought and self.settings.thoughts_bypass_cooldown and not force:
            cooldown_s = self.settings.thoughts_cooldown_s
            dedupe_window_s = self.settings.thoughts_dedupe_window_s
            last_spoken_at = self._last_spoken_at_thought
            last_text = self._last_text_thought
            last_text_at = self._last_text_at_thought
        else:
            cooldown_s = self.settings.cooldown_s
            dedupe_window_s = self.settings.dedupe_window_s
            last_spoken_at = self._last_spoken_at_normal
            last_text = self._last_text_normal
            last_text_at = self._last_text_at_normal

        # Cooldown
        if not force and cooldown_s > 0 and (now - last_spoken_at) < cooldown_s:
            return

        # Dedupe
        if not force and dedupe_window_s > 0:
            if is_thought and self.settings.thoughts_allow_repeat:
                # Explicitly allow repeats for thought channel
                pass
            else:
                if last_text == text and (now - last_text_at) < dedupe_window_s:
                    return

        try:
            audio_path = self._get_or_create_audio(text)
            self._play(audio_path)

            # Update appropriate trackers
            if is_thought and self.settings.thoughts_bypass_cooldown and not force:
                self._last_spoken_at_thought = now
                self._last_text_thought = text
                self._last_text_at_thought = now
            else:
                self._last_spoken_at_normal = now
                self._last_text_normal = text
                self._last_text_at_normal = now

        except Exception as e:
            self.logger.warning(f"TTS failed (non-fatal): {e}")

    # -----------------------------------------------------

    def _verbosity_allows(self, level: str) -> bool:
        # "thought" is handled separately before this is called
        order = {"low": 0, "normal": 1, "high": 2}
        return order.get(level, 1) <= order.get(self.settings.verbosity, 1)

    def _hash_key(self, text: str) -> str:
        # Include settings in the key so switching voices/models creates a new cache entry
        key = f"{self.settings.model}|{self.settings.voice}|{self.settings.format}|{self.settings.speed}|{text}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]

    # -----------------------------------------------------

    def _get_or_create_audio(self, text: str) -> Path:
        key = self._hash_key(text)
        out = self.cache_dir / f"{key}.{self.settings.format}"

        if out.exists() and out.stat().st_size > 0:
            return out

        # Build kwargs, and fall back gracefully if SDK version doesn't accept some params
        kwargs = {
            "model": self.settings.model,
            "voice": self.settings.voice,
            "input": text,
        }

        # Some SDK versions support speed / response_format.
        # We'll attempt them; if it errors, retry without optional params.
        optional_kwargs = {}
        if self.settings.speed and self.settings.speed != 1.0:
            optional_kwargs["speed"] = self.settings.speed

        # Many OpenAI Audio Speech APIs use `response_format` (not `format`).
        # We'll attempt to set it; if not supported, retry.
        if self.settings.format:
            optional_kwargs["response_format"] = self.settings.format

        try:
            audio = self.client.audio.speech.create(**kwargs, **optional_kwargs)
        except TypeError:
            # Older SDK: retry without optional params
            audio = self.client.audio.speech.create(**kwargs)

        out.write_bytes(audio.read())
        return out

    # -----------------------------------------------------

    def _play(self, audio_path: Path) -> None:
        if not self._player:
            raise RuntimeError("No audio player found (install ffplay or mpg123)")

        if self._player[0] == "aplay" and audio_path.suffix.lower() != ".wav":
            raise RuntimeError("aplay requires wav output (set voice_settings.format to 'wav')")

        subprocess.run(
            [*self._player, str(audio_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )