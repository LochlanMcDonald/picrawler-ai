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
    enabled: bool = True
    model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    format: str = "mp3"           # file extension only (mp3 or wav)
    speed: float = 1.0
    cooldown_s: float = 2.5
    dedupe_window_s: float = 15.0
    verbosity: str = "normal"     # low | normal | high
    max_chars: int = 260          # hard clip per utterance (saves cost)


class VoiceSystem:
    """
    Text-to-speech with:
      - OpenAI Audio API
      - on-disk cache
      - cooldown + dedupe
      - local playback
      - non-fatal failure
    """

    def __init__(self, config: dict, cache_dir: str = "logs/tts_cache"):
        self.logger = logging.getLogger(self.__class__.__name__)

        vs = (config.get("voice_settings") or {}) if isinstance(config, dict) else {}
        api_key = config.get("openai_api_key") if isinstance(config, dict) else None

        self.settings = VoiceSettings(
            enabled=bool(vs.get("enabled", True)),
            model=str(vs.get("model", "gpt-4o-mini-tts")),
            voice=str(vs.get("voice", "alloy")),
            format=str(vs.get("format", "mp3")),
            speed=float(vs.get("speed", 1.0)),
            cooldown_s=float(vs.get("cooldown_s", 2.5)),
            dedupe_window_s=float(vs.get("dedupe_window_s", 15.0)),
            verbosity=str(vs.get("verbosity", "normal")),
            max_chars=int(vs.get("max_chars", 260)),
        )

        # OpenAI client (requires API key via env or config)
        self.client = OpenAI(
            api_key=None if api_key in (None, "", "your-api-key-here") else api_key
        )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._last_spoken_at: float = 0.0
        self._last_text: Optional[str] = None
        self._last_text_at: float = 0.0

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
        if not self.settings.enabled:
            return

        text = (text or "").strip()
        if not text:
            return

        # Clip long lines to control TTS cost + latency
        if self.settings.max_chars and len(text) > self.settings.max_chars:
            text = text[: self.settings.max_chars].rstrip(" ,.;:") + "…"

        if not force and not self._verbosity_allows(level):
            return

        now = time.time()

        if not force and (now - self._last_spoken_at) < self.settings.cooldown_s:
            return

        if (
            not force
            and self._last_text == text
            and (now - self._last_text_at) < self.settings.dedupe_window_s
        ):
            return

        try:
            audio_path = self._get_or_create_audio(text)
            self._play(audio_path)

            self._last_spoken_at = now
            self._last_text = text
            self._last_text_at = now

        except Exception as e:
            self.logger.warning(f"TTS failed (non-fatal): {e}")

    # -----------------------------------------------------

    def _verbosity_allows(self, level: str) -> bool:
        # Treat "thought" as high-volume output (same gate as "high")
        order = {"low": 0, "normal": 1, "high": 2, "thought": 2}
        return order.get(level, 1) <= order.get(self.settings.verbosity, 1)

    def _hash_key(self, text: str) -> str:
        key = f"{self.settings.model}|{self.settings.voice}|{self.settings.format}|{self.settings.speed}|{text}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]

    # -----------------------------------------------------

    def _get_or_create_audio(self, text: str) -> Path:
        key = self._hash_key(text)
        out = self.cache_dir / f"{key}.{self.settings.format}"

        # Security: Validate path is within cache directory (prevent path traversal)
        try:
            resolved_out = out.resolve()
            resolved_cache = self.cache_dir.resolve()
            if not str(resolved_out).startswith(str(resolved_cache)):
                raise ValueError(f"Invalid audio path: {out}")
        except Exception as e:
            self.logger.error(f"Path validation failed: {e}")
            raise

        if out.exists() and out.stat().st_size > 0:
            return out

        audio = self.client.audio.speech.create(
            model=self.settings.model,
            voice=self.settings.voice,
            input=text,
        )

        out.write_bytes(audio.read())
        return out

    # -----------------------------------------------------

    def _play(self, audio_path: Path) -> None:
        if not self._player:
            raise RuntimeError("No audio player found (install ffplay or mpg123)")

        if self._player[0] == "aplay" and audio_path.suffix.lower() != ".wav":
            raise RuntimeError("aplay requires wav output")

        # Security: Validate path exists and is within cache directory
        try:
            resolved_path = audio_path.resolve()
            resolved_cache = self.cache_dir.resolve()
            if not str(resolved_path).startswith(str(resolved_cache)):
                raise ValueError(f"Audio path outside cache directory: {audio_path}")
            if not resolved_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
        except Exception as e:
            self.logger.error(f"Audio path validation failed: {e}")
            raise

        subprocess.run(
            [*self._player, str(audio_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )