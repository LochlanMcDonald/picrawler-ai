from __future__ import annotations

import hashlib
import logging
import os
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
    voice: str = "alloy"          # OpenAI built-in voice name
    format: str = "mp3"           # "mp3" is convenient for playback
    speed: float = 1.0            # if supported by model
    cooldown_s: float = 2.5       # minimum time between spoken lines
    dedupe_window_s: float = 15.0 # don't repeat same line too often
    verbosity: str = "normal"     # "low" | "normal" | "high"


class VoiceSystem:
    """
    Simple TTS wrapper:
      - Generates speech via OpenAI Audio API
      - Caches audio by text+voice+model
      - Plays audio locally (ffplay by default)
      - Cooldown + repeat suppression
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
        )

        # OpenAI client (SDK will also use OPENAI_API_KEY env var automatically)
        self.client = OpenAI(api_key=None if (api_key in (None, "", "your-api-key-here")) else api_key)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._last_spoken_at: float = 0.0
        self._last_text: Optional[str] = None
        self._last_text_at: float = 0.0

        # Pick a player. ffplay is the most reliable for mp3 on Pi.
        self._player = self._detect_player()

    def _detect_player(self) -> list[str]:
        # Prefer ffplay; it’s solid and non-blocky when used with -autoexit.
        for cmd in (["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],):
            if self._which(cmd[0]):
                return cmd
        # Fallback: mpg123 for mp3
        for cmd in (["mpg123", "-q"],):
            if self._which(cmd[0]):
                return cmd
        # Last resort: try aplay (only works for wav)
        if self._which("aplay"):
            return ["aplay", "-q"]
        return []

    @staticmethod
    def _which(binary: str) -> bool:
        from shutil import which
        return which(binary) is not None

    def say(self, text: str, *, level: str = "normal", force: bool = False) -> None:
        """
        Speak a line if:
          - enabled
          - passes verbosity
          - passes cooldown/dedupe
        """
        if not self.settings.enabled:
            return

        text = (text or "").strip()
        if not text:
            return

        if not force and not self._verbosity_allows(level):
            return

        now = time.time()

        # cooldown
        if not force and (now - self._last_spoken_at) < self.settings.cooldown_s:
            return

        # dedupe same line
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

    def _verbosity_allows(self, level: str) -> bool:
        order = {"low": 0, "normal": 1, "high": 2}
        want = order.get(level, 1)
        have = order.get(self.settings.verbosity, 1)
        return want <= have

    def _hash_key(self, text: str) -> str:
        key = f"{self.settings.model}|{self.settings.voice}|{self.settings.format}|{self.settings.speed}|{text}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]

    def _get_or_create_audio(self, text: str) -> Path:
        key = self._hash_key(text)
        out = self.cache_dir / f"{key}.{self.settings.format}"
        if out.exists() and out.stat().st_size > 0:
            return out

        # Create speech via OpenAI Audio API (Text-to-Speech)
        # Docs: Audio API / speech endpoint. :contentReference[oaicite:1]{index=1}
        resp = self.client.audio.speech.create(
            model=self.settings.model,
            voice=self.settings.voice,
            input=text,
            format=self.settings.format,
        )
        # Python SDK supports writing response bytes to file
        out.write_bytes(resp.read())
        return out

    def _play(self, audio_path: Path) -> None:
        if not self._player:
            raise RuntimeError("No audio player found (install ffplay or mpg123)")

        # If we ended up with aplay but format isn't wav, fail loudly.
        if self._player and self._player[0] == "aplay" and audio_path.suffix.lower() != ".wav":
            raise RuntimeError("aplay requires wav output; set voice_settings.format='wav'")

        cmd = [*self._player, str(audio_path)]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
