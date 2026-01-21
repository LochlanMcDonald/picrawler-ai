"""Camera capture with graceful fallbacks.

Preferred on Pi: Picamera2
Fallback: OpenCV VideoCapture

Returns both a PIL Image and a base64-encoded JPEG suitable for OpenAI vision.
Includes panic-stop detection on sudden visual change.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageChops, ImageStat

# Voice (optional)
from voice.voice_system import VoiceSystem


class CameraPanicException(Exception):
    """Raised when sudden visual change is detected (safety mechanism)."""
    pass


class CameraSystem:
    def __init__(self, config: dict, save_dir: str = "logs/images"):
        self.logger = logging.getLogger(self.__class__.__name__)
        cs = config.get("camera_settings", {})
        self.resolution = tuple(cs.get("resolution", [640, 480]))
        self.quality = int(cs.get("image_quality", 85))
        self.flip_h = bool(cs.get("flip_h", False))
        self.flip_v = bool(cs.get("flip_v", False))

        # Panic detection settings
        self.enable_panic = bool(cs.get("panic_on_visual_change", True))
        self.panic_threshold = float(cs.get("panic_change_threshold", 35.0))
        self.panic_cooldown_s = float(cs.get("panic_cooldown_s", 1.0))

        self._last_panic_time: float = 0.0
        self._prev_image: Optional[Image.Image] = None

        # Voice system (safe: no-op if disabled or fails)
        self.voice = VoiceSystem(config)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._picam2 = None
        self._cv = None

        self._init_backend()

    def _init_backend(self) -> None:
        # Try Picamera2 first
        try:
            from picamera2 import Picamera2  # type: ignore

            picam2 = Picamera2()
            config = picam2.create_still_configuration(main={"size": self.resolution})
            picam2.configure(config)
            picam2.start()
            time.sleep(0.2)
            self._picam2 = picam2
            self.logger.info("Camera backend: Picamera2")
            return
        except Exception as e:
            self.logger.warning(f"Picamera2 not available ({e}); trying OpenCV")

        # Fallback: OpenCV
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            if not cap.isOpened():
                raise RuntimeError("cv2.VideoCapture(0) not opened")
            self._cv = cap
            self.logger.info("Camera backend: OpenCV")
        except Exception as e:
            self.logger.error(f"No camera backend available: {e}")
            self._cv = None

    def _detect_panic(self, img: Image.Image) -> bool:
        """Return True if sudden visual change is detected."""
        if not self.enable_panic:
            self._prev_image = img
            return False

        now = time.time()
        if now - self._last_panic_time < self.panic_cooldown_s:
            self._prev_image = img
            return False

        if self._prev_image is None:
            self._prev_image = img
            return False

        try:
            # Convert to grayscale and compute mean absolute difference
            diff = ImageChops.difference(
                self._prev_image.convert("L"),
                img.convert("L"),
            )
            stat = ImageStat.Stat(diff)
            mean_diff = stat.mean[0]
        except Exception:
            self._prev_image = img
            return False

        self._prev_image = img

        if mean_diff >= self.panic_threshold:
            self._last_panic_time = now
            self.logger.warning(
                f"PANIC: sudden visual change detected (Δ={mean_diff:.1f} ≥ {self.panic_threshold})"
            )

            # Narrate panic once
            try:
                self.voice.say(
                    "That changed suddenly. Stopping.",
                    level="normal",
                )
            except Exception:
                pass

            return True

        return False

    def capture(self, save: bool = True) -> Tuple[Optional[Image.Image], Optional[str], Optional[str]]:
        """Capture an image.

        Returns:
            (pil_image, base64_jpeg, saved_path)
        """
        img: Optional[Image.Image] = None

        if self._picam2 is not None:
            try:
                array = self._picam2.capture_array()
                img = Image.fromarray(array)
            except Exception as e:
                self.logger.error(f"Picamera2 capture failed: {e}")
                img = None

        if img is None and self._cv is not None:
            try:
                import cv2  # type: ignore

                ok, frame = self._cv.read()
                if not ok:
                    raise RuntimeError("cv2 read() failed")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
            except Exception as e:
                self.logger.error(f"OpenCV capture failed: {e}")
                img = None

        if img is None:
            return None, None, None

        if self.flip_h:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_v:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # --- PANIC DETECTION ---
        if self._detect_panic(img):
            raise CameraPanicException("Sudden visual change detected - emergency stop")
        # ----------------------

        # JPEG encode
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        saved_path = None
        if save:
            ts = time.strftime("%Y%m%d-%H%M%S")
            saved_path = str(self.save_dir / f"frame_{ts}.jpg")
            with open(saved_path, "wb") as f:
                f.write(buf.getvalue())

        return img, b64, saved_path

    def close(self) -> None:
        try:
            if self._cv is not None:
                self._cv.release()
        except Exception:
            pass

        try:
            if self._picam2 is not None:
                self._picam2.stop()
        except Exception:
            pass
