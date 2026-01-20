"""Camera capture with graceful fallbacks.

Preferred on Pi: Picamera2
Fallback: OpenCV VideoCapture

Returns both a PIL Image and a base64-encoded JPEG suitable for OpenAI vision.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image


class CameraSystem:
    def __init__(self, config: dict, save_dir: str = "logs/images"):
        self.logger = logging.getLogger(self.__class__.__name__)
        cs = config.get("camera_settings", {})
        self.resolution = tuple(cs.get("resolution", [640, 480]))
        self.quality = int(cs.get("image_quality", 85))
        self.flip_h = bool(cs.get("flip_h", False))
        self.flip_v = bool(cs.get("flip_v", False))

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
