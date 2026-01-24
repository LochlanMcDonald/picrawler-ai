"""
Monocular depth estimation using learned models.

Provides depth maps from single camera images, enabling 3D spatial
understanding without hardware depth sensors.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    from torchvision.transforms import Compose
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available - depth estimation disabled")


@dataclass
class DepthMap:
    """Depth estimation result."""
    depth_array: np.ndarray  # Normalized 0-1, where 0=close, 1=far
    resolution: Tuple[int, int]  # (width, height)
    inference_time_ms: float
    timestamp: float

    def get_region_depth(self, x_start: float, x_end: float,
                        y_start: float, y_end: float) -> float:
        """Get average depth in a region (coordinates 0-1).

        Args:
            x_start, x_end: Horizontal region (0=left, 1=right)
            y_start, y_end: Vertical region (0=top, 1=bottom)

        Returns:
            Average depth in region (0=close, 1=far)
        """
        h, w = self.depth_array.shape
        x1, x2 = int(x_start * w), int(x_end * w)
        y1, y2 = int(y_start * h), int(y_end * h)

        region = self.depth_array[y1:y2, x1:x2]
        return float(np.mean(region))

    def get_center_depth(self) -> float:
        """Get depth at center of image."""
        return self.get_region_depth(0.4, 0.6, 0.4, 0.6)

    def get_directional_depths(self) -> dict:
        """Get depth in cardinal directions.

        Returns:
            Dict with 'front', 'left', 'right' depths (0=close, 1=far)
        """
        return {
            'front': self.get_region_depth(0.35, 0.65, 0.4, 0.7),
            'left': self.get_region_depth(0.0, 0.3, 0.3, 0.7),
            'right': self.get_region_depth(0.7, 1.0, 0.3, 0.7),
        }


class DepthEstimator:
    """Monocular depth estimation using MiDaS."""

    def __init__(self, model_type: str = "MiDaS_small",
                 input_size: int = 256,
                 cache_duration_s: float = 0.5):
        """
        Args:
            model_type: 'MiDaS_small' (fast, Pi-optimized) or 'DPT_Hybrid' (accurate, slow)
            input_size: Input resolution (256 for Pi, 384 for better quality)
            cache_duration_s: Minimum time between depth estimations
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_type = model_type
        self.input_size = input_size
        self.cache_duration_s = cache_duration_s

        self.model = None
        self.transform = None
        self.device = None

        # Caching
        self.last_depth_map: Optional[DepthMap] = None
        self.last_inference_time: float = 0.0

        if TORCH_AVAILABLE:
            self._load_model()
        else:
            self.logger.warning("Depth estimation unavailable - PyTorch not installed")

    def _load_model(self) -> None:
        """Load MiDaS model."""
        try:
            self.logger.info(f"Loading {self.model_type} depth model...")

            # Use CPU on Pi (no GPU)
            self.device = torch.device("cpu")

            # Load model from torch hub
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                self.model_type,
                pretrained=True,
                skip_validation=True
            )
            self.model.to(self.device)
            self.model.eval()

            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if self.model_type == "MiDaS_small":
                self.transform = midas_transforms.small_transform
            else:
                self.transform = midas_transforms.dpt_transform

            self.logger.info(f"Depth model loaded on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load depth model: {e}")
            self.model = None

    def estimate_depth(self, image: np.ndarray,
                      force_refresh: bool = False) -> Optional[DepthMap]:
        """Estimate depth from RGB image.

        Args:
            image: RGB image (H, W, 3) as numpy array
            force_refresh: Skip cache and run new inference

        Returns:
            DepthMap or None if estimation failed
        """
        if not TORCH_AVAILABLE or self.model is None:
            return None

        # Check cache
        now = time.time()
        if (not force_refresh and
            self.last_depth_map is not None and
            (now - self.last_inference_time) < self.cache_duration_s):
            return self.last_depth_map

        try:
            start_time = time.time()

            # Resize input for faster inference on Pi
            if image.shape[0] > self.input_size or image.shape[1] > self.input_size:
                image = cv2.resize(image, (self.input_size, self.input_size))

            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Transform and run inference
            input_batch = self.transform(image_rgb).to(self.device)

            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Convert to numpy and normalize to 0-1
            depth = prediction.cpu().numpy()
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())

            # Invert so 0=close, 1=far
            depth_normalized = 1.0 - depth_normalized

            inference_time_ms = (time.time() - start_time) * 1000

            depth_map = DepthMap(
                depth_array=depth_normalized,
                resolution=(image_rgb.shape[1], image_rgb.shape[0]),
                inference_time_ms=inference_time_ms,
                timestamp=time.time()
            )

            # Update cache
            self.last_depth_map = depth_map
            self.last_inference_time = now

            self.logger.debug(
                f"Depth estimated in {inference_time_ms:.1f}ms "
                f"(center depth: {depth_map.get_center_depth():.2f})"
            )

            return depth_map

        except Exception as e:
            self.logger.error(f"Depth estimation failed: {e}")
            return None

    def visualize_depth(self, depth_map: DepthMap) -> np.ndarray:
        """Create colorized visualization of depth map.

        Args:
            depth_map: DepthMap to visualize

        Returns:
            RGB image with depth colored (blue=far, red=close)
        """
        # Apply colormap
        depth_colored = cv2.applyColorMap(
            (depth_map.depth_array * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO
        )

        # Add directional overlays
        h, w = depth_map.depth_array.shape

        # Front region
        cv2.rectangle(depth_colored,
                     (int(0.35*w), int(0.4*h)),
                     (int(0.65*w), int(0.7*h)),
                     (255, 255, 255), 2)
        cv2.putText(depth_colored, "FRONT",
                   (int(0.4*w), int(0.35*h)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Left region
        cv2.rectangle(depth_colored,
                     (0, int(0.3*h)),
                     (int(0.3*w), int(0.7*h)),
                     (255, 255, 0), 2)
        cv2.putText(depth_colored, "L",
                   (int(0.05*w), int(0.5*h)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Right region
        cv2.rectangle(depth_colored,
                     (int(0.7*w), int(0.3*h)),
                     (w, int(0.7*h)),
                     (255, 255, 0), 2)
        cv2.putText(depth_colored, "R",
                   (int(0.85*w), int(0.5*h)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return depth_colored
