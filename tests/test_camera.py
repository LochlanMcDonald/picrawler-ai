"""Tests for camera panic detection (no camera hardware required)."""

import logging

import pytest
from PIL import Image

from vision.camera import CameraPanicException, CameraSystem


def make_camera(dummy_voice, *, enable_panic=True, threshold=35.0, cooldown=0.0):
    """Build a CameraSystem without initializing any camera backend."""
    cam = CameraSystem.__new__(CameraSystem)
    cam.logger = logging.getLogger("test-camera")
    cam.enable_panic = enable_panic
    cam.panic_threshold = threshold
    cam.panic_cooldown_s = cooldown
    cam._last_panic_time = 0.0
    cam._prev_image = None
    cam.voice = dummy_voice
    return cam


def solid(color, size=(32, 32)):
    return Image.new("RGB", size, color)


def test_first_frame_never_panics(dummy_voice):
    cam = make_camera(dummy_voice)
    assert cam._detect_panic(solid("black")) is False


def test_sudden_change_panics(dummy_voice):
    cam = make_camera(dummy_voice)
    cam._detect_panic(solid("black"))
    assert cam._detect_panic(solid("white")) is True
    assert dummy_voice.spoken  # narrated the panic


def test_similar_frames_do_not_panic(dummy_voice):
    cam = make_camera(dummy_voice)
    cam._detect_panic(solid((10, 10, 10)))
    assert cam._detect_panic(solid((15, 15, 15))) is False


def test_panic_disabled_by_config(dummy_voice):
    cam = make_camera(dummy_voice, enable_panic=False)
    cam._detect_panic(solid("black"))
    assert cam._detect_panic(solid("white")) is False


def test_panic_cooldown_suppresses_back_to_back_panics(dummy_voice):
    cam = make_camera(dummy_voice, cooldown=60.0)
    cam._detect_panic(solid("black"))
    assert cam._detect_panic(solid("white")) is True
    # Immediately after, another huge change is inside the cooldown window
    assert cam._detect_panic(solid("black")) is False


def test_threshold_boundary(dummy_voice):
    # Mean absolute diff between 100 and 130 gray is 30 < 35 → no panic
    cam = make_camera(dummy_voice, threshold=35.0)
    cam._detect_panic(solid((100, 100, 100)))
    assert cam._detect_panic(solid((130, 130, 130))) is False
    # 100 → 140 is 40 ≥ 35 → panic
    cam2 = make_camera(dummy_voice, threshold=35.0)
    cam2._detect_panic(solid((100, 100, 100)))
    assert cam2._detect_panic(solid((140, 140, 140))) is True


def test_capture_raises_panic_exception_is_exported():
    # BaseBehavior imports and catches this type; keep it stable.
    assert issubclass(CameraPanicException, Exception)
