"""Tests for camera panic detection (no camera hardware required)."""

import logging
from pathlib import Path

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


# ---------------------------------------------------------------- capture


import base64

import numpy as np


class FakePicam2:
    def __init__(self, array):
        self.array = array
        self.stopped = False

    def capture_array(self):
        if isinstance(self.array, Exception):
            raise self.array
        return self.array

    def stop(self):
        self.stopped = True


class FakeCv:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def half_and_half():
    """Left half black, right half white."""
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    a[:, 16:, :] = 255
    return a


def make_capture_camera(tmp_path, dummy_voice, *, picam=None, cv=None, flip_h=False, flip_v=False, enable_panic=False):
    cam = make_camera(dummy_voice, enable_panic=enable_panic)
    cam.quality = 85
    cam.flip_h = flip_h
    cam.flip_v = flip_v
    cam.save_dir = tmp_path / "images"
    cam.save_dir.mkdir(parents=True, exist_ok=True)
    cam._picam2 = picam
    cam._cv = cv
    return cam


def test_capture_without_backend_returns_nones(tmp_path, dummy_voice):
    cam = make_capture_camera(tmp_path, dummy_voice)
    assert cam.capture() == (None, None, None)


def test_capture_encodes_jpeg_and_saves(tmp_path, dummy_voice):
    cam = make_capture_camera(tmp_path, dummy_voice, picam=FakePicam2(half_and_half()))
    img, b64, path = cam.capture(save=True)
    assert img.size == (32, 32)
    raw = base64.b64decode(b64)
    assert raw[:2] == b"\xff\xd8"  # JPEG magic bytes
    assert Path(path).exists()
    assert Path(path).read_bytes() == raw


def test_capture_save_false_writes_nothing(tmp_path, dummy_voice):
    cam = make_capture_camera(tmp_path, dummy_voice, picam=FakePicam2(half_and_half()))
    _, b64, path = cam.capture(save=False)
    assert b64 is not None
    assert path is None
    assert list(cam.save_dir.iterdir()) == []


def test_capture_flip_h(tmp_path, dummy_voice):
    cam = make_capture_camera(tmp_path, dummy_voice, picam=FakePicam2(half_and_half()), flip_h=True)
    img, _, _ = cam.capture(save=False)
    # Originally left is black; after horizontal flip it's white
    assert img.getpixel((0, 0)) == (255, 255, 255)
    assert img.getpixel((31, 0)) == (0, 0, 0)


def test_capture_backend_failure_returns_nones(tmp_path, dummy_voice):
    cam = make_capture_camera(tmp_path, dummy_voice, picam=FakePicam2(RuntimeError("bus error")))
    assert cam.capture() == (None, None, None)


def test_capture_raises_on_panic(tmp_path, dummy_voice):
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    cam = make_capture_camera(tmp_path, dummy_voice, picam=FakePicam2(black), enable_panic=True)
    cam.capture(save=False)  # primes _prev_image
    cam._picam2.array = white
    with pytest.raises(CameraPanicException):
        cam.capture(save=False)


def test_close_releases_backends(tmp_path, dummy_voice):
    picam = FakePicam2(half_and_half())
    cv = FakeCv()
    cam = make_capture_camera(tmp_path, dummy_voice, picam=picam, cv=cv)
    cam.close()
    assert picam.stopped is True
    assert cv.released is True
