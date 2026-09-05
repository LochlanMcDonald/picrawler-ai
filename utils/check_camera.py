#!/usr/bin/env python3

from core.config_loader import load_config
from core.logger import setup_logging
from vision.camera import CameraSystem


def main() -> int:
    setup_logging("INFO")
    config = load_config("config/config.json")
    cam = CameraSystem(config)
    _, b64, path = cam.capture(save=True)
    cam.close()
    if not b64:
        print("Camera capture failed")
        return 1
    print(f"Camera OK. Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
