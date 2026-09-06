#!/usr/bin/env python3
"""
Hardware diagnostic — talks to the SunFounder library DIRECTLY.

Bypasses every part of picrawler-ai. If the robot does not move when this
runs, the problem is power, wiring, servo calibration or the SunFounder
install, not this project's code.

Run on the robot:
    cd ~/picrawler-ai/picrawler-ai-4
    python utils/check_hardware.py

Paste the whole output when asking for help.
"""
from __future__ import annotations

import inspect
import platform
import sys
import time
import traceback


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    section("1. Python / platform")
    print("python  :", sys.version.split()[0], "at", sys.executable)
    print("platform:", platform.platform())

    # ------------------------------------------------------------------
    section("2. robot_hat (servo driver / battery)")
    try:
        import robot_hat  # type: ignore
        print("robot_hat file   :", getattr(robot_hat, "__file__", "?"))
        print("robot_hat version:", getattr(robot_hat, "__version__", "?"))
        try:
            from robot_hat import utils as rh_utils  # type: ignore
            if hasattr(rh_utils, "get_battery_voltage"):
                v = rh_utils.get_battery_voltage()
                print(f"battery voltage  : {v:.2f} V", "(LOW — servos will not move)" if v < 6.5 else "")
            else:
                print("battery voltage  : (no get_battery_voltage in this robot_hat)")
        except Exception as e:
            print("battery voltage  : read failed:", e)
    except Exception as e:
        print("robot_hat import FAILED:", e)

    # ------------------------------------------------------------------
    section("3. picrawler library")
    try:
        import picrawler  # type: ignore
        from picrawler import Picrawler  # type: ignore
        print("picrawler file   :", getattr(picrawler, "__file__", "?"))
        print("do_action sig    :", inspect.signature(Picrawler.do_action))
        if hasattr(Picrawler, "do_step"):
            print("do_step sig      :", inspect.signature(Picrawler.do_step))
    except Exception as e:
        print("picrawler import FAILED:", e)
        traceback.print_exc()
        return 1

    # ------------------------------------------------------------------
    section("4. Create Picrawler() — servos should centre / twitch here")
    t0 = time.time()
    try:
        crawler = Picrawler()
    except Exception as e:
        print("Picrawler() FAILED:", e)
        traceback.print_exc()
        return 1
    print(f"Picrawler() ok in {time.time() - t0:.2f}s")

    ml = getattr(crawler, "move_list", None)
    try:
        names = list(ml.keys()) if hasattr(ml, "keys") else [a for a in dir(ml) if not a.startswith("_")]
        print("motion names     :", names)
    except Exception as e:
        print("motion names     : could not list:", e)

    # ------------------------------------------------------------------
    section("5. Motions (watch the robot!)  each is ONE gait cycle")
    plan = [
        ("stand",      1, 50),
        ("forward",    2, 50),
        ("turn left",  1, 50),
        ("turn right", 1, 50),
        ("backward",   1, 50),
        ("sit",        1, 50),
    ]
    for name, step, speed in plan:
        t0 = time.time()
        try:
            crawler.do_action(name, step=step, speed=speed)
            print(f"  {name:<11} step={step} speed={speed}  ->  {time.time() - t0:5.2f}s")
        except TypeError:
            try:
                crawler.do_action(name, speed)
                print(f"  {name:<11} (old positional API)  ->  {time.time() - t0:5.2f}s")
            except Exception as e:
                print(f"  {name:<11} FAILED: {e}")
        except Exception as e:
            print(f"  {name:<11} FAILED: {e}")
            traceback.print_exc()
        time.sleep(0.3)

    # ------------------------------------------------------------------
    section("6. Ultrasonic")
    try:
        from robot_hat import Ultrasonic, Pin  # type: ignore
        us = Ultrasonic(Pin("D2"), Pin("D3"))
        vals = [us.read() for _ in range(5)]
        print("readings (cm)    :", vals)
    except Exception as e:
        print("ultrasonic       : not available:", e)

    section("Done")
    print("If a motion above took ~1s but the robot did not move: check the")
    print("battery switch / charge and that the servo cables are seated.")
    print("If motions took ~0s: the library found no servos (I2C / robot_hat).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
