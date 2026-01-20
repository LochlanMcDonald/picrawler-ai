#!/usr/bin/env python3
"""Manual step recorder for SunFounder PiCrawler.

Requires: `pip install picrawler` (SunFounder library) and hardware attached.

Controls:
  w/a/s/d: move selected leg in X/Y
  r/f:     move selected leg in Z
  1-4:     select leg
  SPACE:   save the full 4-leg pose
  p:       play saved poses
  ESC:     quit
"""

from picrawler import Picrawler
from time import sleep
import sys
import tty
import termios
import copy

crawler = Picrawler()
SPEED = 80


def readchar() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


MANUAL = '''
Press keys on keyboard to control!
    w: Y++
    a: X--
    s: Y--
    d: X++
    r: Z++
    f: Z--
    1: Select right front leg
    2: Select left front leg
    3: Select left rear leg
    4: Select right rear leg
    Space: Print all leg coodinate & Save this step
    p: Play all saved step
    esc: Quit
'''


new_step = []


def save_new_step() -> None:
    new_step.append(copy.deepcopy(crawler.current_step_all_leg_value()))
    print(new_step)


def play_all_new_step() -> None:
    for step in new_step:
        crawler.do_step(step, SPEED)
        sleep(0.6)


def main() -> None:
    print(MANUAL)
    crawler.do_step('sit', SPEED)

    leg = 0
    coodinate = crawler.current_step_leg_value(leg)

    while True:
        key = readchar().lower()

        if key == 'w':
            coodinate[1] += 2
        elif key == 's':
            coodinate[1] -= 2
        elif key == 'a':
            coodinate[0] -= 2
        elif key == 'd':
            coodinate[0] += 2
        elif key == 'r':
            coodinate[2] += 2
        elif key == 'f':
            coodinate[2] -= 2
        elif key == '1':
            leg = 0
            coodinate = crawler.current_step_leg_value(leg)
        elif key == '2':
            leg = 1
            coodinate = crawler.current_step_leg_value(leg)
        elif key == '3':
            leg = 2
            coodinate = crawler.current_step_leg_value(leg)
        elif key == '4':
            leg = 3
            coodinate = crawler.current_step_leg_value(leg)
        elif key == chr(32):
            print("[[right front],[left front],[left rear],[right rear]]")
            print("saved new step")
            print(crawler.current_step_all_leg_value())
            save_new_step()
        elif key == 'p':
            play_all_new_step()
        elif key == chr(27):
            break

        sleep(0.05)
        crawler.do_single_leg(leg, coodinate, SPEED)

    print("\nQuit")


if __name__ == "__main__":
    main()
