#!/usr/bin/env python3
# Shows the working out behind a thermal reading.
#
# Run this first on a fresh Pi 4. It answers, in order: is the camera there,
# will it stream, are the frames intact, and do the temperatures look sane.
#
#   python3 scripts/thermal_probe.py            # 60 seconds
#   python3 scripts/thermal_probe.py 10         # 10 seconds
#
# The reading drifts for the first minute or two while the sensor warms up,
# which is normal and worth watching for rather than being surprised by.

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from astravigil.drivers.thermal import FrameStats, ThermalCamera, grab_frame
from astravigil.drivers.thermal.calibration import (
    ambient_from_frame,
    calibrate,
    raw_to_linear,
    thermal_pixels,
)
from astravigil.drivers.thermal.constants import METADATA_ROW

DEFAULT_SAVE_TO = "data/raw/thermal/probe_frame.npy"


def describe(raw16, elapsed):
    # Print the same sum the driver does, one step at a time, so a wrong
    # number can be traced to the stage that produced it.
    thermal = thermal_pixels(raw16)
    ambient = ambient_from_frame(raw16)
    raw_min = int(thermal.min())
    raw_max = int(thermal.max())

    linear = raw_to_linear(float(raw_max), ambient)
    final = calibrate(linear)

    print(f"t+{elapsed:5.1f}s  ambient {ambient:6.2f} C   "
          f"raw {raw_min}..{raw_max}   "
          f"linear {linear:6.2f} C   calibrated {final:6.2f} C")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seconds", nargs="?", type=float, default=60.0,
                        help="how long to watch for (default 60)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="seconds between readings (default 2)")
    parser.add_argument("--save-to", default=DEFAULT_SAVE_TO,
                        help=f"where to save the last raw frame "
                             f"(default {DEFAULT_SAVE_TO})")
    args = parser.parse_args()

    stats = FrameStats()
    last = None

    with ThermalCamera() as cam:
        print(f"Watching for {args.seconds:.0f}s. Point it at something and "
              f"leave it be,")
        print("the reading drifts while the camera warms up.\n")

        started = time.monotonic()
        while time.monotonic() - started < args.seconds:
            raw16 = grab_frame(cam.dev, wait_s=5.0, stats=stats)
            if raw16 is None:
                print("  no frame")
                continue
            last = raw16
            describe(raw16, time.monotonic() - started)
            time.sleep(args.interval)

    print(f"\n{stats}")
    if stats.torn_frames or stats.short_frames:
        # Worth flagging rather than burying: on the Pi 4 this usually means
        # the camera is sharing a controller with something greedy.
        print("Some frames were dropped or torn. If that count is climbing, "
              "try a different USB port (see docs/pi4_thermal_bringup.md).")

    if last is not None:
        os.makedirs(os.path.dirname(args.save_to) or ".", exist_ok=True)
        np.save(args.save_to, last)
        print(f"\nSaved one raw frame to {args.save_to}")
        print("First 16 values of the metadata row:")
        print(" ", [int(v) for v in last[METADATA_ROW, :16]])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped")
