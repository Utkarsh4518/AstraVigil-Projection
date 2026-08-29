#!/usr/bin/env python3
# Live thermal viewer for the Raspberry Pi 4.
#
#   python3 scripts/thermal_viewer.py
#   python3 scripts/thermal_viewer.py --headless    # over SSH, no display
#
# Keys: q=quit, s=save, m=processing mode, c=colormap
#
# Trimmed down from the Pi 5 bench viewer, which also rendered a rotatable 3D
# surface plot beside the image. That was diagnostic eye candy for a bench rig
# with CPU to spare; on an A72 it competes with the capture thread for the
# cores this project needs for detection. What is left is the part that
# actually helps you aim and focus the camera.
#
# Two other Pi 4 economies worth knowing about, both in _render():
#   - upscaling uses INTER_LINEAR, not INTER_LANCZOS4
#   - the full-frame temperature map is computed only when it is displayed
#
# Focus is manual on this camera and the useful range tops out around 10 m,
# so getting a sharp image here is a physical job, not a software one.

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from astravigil.drivers.thermal import FrameStats, ThermalCamera, frames
from astravigil.drivers.thermal.calibration import calibrate_frame
from astravigil.drivers.thermal.constants import FRAME_W, THERMAL_ROWS

WINDOW = "AstraVigil - thermal"
DEFAULT_SCALE = 2          # 512x384, comfortable on a Pi 4
INFO_BAR_H = 40
SAVE_DIR = "data/raw/thermal"
HEADLESS_FRAME = "/dev/shm/astravigil_thermal.png"
HEADLESS_INTERVAL_S = 0.2  # ~5 fps is plenty for a remote look

COLORMAPS = [
    (cv2.COLORMAP_INFERNO, "INFERNO"),
    (cv2.COLORMAP_JET, "JET"),
    (cv2.COLORMAP_HOT, "HOT"),
    (cv2.COLORMAP_TURBO, "TURBO"),
]
MODES = ["CLAHE", "Normalize", "HistEq", "Raw"]


def enhance(thermal_norm, mode, clahe):
    if mode == 0:
        return clahe.apply(thermal_norm)
    if mode == 1:
        return cv2.normalize(thermal_norm, None, 0, 255, cv2.NORM_MINMAX)
    if mode == 2:
        return cv2.equalizeHist(thermal_norm)
    return thermal_norm


def _render(raw16, mode, colormap_idx, clahe, scale):
    # One frame to a displayable BGR image, plus the numbers for the info bar.
    thermal = raw16[:THERMAL_ROWS, :]

    raw_min = int(thermal.min())
    raw_max = int(thermal.max())
    min_loc = np.unravel_index(int(np.argmin(thermal)), thermal.shape)
    max_loc = np.unravel_index(int(np.argmax(thermal)), thermal.shape)

    celsius = calibrate_frame(raw16)
    t_min = float(celsius.min())
    t_max = float(celsius.max())
    t_avg = float(celsius.mean())

    span = max(raw_max - raw_min, 1)
    norm = ((thermal.astype(np.float32) - raw_min) * (255.0 / span)).astype(np.uint8)
    colored = cv2.applyColorMap(enhance(norm, mode, clahe),
                                COLORMAPS[colormap_idx][0])

    _mark(colored, max_loc, f"{t_max:.1f}C", (0, 0, 255))
    _mark(colored, min_loc, f"{t_min:.1f}C", (255, 100, 0))

    scaled = cv2.resize(colored, (FRAME_W * scale, THERMAL_ROWS * scale),
                        interpolation=cv2.INTER_LINEAR)

    return np.vstack([
        _info_bar(scaled.shape[1], t_max, t_min, t_avg, colormap_idx, mode),
        scaled,
    ])


def _mark(img, loc, label, color):
    y, x = int(loc[0]), int(loc[1])
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 12, 2)
    tx = x + 8 if x < FRAME_W - 50 else x - 45
    ty = y + 4 if y > 10 else y + 18
    if ty > THERMAL_ROWS - 5:
        ty = y - 8
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def _info_bar(width, t_max, t_min, t_avg, colormap_idx, mode):
    bar = np.zeros((INFO_BAR_H, width, 3), dtype=np.uint8)
    fields = [
        (f"MAX:{t_max:.1f}C", 10, (0, 0, 255)),
        (f"MIN:{t_min:.1f}C", 130, (255, 100, 0)),
        (f"AVG:{t_avg:.1f}C", 250, (230, 230, 230)),
        (f"D:{t_max - t_min:.1f}C", 370, (150, 150, 180)),
    ]
    for text, x, color in fields:
        if x + 100 < width:
            cv2.putText(bar, text, (x, 27), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2)

    tag = f"{COLORMAPS[colormap_idx][1]} / {MODES[mode]}"
    if width > 600:
        cv2.putText(bar, tag, (width - 200, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (110, 110, 140), 1)
    return bar


def run_windowed(dev, scale):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    stats = FrameStats()
    mode = 0
    colormap_idx = 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, FRAME_W * scale, THERMAL_ROWS * scale + INFO_BAR_H)

    print("\nStreaming live...")
    print("Keys: q=quit, s=save, m=mode, c=colormap")

    for raw16 in frames(dev, stats=stats):
        cv2.imshow(WINDOW, _render(raw16, mode, colormap_idx, clahe, scale))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            _save(raw16, stats.frames)
        elif key == ord("m"):
            mode = (mode + 1) % len(MODES)
            print(f"Mode: {MODES[mode]}")
        elif key == ord("c"):
            colormap_idx = (colormap_idx + 1) % len(COLORMAPS)
            print(f"Colormap: {COLORMAPS[colormap_idx][1]}")

    cv2.destroyAllWindows()
    print(f"\n{stats}")


def run_headless(dev, scale):
    # No display, so publish the latest frame to /dev/shm and let something
    # else pick it up - scp it, or serve it. /dev/shm is a ramdisk, so this
    # does not chew through the SD card.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    stats = FrameStats()
    tmp = HEADLESS_FRAME + ".tmp.png"
    last_write = 0.0

    print(f"Headless. Latest frame goes to {HEADLESS_FRAME}")
    print("Ctrl+C to stop.\n")

    try:
        for raw16 in frames(dev, stats=stats):
            now = time.monotonic()
            if now - last_write < HEADLESS_INTERVAL_S:
                continue
            last_write = now

            cv2.imwrite(tmp, _render(raw16, 0, 0, clahe, scale))
            # Rename is atomic, so a reader never catches a half-written file.
            os.replace(tmp, HEADLESS_FRAME)

            if stats.frames % 100 == 0:
                print(f"  {stats}")
    finally:
        for path in (tmp, HEADLESS_FRAME):
            try:
                os.remove(path)
            except OSError:
                pass


def _save(raw16, index):
    os.makedirs(SAVE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"thermal_{stamp}_{index}.npy")
    # Save the raw counts, not the picture. The colour map is a display
    # choice; the 16-bit values are the measurement, and every later stage
    # (calibration, detection, training data) wants those.
    np.save(path, raw16)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Live thermal viewer")
    parser.add_argument("--headless", action="store_true",
                        help="no window; write the latest frame to /dev/shm")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE,
                        help=f"display upscale factor (default {DEFAULT_SCALE})")
    args = parser.parse_args()

    with ThermalCamera() as cam:
        if args.headless:
            run_headless(cam.dev, args.scale)
        else:
            run_windowed(cam.dev, args.scale)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
