"""Frame sources: one interface, several backends.

Every backend returns the same pair - (thermal_raw16, optical_bgr) - so the
pipeline, the detector and the dashboard never learn where frames came from.
That is what lets the whole system be developed and demonstrated on Windows
and then run unchanged on the Pi.

  synthetic  simulated scene, ground-truth homography available   any OS
  replay     saved .npy thermal frames + optional video/webcam     any OS
  hardware   HIKMICRO over USB + Pi Camera                         Linux

Windows cannot drive the thermal camera without replacing its driver with
WinUSB (Zadig), which then breaks it for every other Windows app. Rather than
ask anyone to do that, "hardware" is Linux-only by design and the other two
backends exist to make that not matter.
"""
import glob
import os
import platform

import cv2
import numpy as np


class Source:
    """Returns (thermal_raw16 (344,256) uint16, optical_bgr HxWx3 uint8)."""

    name = "source"
    truth_homography = None      # only simulation knows this

    def frames(self):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ------------------------------------------------------------------ synthetic
class SyntheticSource(Source):
    name = "synthetic"

    def __init__(self, seed=0, fps=25.0, scenario="patrol"):
        from .sim import SyntheticScene
        self.scene = SyntheticScene(seed=seed, scenario=scenario)
        self.name = f"synthetic:{scenario}"
        self.dt = 1.0 / fps
        self.truth_homography = self.scene.H_truth

    def frames(self):
        self.scene.step(self.dt)
        return self.scene.frames()

    def calibration_points(self, n=8, jitter_px=0.0):
        return self.scene.calibration_points(n=n, jitter_px=jitter_px)


# --------------------------------------------------------------------- replay
class ReplaySource(Source):
    """Saved thermal .npy frames, paired with a webcam or a video file.

    Useful once you have captured real thermal footage: the thermal side is
    genuine and the optical side can be whatever is to hand, which is enough
    to exercise detection on real sensor noise.
    """

    name = "replay"

    def __init__(self, thermal_dir="data/raw/thermal", optical=None,
                 loop=True):
        self.paths = sorted(glob.glob(os.path.join(thermal_dir, "*.npy")))
        if not self.paths:
            raise FileNotFoundError(
                f"no .npy thermal frames in {thermal_dir} - capture some with "
                f"scripts/thermal_probe.py, or use --source synthetic")
        self.loop = loop
        self.i = 0
        self.optical_cap = None
        if optical is not None:
            self.optical_cap = cv2.VideoCapture(
                int(optical) if str(optical).isdigit() else optical)
            if not self.optical_cap.isOpened():
                self.optical_cap = None

    def frames(self):
        if self.i >= len(self.paths):
            if not self.loop:
                raise StopIteration
            self.i = 0
        raw = np.load(self.paths[self.i])
        self.i += 1

        optical = None
        if self.optical_cap is not None:
            ok, optical = self.optical_cap.read()
            if not ok:
                optical = None
        if optical is None:
            # Grey field rather than a crash: with no optical feed the thermal
            # half of the pipeline still runs and is still worth watching.
            optical = np.full((480, 640, 3), 60, np.uint8)
            cv2.putText(optical, "no optical source", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return raw, optical

    def close(self):
        if self.optical_cap is not None:
            self.optical_cap.release()


# ------------------------------------------------------------------- hardware
class HardwareSource(Source):
    """Real HIKMICRO over USB plus a real optical camera. Linux only."""

    name = "hardware"

    def __init__(self, optical_index=0, use_picamera=True, swap_rb=False):
        if platform.system() == "Windows":
            raise RuntimeError(
                "the thermal camera cannot be opened on Windows without "
                "replacing its driver with WinUSB, which breaks it for every "
                "other Windows app. Use --source synthetic here and run "
                "--source hardware on the Pi.")

        from .drivers.thermal import ThermalStream
        self.stream = ThermalStream()
        # Cold Pi 4: opening the device, committing the format and getting the
        # first frame out is not instant, and 5 s was tight enough to fail on
        # hardware that then worked fine by hand.
        wait_s = float(os.environ.get("ASTRAVIGIL_THERMAL_WAIT_S", "12"))
        if not self.stream.start(wait_s=wait_s):
            # Stop the capture thread BEFORE raising. It is holding the USB
            # device; letting the exception escape leaves it running, and the
            # interpreter then tears a daemon thread down in the middle of a
            # libusb call. That aborts the process
            # ("libusb_ref_device: Assertion `refcnt >= 2' failed"), which
            # skips cleanup and leaves the camera claimed by a dead process -
            # so every later attempt fails too, and the original cause is
            # buried under a crash that looks unrelated.
            self.stream.stop()
            raise RuntimeError(
                f"thermal camera opened but produced no frame within "
                f"{wait_s:.0f} s.\n"
                f"  Try:  python3 scripts/thermal_probe.py 20\n"
                f"  If that works, raise ASTRAVIGIL_THERMAL_WAIT_S.\n"
                f"  If it does not, see docs/pi4_thermal_bringup.md")

        self.swap_rb = swap_rb
        self.picam = None
        self.cap = None
        if use_picamera:
            try:
                from picamera2 import Picamera2
                self.picam = Picamera2()
                # picamera2's format names follow libcamera's convention, which
                # is the REVERSE of what they look like: "RGB888" delivers a
                # numpy array in B, G, R byte order - already what OpenCV
                # expects. Converting it again turns skin blue.
                self.picam.configure(self.picam.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}))
                self.picam.start()
            except Exception:
                self.picam = None
        if self.picam is None:
            self.cap = cv2.VideoCapture(optical_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._last_raw = None

    def frames(self):
        raw, _ = self.stream.latest()
        if raw is None:
            raw = self._last_raw
        if raw is None:
            raise RuntimeError("no thermal frame available yet")
        self._last_raw = raw

        if self.picam is not None:
            # Already B, G, R - see the format note in __init__. No conversion.
            optical = self.picam.capture_array()
        else:
            ok, optical = self.cap.read()
            if not ok:
                optical = np.full((480, 640, 3), 60, np.uint8)

        # Escape hatch: sensor and libcamera version combinations vary, so if
        # colours still come out inverted, --swap-rb flips them rather than
        # requiring a code edit at the rig.
        if self.swap_rb and optical is not None and optical.ndim == 3:
            optical = optical[:, :, ::-1].copy()
        return raw, optical

    @property
    def healthy(self):
        return self.stream.is_healthy()

    def close(self):
        self.stream.stop()
        if self.picam is not None:
            self.picam.stop()
        if self.cap is not None:
            self.cap.release()


def create(kind="synthetic", **kw):
    kinds = {"synthetic": SyntheticSource,
             "replay": ReplaySource,
             "hardware": HardwareSource}
    if kind not in kinds:
        raise ValueError(f"unknown source {kind!r}, expected one of "
                         f"{sorted(kinds)}")
    return kinds[kind](**kw)
