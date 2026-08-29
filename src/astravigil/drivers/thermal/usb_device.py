# Claiming the HIKMICRO thermal camera and giving it back.
#
# This talks to the camera as a raw USB device rather than through V4L2. That
# is deliberate: the Mini2 Plus V2 is not fully UVC-compliant and uvcvideo
# fails its probe negotiation (the "Failed to query (130) UVC probe control"
# error in dmesg). Detaching the kernel driver and driving the endpoint
# ourselves sidesteps that entirely, and gives us the raw 16-bit thermal
# values that the V4L2 NV12 path throws away.
#
# Because we detach uvcvideo, the "options uvcvideo quirks=2" modprobe fix is
# not needed for this driver, and /dev/video* will disappear while we hold the
# camera. That is expected. See docs/pi4_thermal_bringup.md.

import time

import usb.core
import usb.util

from .constants import (
    COMMIT_CONTROL,
    EXPECTED_FRAME_BYTES,
    FRAME_H,
    FRAME_W,
    PRODUCT_ID,
    PROBE_CONTROL,
    PROBE_FORMAT_INDEX,
    PROBE_FRAME_INDEX,
    PROBE_LENGTH,
    PROBE_MAX_FRAME_SIZE,
    STREAM_FORMAT_INDEX,
    STREAM_FRAME_INDEX,
    STREAM_INTERFACE,
    UVC_GET_CUR,
    UVC_REQ_TYPE_GET,
    UVC_REQ_TYPE_SET,
    UVC_SET_CUR,
    VENDOR_ID,
)

# The Pi 4 brings USB devices up over the VL805 PCIe controller, which can be
# a little slower to settle than the Pi 5's RP1. Retry the format handshake
# rather than failing the first time it is not ready.
HANDSHAKE_ATTEMPTS = 3
HANDSHAKE_BACKOFF_S = 0.4
SETTLE_AFTER_COMMIT_S = 0.5


class ThermalCameraError(RuntimeError):
    # Raised when the camera is present but will not start streaming.
    pass


def find_camera():
    # The device, or None when it is not plugged in.
    return usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)


def open_camera():
    # Find the camera, take it off the kernel driver, and get it streaming.
    # Returns (dev, detached) where detached is the list of interface numbers
    # we took, so close_camera can hand them back.
    dev = find_camera()
    if dev is None:
        return None, []

    detached = _detach_kernel_drivers(dev)

    try:
        dev.set_configuration()
    except usb.core.USBError as exc:
        # Almost always a permissions problem when running as a normal user.
        _reattach(dev, detached)
        raise ThermalCameraError(
            f"cannot configure the thermal camera ({exc}). If this says "
            f"'Access denied', install the udev rule from "
            f"configs/udev/99-hikmicro-thermal.rules and replug the camera."
        ) from exc

    try:
        _negotiate_format(dev)
        usb.util.claim_interface(dev, STREAM_INTERFACE)
    except Exception:
        _reattach(dev, detached)
        raise

    return dev, detached


def _detach_kernel_drivers(dev):
    # The kernel grabs this as a webcam, so take every interface back first.
    detached = []
    for cfg in dev:
        for intf in cfg:
            number = intf.bInterfaceNumber
            try:
                if dev.is_kernel_driver_active(number):
                    dev.detach_kernel_driver(number)
                    detached.append(number)
            except Exception:
                # Some interfaces have no driver attached; nothing to do.
                pass
    return detached


def _check_negotiated(probe):
    # dwMaxVideoFrameSize is the camera's own statement of how big one frame in
    # the mode it has just accepted will be. If that is smaller than the frame
    # the reader cuts, no frame can ever be complete: every one is counted
    # short and the stream looks silent rather than misconfigured, which is a
    # far harder thing to diagnose than an error at open time.
    size = int.from_bytes(
        bytes(probe[PROBE_MAX_FRAME_SIZE:PROBE_MAX_FRAME_SIZE + 4]), "little")

    # Zero means the camera never filled the field in. This device is not fully
    # UVC-compliant, and no answer is not the same as a wrong answer.
    if not size or size >= EXPECTED_FRAME_BYTES:
        return

    raise ThermalCameraError(
        f"the camera negotiated format {probe[PROBE_FORMAT_INDEX]}, frame "
        f"{probe[PROBE_FRAME_INDEX]}, which carries {size} bytes per frame. "
        f"The {FRAME_W}x{FRAME_H} 16-bit stream this driver reads needs "
        f"{EXPECTED_FRAME_BYTES}, so no frame would ever complete."
        f"\n  Check which indices carry the raw thermal mode on this unit:"
        f"\n      lsusb -v -d {VENDOR_ID:04x}:{PRODUCT_ID:04x}"
        f"\n  and re-run with the pair that carries it:"
        f"\n      ASTRAVIGIL_FORMAT_INDEX=n ASTRAVIGIL_FRAME_INDEX=n"
        f" python3 scripts/thermal_probe.py 20"
    )


def _negotiate_format(dev):
    # Ask for the raw 16-bit thermal stream, then commit what the camera
    # negotiates back.
    #
    # This used to read the camera's current format and hand it straight back,
    # on the assumption that the power-on default was already the mode we want.
    # It is not. The Mini2 Plus V2 comes up on its NV12 256x192 mode (format 2,
    # frame 2) and advertises the raw 256x344 16-bit stream separately as
    # format 1, frame 1. Committing the default started a 73728-byte NV12
    # stream while frame_reader was cutting frames at 176128 bytes, so every
    # frame was counted short and none was ever yielded: thermal_probe reported
    # `frames=0, short=499`, ThermalStream.start() timed out waiting for a
    # first frame, and run_dashboard died before it ever bound its port - which
    # is why the kiosk had nothing to open.
    #
    # The GET between the two SETs is the step that is easy to leave out and is
    # load-bearing: the camera fills in dwMaxVideoFrameSize and
    # dwMaxPayloadTransferSize for the mode it has just accepted, and the
    # commit has to carry those, not the NV12 numbers we started from.
    #
    #   GET_CUR probe -> set format/frame -> SET_CUR probe
    #                 -> GET_CUR probe -> SET_CUR commit
    last_error = None

    for attempt in range(HANDSHAKE_ATTEMPTS):
        try:
            probe = bytearray(dev.ctrl_transfer(
                UVC_REQ_TYPE_GET, UVC_GET_CUR,
                PROBE_CONTROL, STREAM_INTERFACE, PROBE_LENGTH, timeout=1000,
            ))
            probe[PROBE_FORMAT_INDEX] = STREAM_FORMAT_INDEX
            probe[PROBE_FRAME_INDEX] = STREAM_FRAME_INDEX

            dev.ctrl_transfer(
                UVC_REQ_TYPE_SET, UVC_SET_CUR,
                PROBE_CONTROL, STREAM_INTERFACE, bytes(probe), timeout=1000,
            )
            negotiated = bytearray(dev.ctrl_transfer(
                UVC_REQ_TYPE_GET, UVC_GET_CUR,
                PROBE_CONTROL, STREAM_INTERFACE, PROBE_LENGTH, timeout=1000,
            ))
            dev.ctrl_transfer(
                UVC_REQ_TYPE_SET, UVC_SET_CUR,
                COMMIT_CONTROL, STREAM_INTERFACE, bytes(negotiated),
                timeout=1000,
            )
            time.sleep(SETTLE_AFTER_COMMIT_S)

            # Deliberately outside the USBError retry: a mode the camera has
            # agreed to that is still the wrong size is a configuration answer,
            # not a transient failure, and asking twice more will not fix it.
            _check_negotiated(negotiated)
            return bytes(negotiated)
        except usb.core.USBError as exc:
            last_error = exc
            if attempt < HANDSHAKE_ATTEMPTS - 1:
                time.sleep(HANDSHAKE_BACKOFF_S)

    raise ThermalCameraError(
        f"thermal camera did not accept the streaming format after "
        f"{HANDSHAKE_ATTEMPTS} attempts ({last_error})"
    )


def _reattach(dev, detached):
    # Best-effort cleanup on a failed open, so the next run starts clean.
    for number in detached:
        try:
            dev.attach_kernel_driver(number)
        except Exception:
            pass
    try:
        usb.util.dispose_resources(dev)
    except Exception:
        pass


def close_camera(dev, detached=(), warn=print):
    # Undo open_camera, in reverse, without ever raising.
    if dev is None:
        return

    # Every step runs even if an earlier one fails - skipping the rest would
    # leave the camera busy for the next run.
    def attempt(what, action):
        try:
            action()
        except Exception as exc:
            warn(f"thermal camera {what} failed ({exc}) - "
                 f"the next run may find the camera still busy")

    attempt("interface release",
            lambda: usb.util.release_interface(dev, STREAM_INTERFACE))

    for number in detached:
        attempt(f"kernel driver reattach on interface {number}",
                lambda n=number: dev.attach_kernel_driver(n))

    # Without this the handle leaks and a re-open fails with "Device busy".
    attempt("handle release", lambda: usb.util.dispose_resources(dev))


class ThermalCamera:
    # Context manager wrapper, so callers cannot forget to hand the camera
    # back. Every tool in scripts/ uses this form.
    #
    #   with ThermalCamera() as cam:
    #       for raw16 in frames(cam.dev):
    #           ...

    def __init__(self, warn=print):
        self.dev = None
        self.detached = []
        self._warn = warn

    def __enter__(self):
        self.dev, self.detached = open_camera()
        if self.dev is None:
            raise ThermalCameraError(
                "HIKMICRO thermal camera not found - check it is plugged in "
                "and shows up in lsusb as 2bdf:0102"
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        close_camera(self.dev, self.detached, warn=self._warn)
        self.dev = None
        self.detached = []
        return False
