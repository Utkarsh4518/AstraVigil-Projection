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
    PRODUCT_ID,
    PROBE_CONTROL,
    PROBE_LENGTH,
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
        _commit_current_format(dev)
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


def _commit_current_format(dev):
    # Read back the format the camera is already set to and hand it straight
    # back to commit it. We never try to negotiate a different format - the
    # camera's own default is the 256x344 16-bit stream we want, and probing
    # for anything else is what upsets this device.
    last_error = None

    for attempt in range(HANDSHAKE_ATTEMPTS):
        try:
            probe = dev.ctrl_transfer(
                UVC_REQ_TYPE_GET, UVC_GET_CUR,
                PROBE_CONTROL, STREAM_INTERFACE, PROBE_LENGTH, timeout=1000,
            )
            dev.ctrl_transfer(
                UVC_REQ_TYPE_SET, UVC_SET_CUR,
                COMMIT_CONTROL, STREAM_INTERFACE, probe, timeout=1000,
            )
            time.sleep(SETTLE_AFTER_COMMIT_S)
            return probe
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
