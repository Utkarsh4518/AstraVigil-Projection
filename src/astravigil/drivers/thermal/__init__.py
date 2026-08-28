# HIKMICRO Mini2 Plus V2 thermal driver, targeting the Raspberry Pi 4.
#
# Ported from a working Raspberry Pi 5 bench rig. The USB path is unchanged -
# it was never board-specific - and the changes are all about the Pi 4's
# slower CPU and busier USB controller. See docs/pi4_thermal_bringup.md.
#
# Typical use, on the Pi:
#
#     from astravigil.drivers.thermal import ThermalStream
#
#     with ThermalStream() as stream:
#         raw16, captured_at = stream.latest()
#         celsius = stream.latest_celsius()
#
# The calibration and frame-parsing helpers are pure numpy and import cleanly
# without pyusb, so saved .npy frames can be worked on from a laptop. The
# USB-backed names below are resolved lazily for exactly that reason - an
# eager import would make `pyusb` a hard requirement just to read a frame off
# disk, and most of the detection work happens off the Pi.

from .calibration import (
    ambient_from_frame,
    calibrate,
    calibrate_array,
    calibrate_frame,
    calibrate_max,
    raw_to_linear,
    thermal_pixels,
)
from .constants import FRAME_H, FRAME_W, THERMAL_ROWS

_USB_BACKED = {
    "ThermalStream": ".stream",
    "ThermalCamera": ".usb_device",
    "ThermalCameraError": ".usb_device",
    "open_camera": ".usb_device",
    "close_camera": ".usb_device",
    "find_camera": ".usb_device",
    "FrameStats": ".frame_reader",
    "frames": ".frame_reader",
    "grab_frame": ".frame_reader",
}


def __getattr__(name):
    # PEP 562 lazy attribute lookup.
    module_name = _USB_BACKED.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    try:
        module = import_module(module_name, __name__)
    except ImportError as exc:
        raise ImportError(
            f"{name} needs the USB stack. Install it with "
            f"'pip install pyusb' (and libusb, 'sudo apt install "
            f"libusb-1.0-0'). Original error: {exc}"
        ) from exc

    value = getattr(module, name)
    globals()[name] = value  # cache, so this only happens once
    return value


def __dir__():
    return sorted(list(globals()) + list(_USB_BACKED))


__all__ = [
    # USB-backed, lazily imported
    "ThermalStream",
    "ThermalCamera",
    "ThermalCameraError",
    "FrameStats",
    "frames",
    "grab_frame",
    "open_camera",
    "close_camera",
    "find_camera",
    # pure numpy, always available
    "calibrate",
    "calibrate_array",
    "calibrate_frame",
    "calibrate_max",
    "thermal_pixels",
    "ambient_from_frame",
    "raw_to_linear",
    "FRAME_W",
    "FRAME_H",
    "THERMAL_ROWS",
]
