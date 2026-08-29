#!/usr/bin/env python3
"""Put the thermal camera back to a state the next run can claim.

    python3 scripts/release_camera.py

The driver claims the HIKMICRO by detaching uvcvideo from its interfaces and
taking the USB endpoint directly. Everything is handed back in close_camera()
- interface released, kernel driver reattached, handle disposed - and on a
clean exit that is all that is needed.

It is the unclean exits this exists for. A process killed with SIGKILL never
runs its cleanup, so the interfaces stay detached and the kernel still
believes a dead process owns them. The next launch then fails with "Device or
resource busy" or "Access denied" on hardware that is working perfectly, and
the only obvious fix - physically replugging the camera - is not available
when the rig is on a roof.

A USB port reset makes the kernel re-enumerate the device, which drops the
stale claim and reattaches uvcvideo. It needs no root: the udev rule in
configs/udev already grants the device to the local user, and a reset takes
the same permission as opening it.

Safe to run when nothing is wrong - resetting an idle camera costs a second.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

VENDOR_ID = 0x2BDF
PRODUCT_ID = 0x0102


def main():
    try:
        import usb.core
        import usb.util
    except ImportError:
        print("pyusb is not installed - nothing to release")
        return 0

    dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    if dev is None:
        print("thermal camera not present (2bdf:0102) - nothing to release")
        return 0

    # Reattach anything still detached, so the kernel driver is back even if
    # the reset below is refused.
    reattached = 0
    for cfg in dev:
        for intf in cfg:
            n = intf.bInterfaceNumber
            try:
                if not dev.is_kernel_driver_active(n):
                    dev.attach_kernel_driver(n)
                    reattached += 1
            except Exception:
                # Already attached, or the interface has no driver. Either way
                # there is nothing to fix here.
                pass
    if reattached:
        print(f"reattached kernel driver on {reattached} interface(s)")

    try:
        dev.reset()
        print("thermal camera reset - ready for the next run")
    except Exception as exc:
        print(f"could not reset the camera ({exc})")
        print("If the next run reports the device is busy, unplug and replug "
              "it, or check the udev rule in configs/udev/.")
        return 1

    try:
        usb.util.dispose_resources(dev)
    except Exception:
        pass

    # Re-enumeration is not instant; give the kernel a moment so an immediate
    # relaunch does not race it.
    time.sleep(1.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
