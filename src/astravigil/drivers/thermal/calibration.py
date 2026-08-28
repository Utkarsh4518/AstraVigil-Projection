# Raw sensor counts to degrees C.
#
# The correction curve and its two reference constants are carried over
# unchanged from the working Pi 5 rig - they put readings on the same scale as
# the HIKMICRO app, which holds the factory calibration.
#
# What changed for the Pi 4 is how the curve is applied. The reference viewer
# used np.vectorize(calibrate) over the whole frame, which is a Python-level
# loop: 256 x 192 = 49,152 calls per frame, 25 frames a second. That is over a
# million Python calls a second and the A72 will not keep up. calibrate_frame()
# below does the same arithmetic with whole-array numpy ops instead.

import numpy as np

from .constants import AMBIENT_SCALE, METADATA_ROW, TEMP_BASELINE, TEMP_SCALE, THERMAL_ROWS

# Puts us on the same scale as the HIKMICRO app, which has the factory cal.
REFERENCE_GAIN = 1.0157
REFERENCE_OFFSET = 3.37

# The curve is piecewise: quadratic up to 150 C, linear above it.
CURVE_KNEE = 150.0


def calibrate(t):
    # Scalar form. Raw linear temperature to degrees C.
    # Kept because the probe tool and the log writer work one reading at a
    # time, and because it documents the curve more plainly than the array
    # version does.
    if t <= CURVE_KNEE:
        curve = -0.00294 * t * t + 1.129 * t - 3.12
    else:
        curve = t - 49.92
    return REFERENCE_GAIN * curve + REFERENCE_OFFSET


def calibrate_array(t):
    # Same curve, whole array at a time. This is the one the live pipeline
    # uses. np.where evaluates both branches, which is still vastly cheaper
    # than crossing into Python once per pixel.
    t = np.asarray(t, dtype=np.float32)
    curve = np.where(
        t <= CURVE_KNEE,
        -0.00294 * t * t + 1.129 * t - 3.12,
        t - 49.92,
    )
    return REFERENCE_GAIN * curve + REFERENCE_OFFSET


def ambient_from_frame(raw16):
    # Camera's own ambient reading, from the first cell of the metadata row.
    return float(raw16[METADATA_ROW, 0]) / AMBIENT_SCALE


def thermal_pixels(raw16):
    # Just the sensor rows, without the metadata and padding rows below them.
    return raw16[:THERMAL_ROWS, :]


def raw_to_linear(raw, ambient):
    # Raw counts to a linear temperature, before the correction curve.
    return ambient + (raw - TEMP_BASELINE) / TEMP_SCALE


def calibrate_frame(raw16):
    # Full frame to a 192x256 float32 array of degrees C.
    #
    # Costs roughly one pass over 49k float32 values. Fine on a Pi 4 at 25 Hz;
    # the np.vectorize version it replaces was not.
    thermal = thermal_pixels(raw16)
    ambient = ambient_from_frame(raw16)
    linear = raw_to_linear(thermal.astype(np.float32), ambient)
    return calibrate_array(linear)


def calibrate_max(raw16):
    # Hottest pixel only, in degrees C.
    #
    # Detection cares about the whole field, but tracking a single max is much
    # cheaper - one scalar through the curve instead of 49k - so anything that
    # only needs the peak should call this rather than calibrate_frame().
    thermal = thermal_pixels(raw16)
    ambient = ambient_from_frame(raw16)
    return calibrate(raw_to_linear(float(thermal.max()), ambient))
