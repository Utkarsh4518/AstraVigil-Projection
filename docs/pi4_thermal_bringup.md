# Thermal camera bring-up on the Raspberry Pi 4

Getting the HIKMICRO Mini2 Plus V2 streaming into AstraVigil, and what changed
porting the working Pi 5 driver across.

## The short version

Nothing in the Pi 5 driver was board-specific. It talks to the camera as a raw
USB device through `pyusb`/`libusb`, which is portable across any Linux — so
the port is not really a port. Everything that changed is about the Pi 4's
slower CPU (Cortex-A72 at 1.5 GHz vs the Pi 5's A76 at 2.4 GHz) and its busier
USB path.

## Setup

```bash
sudo apt install -y python3-numpy python3-opencv libusb-1.0-0
pip install --break-system-packages pyusb

sudo cp configs/udev/99-hikmicro-thermal.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# unplug and replug the camera

python3 scripts/thermal_probe.py 20
```

If the probe prints ambient and calibrated temperatures that move when you
wave a hand in front of the lens, the driver is working.

## Why this bypasses V4L2

The Mini2 Plus V2 is not fully UVC-compliant. `uvcvideo` fails its probe
negotiation, which shows up in `dmesg` as:

```
uvcvideo 1-1.1:1.1: Failed to query (130) UVC probe control : -32 (exp. 34)
```

There are two ways past this and they are mutually exclusive:

1. **The `quirks=2` route** — `options uvcvideo quirks=2` in
   `/etc/modprobe.d/`, which makes `uvcvideo` tolerate the odd negotiation.
   You then read frames through V4L2/ffmpeg as NV12.
2. **The raw USB route** — detach `uvcvideo` entirely and drive the streaming
   endpoint ourselves. This is what the driver does.

This project uses route 2, for a reason that matters beyond the driver: the
V4L2 NV12 path gives you an 8-bit grayscale picture, but the raw endpoint
gives you the **16-bit per-pixel sensor counts**. Those counts are what the
temperature calibration needs, and temperature is a real feature for
discriminating a drone's motor heat from a bird's body heat. Route 1 throws
that away before you ever see it.

Consequence to expect: while the driver holds the camera, `/dev/video*` for it
disappears. That is correct, not a fault. The `quirks=2` modprobe line is
harmless to leave in place but does nothing for this path.

## What changed from the Pi 5 rig

### Whole-frame calibration was a Python loop

The biggest one. The reference viewer applied the temperature curve with
`np.vectorize(calibrate)` over the full frame — that is a Python-level loop
running 256 × 192 = **49,152 calls per frame**. At 25 fps that is over a
million Python calls a second. The Pi 5 got away with it in a diagnostic tool.
The Pi 4 will not, and the detection pipeline needs those cores anyway.

`calibration.calibrate_frame()` does the identical arithmetic as whole-array
numpy operations. Same curve, same constants, same numbers out.

There is also `calibrate_max()`, for the common case of wanting only the
hottest pixel — one scalar through the curve instead of 49k.

### Torn frames are now detected

Each UVC payload header carries a frame-id bit that toggles between frames.
The reference reader ignored it and cut a frame only on the end-of-frame flag.
If that EOF packet is ever dropped, two half-frames get welded into one buffer
that is the right length and completely wrong — it passes the length check and
produces nonsense.

The Pi 4's USB is busier, so this is likelier here. `frame_reader.frames()`
tracks the frame-id and discards a frame whose id flips without an EOF, and
also honours the header's error bit. A dropped frame is cheap; a torn frame
looks to the anomaly detector exactly like a scene that changed violently,
which is the thing we are watching for.

### Fewer, larger USB reads

A frame is 176 KB. At the reference 16 KB chunk that is 11+ round trips
through `pyusb` per frame, 25 times a second. `READ_CHUNK_BYTES` is 32768
here, halving that. If reads turn unreliable, drop it back to `16384` in
`constants.py`.

### The viewer lost its 3D surface plot

The bench viewer rendered a rotatable 3D temperature surface next to the
image. On a Pi 4 that competes for the cores detection needs. The viewer keeps
the part that helps you aim and focus, and gains a `--headless` mode that
writes the latest frame to `/dev/shm` — the normal case when the Pi is on a
roof and you are on SSH.

### Failure is louder

A thermal camera that has silently stopped reads downstream as "no
detections", which is the one failure mode a perimeter system must never be
quiet about. So:

- `ThermalStream.is_healthy()` reports staleness, and `start()` returns
  `False` if no first frame arrives
- read failures warn from *inside* the read loop, since a dead camera yields
  no frames at all and anything watching only the frame stream would sit
  silent forever
- health is judged on *consecutive* read failures, because ordinary timeouts
  accumulate in the lifetime total even on a perfectly healthy feed

## Validated against real hardware (over WSL)

The driver has been run against a real Mini2 Plus V2, though **not yet on a Pi
4** — the camera was passed into WSL2 from Windows over USB/IP. That exercises
the USB handshake, frame reassembly, and calibration; it does not exercise the
Pi's CPU or its USB controller, so the Pi 4 performance claims below are still
inferred rather than measured.

Results, 250 consecutive frames:

| Measure | Result |
|---|---|
| Sustained rate | 26.5 fps (camera's full 25 Hz) |
| Short / torn / error frames | 0 / 0 / 0 |
| Failed reads | 0 |
| `calibrate_frame()` | 0.10 ms/frame |
| Scene | 23.0–31.7 °C, shape (192, 256) |

And the change that motivated the port, benchmarked on a real captured frame:

| Approach | ms/frame | % of one core at 25 fps |
|---|---|---|
| `np.vectorize` (Pi 5 code) | 4.99 | 12 % |
| `calibrate_frame()` (ours) | 0.10 | 0.3 % |

**50× faster, and the outputs differ by 0.000006 °C** — six orders of
magnitude below the sensor's 40 mK noise floor. Those figures are from an
x86-64 laptop; the A72 is several times slower, which is what would have made
the original approach cost most of a core on a Pi 4 doing nothing but
converting temperatures.

The one `short` frame that shows up on a fresh open is expected: the stream is
joined mid-frame, so the first partial frame is discarded rather than passed
on as a corrupt one.

### Testing on a laptop, without a Pi

Bulk endpoints survive USB/IP, so the whole driver can be exercised from
Windows + WSL2. This is worth having — it means driver changes can be tested
without the Pi rig assembled.

```powershell
winget install --exact --id dorssel.usbipd-win
usbipd list                         # find the BUSID for 2bdf:0102
usbipd bind --busid <BUSID>         # needs admin, once
usbipd attach --wsl --busid <BUSID> # each session
```

Then inside WSL (`wsl -u root` avoids needing a sudo password, and root is
needed anyway since `/dev/bus/usb` nodes are `crw-rw-r--`):

```bash
apt-get install -y libusb-1.0-0 python3-usb python3-numpy
cd /mnt/c/path/to/AstraVigil
python3 scripts/thermal_probe.py 20
```

`usbipd detach --busid <BUSID>` gives the camera back to Windows. Note that
while it is attached, Windows apps — including the HIKMICRO app — cannot see
it.

Caveat: this validates protocol and maths, not the Pi. Frame rate and drop
behaviour on the Pi 4's VL805 controller still need checking on the Pi itself.

## Frame format

Each frame is 256 × 344 uint16 little-endian values:

| Rows      | Contents                                            |
|-----------|-----------------------------------------------------|
| 0–191     | thermal sensor data, 256 × 192                      |
| 192       | metadata; cell 0 is ambient in hundredths of a degree |
| 193–343   | trailing rows, unused                               |

Raw counts become degrees C in two steps:

```
linear = ambient + (raw - 4850) / 31.0
celsius = 1.0157 * curve(linear) + 3.37

curve(t) = -0.00294*t² + 1.129*t - 3.12   for t <= 150
           t - 49.92                       for t > 150
```

The gain and offset put readings on the same scale as the HIKMICRO phone app,
which holds the factory calibration. They were derived on the bench rig and
carried over unchanged.

## How far the temperature readings can be trusted

Two different claims live here and they have very different evidence behind
them. Keep them apart, especially on stage.

### The arithmetic is verified

Checked against an independent re-derivation that shares only the raw
constants:

- driver output matches an independent recompute to **5.9e-06 °C** (float32
  rounding, nothing more)
- the piecewise curve is **continuous at the 150 °C knee** to 1.4e-14 — the
  linear branch was deliberately fitted to meet the quadratic exactly
- **monotonic** across -50…500 °C, so no two temperatures collide. The
  quadratic's apex is at 192 °C, safely above the knee where the linear
  branch has already taken over
- frame geometry is confirmed by the camera itself: metadata cells 8 and 9
  read 192 and 256
- `TEMP_SCALE = 31` counts/°C is consistent with the hardware — quantisation
  is 1 count (32 mK), and measured noise lands in the datasheet's range. A
  scale wrong by 10× would put NETD at either 3921 mK or 39 mK, both absurd

### The absolute accuracy is NOT verified

The constants came from the Pi 5 rig, which was a **PCB burn-in bench** —
`bench/instruments/` drives power supplies and an electronic load, and
`thermal_cam.py` describes its output as "max PCB temperature". The curve was
fitted to hot electronics, and it shows:

| Input | Correction applied |
|---|---|
| -20 °C | **-3.93** |
| 0 °C | +0.20 |
| 25 °C | +2.00 |
| 40 °C | +1.29 |
| 100 °C | **-14.99** |
| 150 °C | **-44.98** |
| 400 °C | **-41.05** |

The fit is dominated by the high-temperature regime, where it does enormous
work. Across AstraVigil's actual range — cold sky to warm motor, roughly
-20 to +40 °C — it only ever moves the reading by a couple of degrees, and at
the cold end it is **extrapolating well outside what it was fitted for**.

Nothing here has been checked against a temperature standard. The camera's
internal sensor reports ~16-17 °C while the scene minimum calibrates to
~23 °C, and without knowing the true room temperature there is no way to say
which is closer to right.

**To actually validate it**, in order of value:

1. Point the camera at a scene through the **HIKMICRO phone app** and compare.
   The gain and offset were fitted to match that app, so agreement is the
   real pass criterion.
2. **Ice water at 0.0 °C** — cheap, reliable, and it probes the cold end where
   the fit is weakest.
3. Skin (~33-34 °C palm) as a rough sanity check.

**For detection this mostly does not matter.** Background subtraction and
contrast thresholding are unaffected by a uniform offset. It starts to matter
only if thermal *magnitude* becomes a classification feature — "motor heat
signature" — which is exactly the sort of claim to avoid making until it has
been checked against a reference.

### The camera filters internally — watch this one

**74.7% of pixels are bit-identical between consecutive frames.**

Apparent noise is ~12.6 mK, *below* the datasheet's < 40 mK NETD, which is
only possible because the camera is applying temporal filtering before the
frames reach us.

That is a mixed blessing. It gives a very quiet baseline, so a warm object
against cold sky clears the noise floor easily and SNR is not the limiting
factor — angular resolution at range is. But **temporal filtering smears fast
movers**, and a drone crossing the frame quickly is exactly that. This has not
been tested, and it is the kind of thing that would surface at the worst
moment. Wave something warm rapidly across the frame and check whether it
survives before depending on fast-crossing detection.

## Troubleshooting

**`Access denied (insufficient permissions)`** — the udev rule is not
installed, or the camera was not replugged after installing it.

**`HIKMICRO thermal camera not found`** — check `lsusb` for `2bdf:0102`. If it
is missing, it is a cable or power problem, not software; the Mini2 needs a
data-capable USB-C cable and plenty of charge-only cables look identical.

**`Device or resource busy`** — a previous run did not release the handle.
Replug the camera. If it keeps happening, something else grabbed it first —
check for a stray `thermal_viewer.py` still running.

**Frames counted as `short` or `torn` are climbing** — the camera is sharing
bandwidth. Move it to one of the blue USB 3.0 ports so it is not behind the
same companion controller as everything else, and keep it off any hub shared
with the Pi Camera capture.

**Temperatures look plausible but drift for the first minute** — expected. The
sensor warms up. Let it settle before trusting absolute readings; relative
contrast, which is what detection uses, is usable immediately.

## Reference source

Ported from `ThermalCam/thermal_camera/` — the working Pi 5 bench rig. That
tree also carries an instrument/serial-bus harness for bench power control,
which is unrelated to this project and was not brought across.
