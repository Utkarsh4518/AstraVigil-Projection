# AstraVigil — project context

## Event
European Defense Tech Hackathon, Hamburg. Team name: **AstraVigil**. GitHub repo: `astravigil-cuas`.

## Challenge
**Challenge 04 — Protecting a Critical Site From Small Drones**

> A bomb-carrying quadcopter crossed a major cargo airport undetected and sat on the tarmac for hours. Fixed critical sites cannot reliably see the small, low-signature intruder. Design better protection for one fixed critical site — an airport apron, a port, or an energy node. Build an affordable, layered detect-classify-alert system that catches the small, low-signature intruder the current perimeter misses, and fuses its sensors into a single coherent situational picture rather than a wall of separate alarms.

We chose this over challenges 01 (unjammable drone, 50-100km, needs defeat mechanism), 02 (maritime/AIS), 03 (disinformation), 05 (last-metre defense) — 04 is the closest word-for-word match to what we can build with our hardware, no defeat mechanism required, no unrealistic range requirement.

**Site framing**: rooftop-mounted sensor unit protecting an **airport apron perimeter** (matches the brief's own tarmac scenario).

**Tagline**: "The drone sat on the tarmac for hours. We built the eyes that would've caught it in seconds." (under 115 characters)

## Team
Utkarsh Maurya — MSc Information and Communication Systems, TUHH Hamburg. Background: RF/DSP, link-budget analysis, LTE/5G, OMNeT++ MAC protocol evaluation, wireless localization (Eppendorf project), PyTorch/LangChain.

## Hardware available
- Laptop
- 3 phones
- **HIKmicro Mini2 Plus V2** thermal camera (USB-C phone attachment, but confirmed working via USB on Raspberry Pi — see below)
- **Raspberry Pi** with **Raspberry Pi Camera Module 2** (optical)
- No SDR — ruled out RF-based approaches (fingerprinting, GNSS spoofing detection) early on due to lack of SDR hardware

## HIKmicro Mini2 Plus V2 specs
Verified against the HIKMICRO published specification:

- Body: **26.6 x 26.6 x 25 mm**, 24 g
- Thermal resolution: 256 x 192 pixels, **12 µm pixel pitch**
- NETD: < 40 mK @ 25°C, F#=1.0
- Lens: **6.9 mm, f/1.0**, FOV 25° x 18.8°
- **Focus range: 0.1 m to 50 m**
- Frame rate: 25 Hz
- Temperature range: -20°C to +400°C
- Interface: USB-C

**CORRECTION — the "7 cm to 10 m" figure previously recorded here was wrong.** The manufacturer specifies **0.1 m to 50 m**. This matters because the whole "~10 m close-range confirm layer" pitch framing was built on it.

Independent optics check agrees. Deriving the lens from the published FOV and pixel pitch gives 6.9 mm — exactly the specified focal length — and at f/1.0 the **hyperfocal distance is ~4 m**, so with focus at the far setting everything from ~2 m to infinity is sharp. There was never a 10 m focus wall.

**The real limit is angular resolution, not focus.** IFOV is 1.70 mrad/px:

| Target | detect (~2 px) | classify (~8 px) |
|---|---|---|
| DJI Mini class (0.25 m) | ~73 m | ~18 m |
| Mavic class (0.35 m) | ~103 m | ~26 m |

**Revised pitch framing**: detection to roughly 50-70 m (focus supports 50 m, geometry supports ~73 m), shape classification to roughly 20 m. Still honestly a short-to-medium range layer that a production system would pair with longer-range radar/RF cueing — but 50-70 m is a materially stronger and more accurate story than 10 m.

These are geometric limits only; they do not account for thermal contrast falloff, atmospheric attenuation, or the camera's internal temporal filtering. **Verify empirically**: point it out a window at something 50-100 m away with focus at the far setting and check sharpness.

## System architecture
Two-stage thermal-optical sensor fusion:

1. **Rooftop sensor unit** — thermal camera (Mini2 Plus V2) + optical camera (Pi Cam Module 2), rigidly mounted side by side, overlooking the apron
2. **Frame sync + anomaly detection** — thermal background subtraction / frame differencing flags warm moving blobs against the (cold) sky
3. **Shape + size classification** — the flagged region is mapped into the optical frame (via homography, see below) and cropped; contour/shape analysis (rotor silhouette vs. bird body/wing shape) plus size estimation discriminates drone vs. bird vs. clutter. **Built, and bidirectional**: optical also detects independently and can cue thermal, which is what covers thermal crossover and cold-soaked airframes. See `docs/cross_cueing.md`. **Caveat**: at the Pi camera's configured 640x480 the optical IFOV equals the thermal IFOV (1.70 mrad/px both), so optical adds no shape information at range. Capturing a high-resolution crop on cue would take shape classification from ~18 m to ~94 m and is the highest-value change left.
4. **Sensor fusion** — thermal confidence + optical confidence combined into a single confidence score and one unified track (directly answers the brief's "single coherent situational picture" requirement)
5. **Alert dashboard** — one alert per object: timestamp, confidence, classification, location. **Built** — Flask page, four live views, stateful alerts. See `docs/dashboard.md`.

Rationale for thermal-first: thermal contrast against sky is strong even for small objects, and works day/night. Optical adds shape discrimination, which is the documented weak point of thermal-only systems (bird vs. drone false positives are a known unsolved problem in the literature).

## Adaptive Site Intelligence (new concept)

**Core idea**: "Detect what it is. Learn where it belongs. Alert when it doesn't." AstraVigil should not rely solely on a fixed classifier answering "is this a drone, bird, or other object?" — it should also learn what *normal activity looks like at the specific protected site* over time, so it's a site-specific system rather than a generic object detector.

### Two complementary signals, kept separate
1. **Fixed object classification** — "what is this object?" A lightweight local classifier (candidate: **MobileNetV3-Small**, quantized to TFLite/INT8 for real-time inference on a Raspberry Pi 4) outputs a class: drone / bird / aircraft / vehicle / person / unknown. This classifier stays stable — it does not continuously retrain itself from every observation.
2. **Adaptive site baseline** — "is this normal for this location?" A separate, incrementally-built model of recurring activity at the site, keyed on features per tracked object: position, time of day, frequency, velocity, direction, trajectory, approximate size, optical appearance, thermal intensity/signature, duration of presence, classification confidence.

**Why keep them separate**: an object can be correctly classified but still be suspicious (a vehicle in a restricted area at 3am), or unclassifiable but clearly anomalous (unknown object, odd thermal signature, erratic trajectory). Object identity and behavioral/site anomaly are different questions and should produce two independent signals that get fused, not one signal overriding the other.

**Safety/design principle**: the system must not learn that every repeatedly-observed object is "friendly" — it learns a baseline of *expected behavior*, not a whitelist of objects. So `known object + abnormal behavior` and `unknown object + anomalous behavior` must both still be flaggable as suspicious. The learning layer adjusts the contextual/anomaly score; it never overrides the fixed object classifier.

### Decision pipeline
```
Thermal + Optical Sensors
        ↓
   Object Detection
        ↓
    Object Tracking
        ↓
  ┌─────┴──────┐
  ▼            ▼
Object       Site
Classifier   Baseline
"What is    "Is it
 it?"        normal
             here?"
  └─────┬──────┘
        ↓
  Sensor Fusion
        ↓
 Threat Assessment
        ↓
  ┌─────┴──────┐
Normal      Suspicious
(continue    → ALERT
 tracking)
```

### Thermal camera's expanded role
The Mini2 Plus V2 isn't just a night-vision input — its readings (intensity/contrast, apparent size, warm-object signature) become another feature stream feeding the site baseline, alongside optical appearance, movement, trajectory, location, and time. Thermal stays the primary channel for spotting small warm moving objects against cold sky; optical still does shape confirmation.

### Worked example
Hours/days of apron observation establish a baseline: vehicles in designated areas, aircraft on expected corridors, birds around the perimeter, people in known operational zones, no drones ever seen. Then a small object appears — thermal flags a warm mover, optical shows a compact rotorcraft-like shape, classifier says drone = 0.86, but behaviorally it enters the restricted apron area on a trajectory outside anything in the learned baseline → site baseline flags "highly anomalous" → fusion combines classification confidence + anomaly score → high overall threat → ALERT. The alert comes from combining *what it is + how it looks thermally + where it is + how it's moving + whether that's normal here*, not from classification alone.

### Hackathon-scoped implementation (no big self-training NN)

**Status: built.** `src/astravigil/site_intelligence/` learns a per-cell scene
reference and activity model, persists it to `data/baseline/site.npz`, and
scores each object on four channels (position, appearance, feature, dwell).
`src/astravigil/fusion/threat.py` combines that with the classifier via a
noisy-OR so neither signal can overrule the other, and
`src/astravigil/alerting/` emits one stateful alert per object. Full write-up
including the three failure modes that had to be closed: `docs/site_intelligence.md`.
1. Detect and track objects locally on the Pi.
2. Extract a feature vector per track (see feature list above).
3. Store observations associated with the site.
4. Build an incremental baseline of normal activity.
5. Compute an anomaly score for new observations against that baseline.
6. Combine anomaly score with optical classification + thermal confidence into one threat score.
7. Generate one unified alert.
8. Keep accumulated observations as the site's evolving knowledge base.

Lightweight, CPU-friendly candidates for the baseline/anomaly layer: incremental clustering, distance-based anomaly detection, or simple statistical baselines (mean/variance per feature, per time-of-day bucket) — not a deep model.

### Featherless AI as secondary confirmation (optional, not required per-detection)
Local classifier (MobileNetV3-Small) handles fast, continuous, offline classification. For ambiguous mid-confidence detections only, escalate to the **Miril-DroneVLM-2B-2** API via Featherless AI as a secondary confirmation layer:
```
Drone = 0.95 → immediate local classification, no API call
Drone = 0.54 → ambiguous → escalate to Miril-DroneVLM-2B-2 → secondary confirmation
```
This keeps the core system fully functional offline; cloud AI is a confidence booster, not a dependency.

### Positioning
- Original concept: thermal + optical sensor fusion for small-drone detection.
- Expanded concept: an adaptive, site-specific detect-classify-alert system that learns normal activity at a protected site and flags objects whose identity, thermal/visual characteristics, location, or behavior indicate an anomaly.
- Differentiator line: "AstraVigil doesn't only ask what an object is. It learns what belongs at the site and detects when something doesn't."
- Main name for the capability: **Adaptive Site Intelligence**. Supporting terms: site-specific baseline, normal-activity model, behavioral anomaly detection, adaptive threat assessment, context-aware sensor fusion, continuous site learning.
- Avoid claiming full autonomy or continuous neural-net retraining in the pitch — the accurate framing is *adaptive anomaly learning*: the site baseline evolves from observed data, the core object classifier stays controlled and stable.

## Frame matching (current focus)
Since the two cameras differ in resolution, FOV, and physical position, we need a pixel-to-pixel mapping.

**Method**: homography via OpenCV.
1. Mount both cameras rigidly, close together, same facing direction.
2. Capture a calibration pair — point both cameras at 4-8 identifiable reference points (heat-visible for thermal: use point heat sources like a hand/mug/lamp rather than a printed checkerboard, since paper is thermally invisible).
3. Manually identify pixel coordinates of the same physical points in both frames.
4. Compute homography: `cv2.findHomography(thermal_pts, optical_pts)` → save matrix `H`.
5. Apply per-frame: `cv2.perspectiveTransform()` maps any thermal-detected point into optical-frame coordinates, so the corresponding region can be cropped for classification.
6. Sanity check: warp the full thermal frame into optical coordinate space (`cv2.warpPerspective`) and overlay with transparency — hot regions should visually align with the corresponding real object.

Re-calibrate any time the camera mount is bumped.

## Hardware bring-up log (Raspberry Pi + thermal camera)
Working setup achieved after troubleshooting:

- `lsusb` and `v4l2-ctl --list-devices` confirmed thermal camera enumerates as UVC device: `Camera: UVC Camera (usb-0000:01:00.0-1.1)` → `/dev/video1`, `/dev/video2`, `/dev/media5`
- Native thermal format: **NV12, 256x192, 25fps** (also exposes several larger YUYV sizes — these include extra metadata rows, not used)
- **Blocker hit**: `ffmpeg`/OpenCV/`v4l2-ctl` all failed with `Input/output error` when opening `/dev/video1`. `dmesg` showed the root cause: `uvcvideo 1-1.1:1.1: Failed to query (130) UVC probe control : -32 (exp. 34)` — a known Linux `uvcvideo` driver incompatibility with non-fully-UVC-compliant thermal sensors.
- **Fix**: added a kernel module quirks flag.
  ```
  echo "options uvcvideo quirks=2" | sudo tee /etc/modprobe.d/uvcvideo-thermal.conf
  sudo reboot
  ```
  (`quirks=2` = `UVC_QUIRK_PROBE_MINMAX`, tells the driver to tolerate the non-standard probe negotiation.)
- After reboot: `ffmpeg -f v4l2 -input_format nv12 -video_size 256x192 -i /dev/video1 -frames:v 1 -update 1 output.jpg` succeeds and produces a valid frame.
- Captured test image is grayscale (expected — raw NV12 has no color palette; the HIKmicro app applies false-color palettes in software, not in the raw stream). For a colorized display, apply `cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)` — cosmetic only, not needed for the actual detection pipeline, which should work directly on raw grayscale intensity values.
- OpenCV's `pip`-installed `opencv-python` had trouble opening the device directly (`VIDEOIO(V4L2): backend is generally available but can't be used to capture by name`) even after the quirks fix — ffmpeg capture worked, OpenCV did not.

**Superseded — resolved via a raw USB driver.** The V4L2 path is no longer used. `src/astravigil/drivers/thermal/` drives the camera's streaming endpoint directly over `pyusb`/`libusb`, detaching `uvcvideo` entirely, which sidesteps the probe-negotiation bug without needing the `quirks=2` workaround at all. Ported from a working Raspberry Pi 5 bench rig (kept in `ThermalCam/` for reference) and hardened for the Pi 4 — see `docs/pi4_thermal_bringup.md`.

The decisive advantage is not just that it works: **the V4L2/NV12 path yields 8-bit grayscale, while the raw endpoint yields the full 16-bit per-pixel sensor counts.** Those counts are what the temperature calibration consumes, so this is what makes actual *temperature* — as opposed to brightness — available as a discriminating feature (motor heat vs. bird body heat). The old path discarded that before we ever saw it.

Frame format: 256 x 344 uint16 LE per frame — rows 0-191 thermal, row 192 metadata (cell 0 = ambient in hundredths °C), rest padding. Calibration: `linear = ambient + (raw - 4850)/31.0`, then a piecewise curve with gain 1.0157 / offset 3.37 that matches the HIKMICRO app's factory calibration.

## Known constraints / honest limitations (for the pitch)
- ~~Effective range ~10m due to thermal lens focus spec~~ — **corrected**: the spec is 0.1–50 m, and the real limit is angular resolution (~73 m detect, ~18 m classify for a Mini-class drone). See the specs section above
- Bird-vs-drone discrimination at ~18 m classification range is the genuinely hard problem — a pigeon and a DJI Mini are within a factor of ~1.2 in size
- The camera applies internal temporal filtering (74.7% of pixels bit-identical frame to frame), which may smear fast-crossing targets — untested
- No SDR — ruled out RF-based detection/fingerprinting entirely
- No defeat mechanism — challenge 04 doesn't require one (unlike challenge 01), so this is fully in scope
- ~~Alert dashboard not yet built~~ — built; see `docs/dashboard.md`
- The site baseline assumes a fixed camera. A bumped mount invalidates it exactly as it invalidates the homography, and there is no automatic detection of that yet
- Site intelligence is validated **in simulation only**. The scenarios were written by the same person who wrote the thresholds; it demonstrates the mechanism, not real-world performance
- No restricted-zone concept: the model learns where traffic goes, not which areas are off-limits
- The two cameras are not time-synchronised: 0-40 ms of skew, ~6 px of misregistration on a fast crosser, zero on a stationary one
- Optical detection is confined to the thermal footprint, which is only ~15% of the optical frame area; contacts outside it are reported but cannot be cross-checked on a fixed mount
- **No neural network anywhere in the codebase.** Classification is hand-weighted rules over four interpretable features. MobileNetV3-Small and Miril-DroneVLM remain aspirations, not implementations - do not claim them
- ~~OpenCV direct device access not fully solved~~ — resolved: raw USB driver in `src/astravigil/drivers/thermal/`, which also unlocked 16-bit temperature data the V4L2 path was discarding

## Reference / prior art (for the "why this is credible" slide)
- Commercial systems (HGH Infrared SPYNEL, FLIR/Teledyne, Pelco) use the same core approach (thermal + EO fusion) at much higher cost (~$50K-150K for lower-cost imaging-only systems, <1000m range)
- Bird-vs-drone discrimination via thermal alone is a documented open research problem (high false-positive rate in existing literature) — our optical shape-confirmation stage directly targets this gap
- Small drones occupy as few as 2x2 to 10x10 pixels at range in thermal — explains why range is fundamentally hard, not a hardware failure on our part

## Next steps

Done: Pi Camera capturing alongside thermal; homography calibration script and
ground-truth-scored sanity check; thermal anomaly detection; fusion into a
single track; adaptive site intelligence with persistent baseline; alert
dashboard.

Remaining, in order of what the demo actually needs:

1. **Capture real test footage** — drone at various distances, birds and
   clutter as negative examples. Everything below the hardware line is still
   simulation-validated only, and this is the critical path.
2. **Learn a real site baseline on the Pi** — `scripts/learn_site.py --source hardware --minutes 20`,
   then confirm a real object left in frame is caught after a restart. This is
   the headline claim and it has only been shown in simulation.
3. **Re-measure timings on the Pi 4** rather than trusting an 8x multiplier off
   the laptop numbers; the budget at 25 Hz is no longer comfortable.
4. Shape/size classification on the optical crop (the crop handoff exists,
   nothing consumes it yet).
5. Retune the classifier and the site thresholds against captured negatives.
6. Record a backup demo video in case the live demo fails on stage.
7. Prep slides: problem → architecture diagram → live/recorded demo →
   confusion matrix → honest limitations → what a production version would add.
