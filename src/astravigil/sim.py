"""Synthetic thermal + optical scene, for developing without the hardware.

The point of this module is not to look pretty. It is to make the frame
matching *checkable*: both views are rendered from one flat world by two
different homographies, so the true thermal->optical mapping is known exactly.
Calibration can therefore be scored against ground truth instead of eyeballed,
which is the only way to catch a homography that is subtly wrong - and a
subtly wrong homography is invisible until detections start landing on the
wrong optical pixels.

The world is a plane, which is also the assumption a homography makes, so the
mapping here is exact by construction. Real scenes are not planar, and that
difference is parallax - see hardware/README.md.
"""
import math

import cv2
import numpy as np

from .drivers.thermal.constants import (
    AMBIENT_SCALE,
    FRAME_H,
    FRAME_W,
    METADATA_ROW,
    TEMP_BASELINE,
    TEMP_SCALE,
    THERMAL_ROWS,
)

WORLD_W, WORLD_H = 1400, 1000
OPTICAL_W, OPTICAL_H = 640, 480

# Scene temperatures, degrees C. Cold sky against warm movers is the whole
# reason thermal leads this pipeline.
T_SKY = -5.0
T_HORIZON = 8.0
T_GROUND = 16.0
T_BUILDING = 14.0
T_AMBIENT = 16.0
# Measured off the real HIKMICRO Mini2 Plus V2, not assumed. Two captures of
# an actual quadcopter:
#
#   close range   max 45.4 C against a 20.5 C background
#   mid range     max 35.0 C against a 19.2 C background
#
# The important correction: the HOTTEST PART IS THE CENTRE. Battery, ESCs and
# the video transmitter all sit in the body, and they run hotter than the
# motors on a small airframe at hover. This file previously had it backwards -
# motors at 42 C around a 28 C body - which is the intuitive guess and is
# wrong. It matters because the classifier's strongest surviving feature is
# peak-above-own-mean, and a model that puts the peak in the wrong place
# produces a blob with the wrong statistics to tune against.
T_DRONE_CORE = 44.0     # battery / ESC / VTX stack, centre of the airframe
T_DRONE_BODY = 33.0     # shell around the core
T_DRONE_MOTOR = 30.0    # motor cans - warm, but cooler than the core
T_BIRD = 31.0

# Sensor noise, in raw counts. Measured NETD on the real camera was ~13 mK
# apparent (about 0.4 counts), but that is after its internal temporal
# filtering. Use something slightly pessimistic so detection thresholds tuned
# here are not optimistic on real data.
NOISE_COUNTS = 1.2


# Targets patrol inside the thermal field of view rather than roaming the whole
# world. Letting them wander out of frame means tracks die and re-form with new
# IDs every few seconds, which starves the temporal features - flap and
# straightness both need continuous history to mean anything - and makes the
# demo mostly empty sky.
PATROL_X = (520.0, 930.0)
PATROL_Y = (330.0, 620.0)


class Target:
    """A mover on the world plane, bouncing inside the patrol box."""

    def __init__(self, kind, x, y, vx, vy, size):
        self.kind = kind          # "drone" | "bird"
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.size = float(size)
        self.phase = 0.0
        # Per-target rather than global constants, because the intruder below
        # has to cool down after it lands - which is the point of it.
        self.body_c = T_DRONE_BODY if kind == "drone" else T_BIRD
        self.motor_c = T_DRONE_MOTOR
        self.core_c = T_DRONE_CORE
        self.visible = True
        self.crossover = False

    def step(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        # A bird's path wanders; a drone under command holds its line. That
        # difference is what `straightness` is meant to pick up, so it has to
        # exist in the simulation or the feature tests nothing.
        if self.kind == "bird":
            self.vy += math.sin(self.phase * 0.7) * 26.0 * dt
            self.vx += math.cos(self.phase * 0.4) * 14.0 * dt
        self.phase += dt * (9.0 if self.kind == "bird" else 60.0)

        if self.x < PATROL_X[0]:
            self.x, self.vx = PATROL_X[0], abs(self.vx)
        if self.x > PATROL_X[1]:
            self.x, self.vx = PATROL_X[1], -abs(self.vx)
        if self.y < PATROL_Y[0]:
            self.y, self.vy = PATROL_Y[0], abs(self.vy)
        if self.y > PATROL_Y[1]:
            self.y, self.vy = PATROL_Y[1], -abs(self.vy)


# Where the intruder puts down: on the ground, below the horizon, inside the
# thermal field of view. The whole point of the scenario is that this is a
# place nothing ever goes, so the site model has never seen anything there.
LANDING_XY = (700.0, 645.0)
APPROACH_XY = (540.0, 340.0)
MOTOR_COOL_TAU_S = 25.0
BODY_COOL_TAU_S = 70.0


class Intruder(Target):
    """A quadcopter that flies in, lands, and then just sits there.

    This is the challenge brief's own scenario, and it is the case a motion
    detector cannot hold: for a few seconds it is an obvious warm mover, and
    after that it is a rock. Its motors then cool towards its airframe, so
    even the hotspot cue - the strongest thing the classifier has - fades out
    while the object is still sitting on the apron.

    What is left after a minute is a small warm patch that has not moved and
    was not there yesterday. Only the site baseline can still see it.
    """

    def __init__(self, enter_at=6.0, travel_s=7.0, landed=False,
                 crossover=False):
        # Thermal crossover: the airframe sits at exactly the temperature of
        # whatever is behind it, so it has no thermal contrast at all. This
        # happens for real twice a day as the background sweeps through
        # ambient, and it is also what a drone parked long enough to
        # cold-soak looks like. The thermal camera is genuinely blind to it -
        # not degraded, blind - and only the optical camera can see it.
        super().__init__("drone", APPROACH_XY[0], APPROACH_XY[1],
                         0.0, 0.0, 22.0)
        # After super(), which resets it to the Target default.
        self.crossover = crossover
        self.enter_at = 0.0 if landed else enter_at
        self.travel_s = 1e-6 if landed else travel_s
        self.landed_at = None
        self.t = 0.0
        self.visible = landed
        if landed:
            self.x, self.y = LANDING_XY
            # Already been there for hours: fully cold-soaked, which is the
            # hardest version of the problem.
            self.body_c = T_GROUND + 4.5
            self.motor_c = self.core_c = self.body_c

    def step(self, dt):
        self.t += dt
        self.phase += dt * 60.0
        if self.t < self.enter_at:
            self.visible = False
            return
        self.visible = True

        p = min(1.0, (self.t - self.enter_at) / self.travel_s)
        if p < 1.0:
            # Ease-out: it arrives fast and settles slowly, which is what a
            # descent under command looks like and keeps the tracker fed.
            e = 1.0 - (1.0 - p) ** 2
            self.x = APPROACH_XY[0] + (LANDING_XY[0] - APPROACH_XY[0]) * e
            self.y = APPROACH_XY[1] + (LANDING_XY[1] - APPROACH_XY[1]) * e
            return

        if self.landed_at is None:
            self.landed_at = self.t
        self.x, self.y = LANDING_XY
        dwell = self.t - self.landed_at
        floor = T_GROUND + 4.5
        # The core carries most of the stored heat, so it is what fades most
        # visibly once the airframe is on the ground and unpowered.
        self.core_c = (floor + (T_DRONE_CORE - floor)
                       * math.exp(-dwell / MOTOR_COOL_TAU_S))
        self.motor_c = (floor + (T_DRONE_MOTOR - floor)
                        * math.exp(-dwell / MOTOR_COOL_TAU_S))
        self.body_c = (floor + (T_DRONE_BODY - floor)
                       * math.exp(-dwell / BODY_COOL_TAU_S))


class SyntheticScene:
    """One flat world, two virtual cameras, one known homography between them.

    Optical sees a much wider field than thermal (62 deg vs 25 deg on the real
    hardware), so the thermal view sits inside the optical view - the same
    relationship the printed mount produces.
    """

    SCENARIOS = ("clean", "patrol", "intrusion", "resident", "crossover")

    def __init__(self, seed=0, baseline_px=26.0, scenario="patrol"):
        if scenario not in self.SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}, expected one "
                             f"of {list(self.SCENARIOS)}")
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self._build_world()
        # Somewhere for thermally-invisible targets to be drawn and discarded.
        self._scratch = np.zeros_like(self.world_temp)

        # Optical camera: wide view of the world centre.
        opt_src = np.float32([[120, 90], [1280, 70], [1310, 900], [90, 920]])
        # Thermal camera: narrower, and shifted sideways to stand in for the
        # 50 mm lens baseline on the real mount.
        b = baseline_px
        t_src = np.float32([[470 + b, 300], [930 + b, 292],
                            [940 + b, 660], [462 + b, 668]])

        self.H_world_to_optical = cv2.getPerspectiveTransform(
            opt_src, np.float32([[0, 0], [OPTICAL_W, 0],
                                 [OPTICAL_W, OPTICAL_H], [0, OPTICAL_H]]))
        self.H_world_to_thermal = cv2.getPerspectiveTransform(
            t_src, np.float32([[0, 0], [FRAME_W, 0],
                               [FRAME_W, THERMAL_ROWS], [0, THERMAL_ROWS]]))

        # Ground truth: thermal pixel -> optical pixel.
        self.H_truth = (self.H_world_to_optical
                        @ np.linalg.inv(self.H_world_to_thermal))
        self.H_truth /= self.H_truth[2, 2]

        # A bird patrols in every scenario. It is not decoration - it is the
        # negative example. The site model has to learn that birds crossing
        # the perimeter are normal here, or the demo is just an alarm that
        # fires at everything.
        # "clean" is that and nothing else: the scene as it is supposed to
        # be, which is what a site baseline has to be learned from. Learning
        # from a scene that already contains the intruder teaches the system
        # that the intruder belongs.
        self.targets = [Target("bird", 890, 500, -44.0, -18.0, 17.0)]
        if scenario == "patrol":
            self.targets.insert(0, Target("drone", 560, 380, 58.0, 11.0, 22.0))
        elif scenario == "intrusion":
            self.targets.append(Intruder())
        elif scenario == "resident":
            # Already on the ground before the first frame, so the motion
            # detector's background absorbs it instantly and never reports
            # it. Needs a site model learned earlier to be seen at all.
            self.targets.append(Intruder(landed=True))
        elif scenario == "crossover":
            # Same approach and landing, but at ambient temperature the whole
            # way. Nothing thermal to detect, nothing thermal to learn a
            # baseline deviation from. If the optical camera cannot raise
            # this on its own, the system does not see it at all.
            self.targets.append(Intruder(crossover=True))
        self.t = 0.0

    # ------------------------------------------------------------ world
    def _build_world(self):
        """Static background: sky above, ground and a building below."""
        temp = np.full((WORLD_H, WORLD_W), T_GROUND, np.float32)
        bgr = np.zeros((WORLD_H, WORLD_W, 3), np.uint8)

        horizon = 620
        for y in range(horizon):
            f = y / horizon
            temp[y, :] = T_SKY + (T_HORIZON - T_SKY) * f
            bgr[y, :] = (int(200 - 40 * f), int(150 + 40 * f),
                         int(105 + 60 * f))
        bgr[horizon:, :] = (95, 100, 105)

        # Apron markings, so the optical view has structure to match against.
        for x in range(60, WORLD_W, 190):
            cv2.line(bgr, (x, horizon + 40), (x - 40, WORLD_H),
                     (170, 175, 180), 6)
        cv2.line(bgr, (0, horizon), (WORLD_W, horizon), (140, 145, 150), 4)

        # A building on the skyline - warm, and a classic false positive.
        cv2.rectangle(bgr, (980, 430), (1210, horizon), (88, 92, 100), -1)
        cv2.rectangle(temp, (980, 430), (1210, horizon), T_BUILDING, -1)
        for wy in range(455, horizon - 30, 45):
            for wx in range(1000, 1190, 40):
                cv2.rectangle(bgr, (wx, wy), (wx + 22, wy + 26),
                              (130, 140, 150), -1)
                cv2.rectangle(temp, (wx, wy), (wx + 22, wy + 26),
                              T_BUILDING + 3.0, -1)

        self.world_temp = temp
        self.world_bgr = bgr

    def _draw_targets(self, temp, bgr):
        for tg in self.targets:
            if not tg.visible:
                continue
            x, y, s = int(tg.x), int(tg.y), tg.size
            # A crossover target is written to a throwaway array instead of
            # the real thermal frame, so it leaves no signature of any kind.
            # Matching the background by sampling one pixel does not work:
            # the sky has a vertical gradient across the target's own
            # footprint, and the residual edge contrast is enough for the
            # detector to find. Blind has to mean blind.
            T = self._scratch if tg.crossover else temp
            if tg.kind == "drone":
                # Body plus four rotor hubs. The hubs are the hottest thing in
                # the scene, which is what a real quad looks like in LWIR.
                cv2.rectangle(bgr, (x - int(s * .35), y - int(s * .25)),
                              (x + int(s * .35), y + int(s * .25)),
                              (45, 45, 50), -1)
                # Shell first, then the hot core inside it. Drawn in this
                # order so the core wins - it is the peak the classifier's
                # hotspot feature is measuring.
                cv2.rectangle(T, (x - int(s * .35), y - int(s * .25)),
                              (x + int(s * .35), y + int(s * .25)),
                              tg.body_c, -1)
                cv2.rectangle(T, (x - int(s * .17), y - int(s * .13)),
                              (x + int(s * .17), y + int(s * .13)),
                              tg.core_c, -1)
                for dx in (-1, 1):
                    for dy in (-1, 1):
                        cx = x + int(dx * s * 0.62)
                        cy = y + int(dy * s * 0.42)
                        cv2.circle(bgr, (cx, cy), max(2, int(s * .16)),
                                   (35, 35, 40), -1)
                        cv2.circle(T, (cx, cy), max(2, int(s * .16)),
                                   tg.motor_c, -1)
                        # Rotor disc: faint, wide, and cooler than the hub.
                        cv2.circle(bgr, (cx, cy), max(3, int(s * .40)),
                                   (70, 70, 75), 1)
            else:
                # Bird: body plus flapping wings. The wingbeat is the cue that
                # actually separates it from a quad, so it has to be here.
                flap = np.sin(tg.phase)
                cv2.ellipse(bgr, (x, y), (int(s * .34), int(s * .20)), 0, 0,
                            360, (40, 38, 36), -1)
                cv2.ellipse(T, (x, y), (int(s * .34), int(s * .20)), 0, 0,
                            360, T_BIRD, -1)
                span = int(s * (0.5 + 0.75 * abs(flap)))
                lift = int(-flap * s * 0.55)
                for sgn in (-1, 1):
                    pts = np.array([[x, y],
                                    [x + sgn * span, y + lift],
                                    [x + sgn * int(span * .5),
                                     y + int(s * .16)]], np.int32)
                    cv2.fillPoly(bgr, [pts], (40, 38, 36))
                    cv2.fillPoly(T, [pts], T_BIRD - 1.5)

    # ------------------------------------------------------------ render
    def step(self, dt=1.0 / 25.0):
        for tg in self.targets:
            tg.step(dt)
        self.t += dt

    @property
    def truth_targets(self):
        """Targets currently rendered into the world, for scoring."""
        return [tg for tg in self.targets if tg.visible]

    def _rendered_world(self):
        temp = self.world_temp.copy()
        bgr = self.world_bgr.copy()
        self._draw_targets(temp, bgr)
        return temp, bgr

    def frames(self):
        """One synchronised (thermal_raw16, optical_bgr) pair."""
        temp, bgr = self._rendered_world()

        t_view = cv2.warpPerspective(temp, self.H_world_to_thermal,
                                     (FRAME_W, THERMAL_ROWS),
                                     flags=cv2.INTER_LINEAR)
        o_view = cv2.warpPerspective(bgr, self.H_world_to_optical,
                                     (OPTICAL_W, OPTICAL_H),
                                     flags=cv2.INTER_LINEAR)

        # Degrees C back to raw counts, the inverse of the driver's linear
        # step, so the rest of the pipeline consumes ordinary raw frames and
        # never learns it is being simulated.
        raw = TEMP_BASELINE + (t_view - T_AMBIENT) * TEMP_SCALE
        raw += self.rng.normal(0.0, NOISE_COUNTS, raw.shape)

        frame = np.zeros((FRAME_H, FRAME_W), np.uint16)
        frame[:THERMAL_ROWS, :] = np.clip(raw, 0, 65535).astype(np.uint16)
        frame[METADATA_ROW, 0] = int(T_AMBIENT * AMBIENT_SCALE)
        frame[METADATA_ROW, 8] = THERMAL_ROWS
        frame[METADATA_ROW, 9] = FRAME_W

        noise = self.rng.normal(0, 2.5, o_view.shape)
        o_view = np.clip(o_view.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return frame, o_view

    # ------------------------------------------------------- calibration
    def calibration_points(self, n=8, jitter_px=0.0):
        """Matched thermal/optical point pairs, as a calibration would give.

        jitter_px simulates imprecise clicking, which is the dominant error in
        a hand-picked calibration and worth being able to reproduce.
        """
        pts = np.float32([
            [40, 30], [216, 28], [222, 160], [36, 166],
            [128, 44], [70, 120], [190, 118], [128, 150],
        ])[:n]
        thermal = pts.reshape(-1, 1, 2)
        optical = cv2.perspectiveTransform(thermal, self.H_truth)
        if jitter_px > 0:
            optical = optical + self.rng.normal(
                0, jitter_px, optical.shape).astype(np.float32)
        return thermal.reshape(-1, 2), optical.reshape(-1, 2)
