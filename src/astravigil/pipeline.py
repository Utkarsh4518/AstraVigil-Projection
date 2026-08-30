"""One frame in, one situational picture out.

Wires the stages together: capture -> calibrate -> detect -> track -> classify
-> learn the site -> assess -> alert -> map into the optical frame. Everything
the dashboard draws comes from the Result this returns, so the UI holds no
pipeline logic and the pipeline can be run headless for testing.

Three detection paths run side by side, and neither camera is merely a
confirmation service for the other:

  THERMAL MOTION   background subtraction over ~2 s finds warm movers. Fast,
                   and blind to anything that stops or was already there.
  SITE BASELINE    a persistent model over ~90 s finds patches of scene that
                   are not where the site says they should be. Slow, and the
                   only thing that sees a drone that has landed and gone quiet.
  OPTICAL MOTION   independent detection in the visible band. Covers thermal's
                   two blind spots - crossover at dawn and dusk, and a
                   cold-soaked airframe with no signature left to find.

Each camera can put a question to the other through the homography. Thermal
asks "what shape is this?", because it finds small things well and cannot tell
a bird from a quadcopter. Optical asks "is this warm?", because it reads shape
well and cannot tell an aircraft from a cloud edge.

Everything is associated BEFORE assessment, not after. One aircraft seen by
both sensors and by both thermal paths is one object and one alert - anything
else rebuilds the wall of separate alarms the brief exists to complain about.
"""
import collections
import time

import numpy as np


from .alerting import AlertManager
from .calibration import homography
from .classification.rules import classify, not_airborne
from .detection.thermal import ThermalDetector
from .drivers.thermal.calibration import calibrate_frame
from .detection.objects import (
    AIRCRAFT_NAMES, AUTOFETCH, CAN_OVERRULE, DRONE_CONF, DRONE_PATH,
    ObjectRecogniser, best_overlap, find_second_model)
from .detection.optical import OpticalDetector
from .fusion import (OpticalContactLog, assess_optical_only, assess_static,
                     assess_track, associate, verify_optical, verify_thermal)
from .fusion.cues import CueBoard
from .llm import Escalator
from .site_intelligence import DwellMonitor, SiteBaseline
from .site_intelligence.optical_baseline import OpticalBaseline
from .site_intelligence.baseline import MIN_LEARNED_FRAMES
from .tracking.tracker import Tracker
from .utils.env import env_int


# How many cross-cue exchanges the trail remembers. Enough to read the recent
# conversation at a glance, few enough that it stays a glance.
CROSS_LOG_MAX = 14

# Seconds an unidentifiable, cold, stationary optical contact must persist
# before it is worth an API call.
#
# The escalation trigger for a thermal track is threat, which is right: a
# warm mover scoring high is a hard case and a bigger model may name it. It
# is the WRONG trigger for the one object this system exists to catch. A
# drone that has landed and gone cold has no heat, no motion and no
# classifier output, so nothing local can push its threat up quickly - and by
# the time dwell alone has, minutes have passed. Cold, still, and unnameable
# is itself the signature, so it gets its own trigger.
ESCALATE_COLD_DWELL_S = 20.0

# Most optical contacts one frame may show at once.
#
# A backstop, not a threshold. The detector's own noise floor now rises with
# the light, which is the real fix for a dark room producing contacts out of
# sensor grain - but any threshold has a bad night, and the failure mode when
# it does is not one wrong box, it is forty, each numbered, over a picture of
# an empty room. Past a certain count the answer is not "here are your
# contacts" but "this camera cannot currently tell", and a console should say
# the second rather than draw the first.
#
# Kept as the largest few by area, which is the order the detector returns:
# if something real is in a noisy frame it is not the smallest thing in it.
MAX_OPTICAL_SHOWN = 8

# How long something must have been sitting there before it is worth an
# operator's attention, and how many to list.
#
# Six seconds is the same threshold the site models use to call a patch
# settled rather than passing, so this list contains exactly what those
# models are prepared to stand behind.
WATCH_MIN_DWELL_S = 6.0
WATCH_MAX = 12

# Confidence below which the local classifier counts as unsure, and the
# threat above which being unsure is worth an API call.
#
# Threat alone is the wrong trigger for this. It fires when the local
# evidence already agrees on something alarming, which is the case that needs
# a second opinion LEAST - the answer is already known and acted on. The case
# that needs one is the opposite: something is clearly there, clearly not
# nothing, and nothing on this machine can name it. That is what a bigger
# model is for.
UNSURE_CONF = 0.65
ESCALATE_DOUBT_AT = 0.35

# How confident the optical recogniser must be before its name overrules the
# thermal classifier's.
OPTICAL_NAME_TRUST = 0.50


class Result:
    __slots__ = ("thermal_raw", "thermal_c", "optical", "detections",
                 "tracks", "mask", "proc_ms", "capture_ms", "frame_index",
                 "stage_ms",
                 "healthy", "assessments", "alerts", "alert_history",
                 "recognitions", "recogniser_stats", "watching",
                 "static_anomalies",
                 "site_stats", "optical_site_stats", "cross_log",
                 "cue_numbers", "optical_cues", "optical_regions", "novelty",
                 "optical_detections",
                 "optical_evidence", "cross", "optical_roi", "learning",
                 "identifications", "escalation")

    def __init__(self):
        self.thermal_raw = None
        self.thermal_c = None
        self.optical = None
        self.detections = []
        self.tracks = {}
        self.mask = None
        # Time to process one frame, NOT the frame rate - the capture loop
        # paces itself to the camera, so low numbers mean headroom rather than
        # throughput.
        #
        # Capture is timed separately because it is the one stage that lies
        # about the target platform: in simulation it renders an entire world
        # (~19 ms) while on real hardware it is a USB read. Folding the two
        # together made the pipeline look ~40x more expensive than it is.
        # proc_ms is the part that carries over to the Pi.
        self.proc_ms = 0.0
        self.capture_ms = 0.0
        # Where the frame time actually went, by stage.
        #
        # proc_ms is one number and a frame rate complaint needs more than
        # one: "detection is slow" and "the site models are slow" and "the
        # optical detector is slow" all look identical from outside, and
        # guessing which it is from a screenshot is how an afternoon
        # disappears. Measured, not inferred.
        self.stage_ms = {}
        self.frame_index = 0
        self.healthy = True

        self.assessments = []       # one verdict per object, both paths
        self.alerts = []            # currently open alerts
        # Alerts that have closed, newest first. An operator watching an alert
        # vanish off the board has no way to tell "the object left" from "the
        # console dropped it", and the difference is the whole credibility of
        # the thing. The manager has kept these all along; nothing showed them.
        self.alert_history = []
        self.static_anomalies = []
        self.site_stats = {}
        self.novelty = None         # cell z-map upsampled to frame size

        # Things that have arrived and stayed, with what is known about
        # each. An operator's working list: everything here is either
        # something that was put down, or something that should not have
        # been, and the two are told apart by looking rather than by rule.
        self.watching = []
        self.recognitions = []         # named objects in the optical frame
        self.recogniser_stats = {}
        self.optical_detections = []   # what the optical camera found alone
        self.optical_evidence = {}     # track_id -> optical's shape answer
        self.cross = {}                # cue counts, for the dashboard
        # The small number every pane draws on this object, so the same thing
        # can be found on five views, in the trail and in the table.
        self.cue_numbers = {}          # key -> small int
        self.optical_cues = {}         # optical-only box tuple -> small int
        self.optical_regions = []      # optical site model's settled patches
        self.optical_roi = None        # where optical is allowed to look
        self.learning = {}             # operator-driven learning run
        self.identifications = {}      # key -> remote model's answer
        self.escalation = {}           # escalator health


class Pipeline:
    def __init__(self, source, H=None, threshold_c=1.5, site=None,
                 alerts=None, fps=25.0, clock="wall", learn=True,
                 cross_cue=True, optical_every_n=None, escalator=None,
                 escalate_at=0.75, policy=None, auto_calibrate=True,
                 calibration_path=None, min_area=None, recogniser=None):
        self.source = source
        self.H = H
        # Authored site policy - what is ALLOWED here, as opposed to what the
        # baseline has learned is normal here. Optional: with no policy the
        # system behaves exactly as before, judging on statistics alone.
        self.policy = policy
        self.detector = ThermalDetector(
            threshold_c=threshold_c,
            **({"min_area": min_area} if min_area else {}))
        # Optical frames carry six times the pixels of thermal ones and far
        # more clutter, so this runs at a fraction of the rate. Thermal keeps
        # every frame: it is the primary sensor and it is the cheap one.
        self.cross_cue = cross_cue
        # Frames between optical passes. Everything optical - the motion
        # detector, the site baseline, the calibrator's candidate frames -
        # runs on this tick, so it is the one knob that trades optical
        # freshness for frame rate without changing what anything decides.
        # Raise it on a machine that cannot keep up; the thermal path, which
        # is the primary sensor, is untouched by it.
        if optical_every_n is None:
            optical_every_n = env_int("ASTRAVIGIL_OPTICAL_EVERY_N", 3)
        self.optical = OpticalDetector(every_n=optical_every_n)
        # Names for what the optical camera is looking at. Optional and
        # self-disabling: with no weights file this reports nothing and every
        # path below falls back to the geometry it used before.
        self.recogniser = (recogniser if recogniser is not None
                           else ObjectRecogniser())
        # An optional second model, taking alternate passes with the first.
        # One knows furniture and has never seen a quadcopter; the other
        # knows quadcopters and nothing else. Alternating costs one inference
        # per pass either way, and the twelve-second hold keeps both sets of
        # names on screen between their turns.
        self.recogniser2 = None
        second = find_second_model(self.recogniser.path) \
            if recogniser is None else None
        if second:
            self.recogniser2 = ObjectRecogniser(path=second, conf=DRONE_CONF)
        elif recogniser is None and AUTOFETCH:
            # Nothing in the second slot. Go and get the one model that can
            # say "drone", rather than leaving it as a command to remember.
            self.recogniser2 = ObjectRecogniser(path=DRONE_PATH,
                                                conf=DRONE_CONF,
                                                autofetch=False)
            self.recogniser2.fetch_drone_model()
            # Half a cycle out of step, so the two never land on one frame.
            self.recogniser2._tick = self.recogniser2.every_n // 2
        self.optical_contacts = OpticalContactLog()
        # Remote identification, off unless a key is configured and it is
        # explicitly switched on. Nothing downstream depends on it.
        self.escalator = escalator if escalator is not None else Escalator()
        self.escalate_at = escalate_at
        self._roi_set = False
        # Frame matching from the scene itself, for the case where nobody has
        # run scripts/calibrate_homography.py yet. Only ever created when
        # there is no homography to start with: a manual fit made at watch
        # range beats one learned from whatever happened to walk past, and
        # replacing it would be helpfulness nobody asked for.
        self.autocal = None
        self.calibration_path = calibration_path
        if auto_calibrate and cross_cue and H is None:
            from .calibration.auto import AutoCalibrator
            self.autocal = AutoCalibrator()
        self.tracker = Tracker()
        self.fps = fps
        # "wall" is right when the loop is paced to real time, which is every
        # live run. "frames" makes dwell deterministic and independent of how
        # fast the machine chews through a headless evaluation.
        self.clock = clock
        self.learn = learn
        self.site = site if site is not None else SiteBaseline(fps=fps)
        # The other camera's model of the same place. Built lazily on the
        # first optical frame, because its grid has to match that frame's
        # size and nothing here knows it until one arrives.
        self.optical_site = None
        self._optical_tick = 0
        # The last few questions one camera asked the other, newest first.
        #
        # The cross-cue counters say two exchanges happened. They do not say
        # what was asked, what came back, or whether it changed the verdict -
        # which is the entire argument for having two sensors, and until now
        # the one part of it nobody could see.
        self.cross_log = collections.deque(maxlen=CROSS_LOG_MAX)
        # One number per object under discussion, shared by every pane.
        self.cues = CueBoard()
        self.dwell = DwellMonitor()
        self.alerts = alerts if alerts is not None else AlertManager()
        self.frame_index = 0
        self._times = []
        # Tracks the previous frame judged to be under alert. They are held
        # out of the activity model so an intruder cannot make its own
        # behaviour the site's definition of normal. One frame of lag is
        # irrelevant at 25 Hz.
        self._alerting = set()
        # An operator-driven learning run: explicit start, visible progress,
        # and a definite end. The model also learns continuously in normal
        # operation, but "point it at the site and press the button" is how a
        # person actually commissions one of these, and a progress bar that
        # reflects real coverage is what makes it trustworthy.
        self._session = None
        self.result = Result()

    @property
    def calibrated(self):
        return self.H is not None

    def set_homography(self, H):
        self.H = H

    # -------------------------------------------------------- learning run
    def start_learning(self, seconds=90.0, reset=True):
        self.learn = True
        if reset:
            self.site.reset()
            self.detector.reset()
            self.optical.reset()
            # The optical baseline was left out of this, so "learn this site"
            # rebuilt half the site. The button says the site, both cameras
            # watch it, and an operator standing in front of a console
            # watching one pane fill has no way to guess the other one is not
            # being rebuilt at all.
            if self.optical_site is not None:
                self.optical_site.reset()
        self._session = {"target_s": float(seconds), "started": self.now(),
                         "done": False}
        return self.learning_status()

    def stop_learning(self, save_path=None):
        saved = None
        if save_path:
            try:
                saved = self.save_site(save_path)
            except OSError as exc:
                print(f"could not save site model: {exc}")
        self._session = None
        return {"stopped": True, "saved": saved,
                "frames": self.site.frames}

    @property
    def settled_ready(self):
        """Has this site been learned well enough to call anything new?

        Both models, and not during a learning run. The thermal one carries
        the dwell and the temperature history; the optical one is the only
        thing that can see an object with no heat signature. An arrival
        judged against one of the two is half an answer, and the half that is
        missing is the half this system exists for.
        """
        if self._session is not None:
            return False
        if self.site.learning:
            return False
        return self.optical_site is None or not self.optical_site.learning

    def learning_status(self):
        s = self.site.stats()
        cov = float(np.clip(self.site.ref_n / MIN_LEARNED_FRAMES, 0, 1).mean())
        # The optical half, reported separately. They fill at different rates
        # - the optical model runs at a fraction of the frame rate and has
        # six times the pixels - and averaging them into one number would
        # hide whichever is behind.
        ocov = (float(self.optical_site.coverage().mean())
                if self.optical_site is not None else 0.0)
        out = {"active": self._session is not None,
               "settled_ready": bool(self.settled_ready),
               "coverage": round(cov, 3),
               "optical_coverage": round(ocov, 3),
               "scene_maturity": s["scene_maturity"],
               "activity_maturity": s["activity_maturity"],
               "frames": s["frames"]}
        if self._session is not None:
            elapsed = self.now() - self._session["started"]
            target = self._session["target_s"]
            out["elapsed_s"] = round(elapsed, 1)
            out["target_s"] = target
            out["remaining_s"] = max(0.0, round(target - elapsed, 1))
            # Progress is the SLOWER of clock and coverage. Time alone would
            # let the bar reach 100% while a third of the frame had never been
            # modelled - which is precisely the situation an operator needs to
            # be told about rather than reassured through.
            # The slowest of the three, so the bar cannot reach 100% while
            # either camera still has unmodelled frame left.
            out["progress"] = round(
                min(elapsed / max(target, 1e-6), cov, ocov or cov), 3)
            out["done"] = elapsed >= target
        return out

    def _note_cross(self, direction, key, label, asked, answered, threat):
        self.cross_log.appendleft({
            "dir": direction, "key": key, "label": label,
            # The same number that is drawn on the object itself. Without it
            # the trail is a list of things a reader has to go and find.
            "cue": self.cues.number(key, self.now()),
            "asked": asked, "answered": answered,
            "threat": round(float(threat), 2),
            "frame": self.frame_index,
        })

    @staticmethod
    def _optical_answer(ev, named=None):
        """What optical said back, in words rather than fields.

        A name goes first when there is one. "solidity 0.71, aspect 1.30" is
        the honest measurement and it is not an answer to "what is it" - a
        quadcopter and the back of a chair produce nearly the same numbers,
        which is the whole reason the question is being asked.
        """
        head = (f"it is a {named.label} ({named.confidence:.2f})"
                if named is not None else None)
        if ev is None:
            return head or "not asked"
        if not ev.found:
            return head or ev.reason or "nothing there"
        shape = (f"solidity {ev.solidity:.2f}, aspect {ev.aspect:.2f}, "
                 f"{int(ev.pixels)} px")
        if head:
            return f"{head} - {shape}"
        if ev.label and ev.label != "unknown":
            return f"looks like {ev.label} ({ev.confidence:.2f}) - {shape}"
        return f"shape unclear - {shape}"

    @staticmethod
    def _thermal_answer(ev):
        """What thermal said back. It cannot name a thing, only measure it.

        Worth spelling out in the trail rather than hiding: an operator
        reading "nothing warm here" next to an optical contact that has not
        moved in a minute is looking at the exact signature of a cold-soaked
        airframe, and the interface should let them see that reasoning.
        """
        if ev is None:
            return "not asked"
        if not ev.found:
            return ev.reason or "outside the thermal field of view"
        if ev.warm:
            return (f"warm - {ev.peak_c:.1f} C peak, {ev.contrast_c:+.1f} C "
                    f"above surroundings")
        return (f"NOT warm - {ev.peak_c:.1f} C peak, only "
                f"{ev.contrast_c:+.1f} C above surroundings")

    def _watch_list(self, assessments, now):
        """Everything that has arrived and not left, newest concern first.

        Only the settled channels. A warm mover is a track and belongs in the
        tracks table; this is the other kind of object entirely - the one
        that is not moving, cannot be classified from motion, and is
        therefore either furniture or the thing this system was built for.
        Both look identical to every sensor on the rig, which is exactly why
        it has to be a list a person can look down.
        """
        out = []
        for a in assessments:
            if a.kind not in ("static", "optical"):
                continue
            if a.dwell_s < WATCH_MIN_DWELL_S:
                continue
            out.append({
                "key": a.key,
                "cue": self.cues.number(a.key, now),
                "label": a.label,
                "kind": a.kind,
                "dwell_s": round(a.dwell_s, 1),
                "threat": round(a.threat, 3),
                "level": a.level,
                "sensors": a.sensors,
                "thermal": self._thermal_answer(
                    getattr(a, "thermal_check", None)),
                "reasons": list(a.reasons)[:3],
            })
        out.sort(key=lambda w: (-w["threat"], -w["dwell_s"]))
        return out[:WATCH_MAX]

    def _save_calibration(self, H):
        """Write the learned homography out, so the next run starts with it.

        Failing to save is not failing to calibrate: the fit is already in
        self.H and the run continues with it either way. Say so and carry on
        rather than losing a calibration that took minutes to earn over a
        directory that is not writable.
        """
        if not self.calibration_path:
            return
        st = self.autocal.status()
        try:
            homography.save(self.calibration_path, H, {
                "source": "auto",
                "n_points": st["pairs"],
                "reprojection_mean_px": st["error_px"],
            })
            print(f"auto-calibration: saved {self.calibration_path} "
                  f"({st['pairs']} pairs, {st['error_px']} px)")
        except OSError as exc:
            print(f"auto-calibration: fitted but could not save "
                  f"({exc}) - this run is calibrated, the next is not")

    def now(self):
        if self.clock == "frames":
            return self.frame_index / self.fps
        return time.time()

    def step(self):
        t_cap = time.monotonic()
        raw, optical = self.source.frames()
        capture_s = time.monotonic() - t_cap

        t0 = time.monotonic()
        res = Result()
        res.thermal_raw = raw
        res.optical = optical
        res.thermal_c = calibrate_frame(raw)

        t_stage = time.monotonic()
        detections = self.detector.update(res.thermal_c)
        tracks = self.tracker.update(detections)
        stage = {"detect": time.monotonic() - t_stage}
        t_stage = time.monotonic()

        # Naming runs on the optical frame at its own reduced rate, and runs
        # HERE - before classification, not after - because what the optical
        # camera can see is evidence about what the thermal blob is, and
        # evidence that arrives after the verdict is not evidence.
        recognitions = []
        if optical is not None:
            recognitions = list(self.recogniser.update(optical))
            if self.recogniser2 is not None:
                recognitions += self.recogniser2.update(optical)

        stage["recognise"] = time.monotonic() - t_stage
        t_stage = time.monotonic()

        for det in detections:
            tr = tracks.get(det.track_id)
            det.label, det.confidence = classify(det, tr)
            det.not_airborne = not_airborne(det, tr) if tr is not None else None

            # The other camera gets a say in what this is. A thermal blob is
            # a temperature and a silhouette; "laptop" is a fact about the
            # object, and it beats every inference drawn from four pixels.
            #
            # Only downgrades, and only from names that could not possibly
            # be a drone.
            #
            # This was too permissive and it did real damage. COCO has no
            # drone class, so a COCO model shown a quadcopter answers with
            # whichever of its eighty everyday objects it finds least unlike
            # - measured on the rig, "bowl 0.56". The old rule accepted any
            # name that was not explicitly airborne, so that guess overwrote
            # a correct thermal verdict of "drone" with "bowl", and the one
            # object this system exists to find was renamed into crockery by
            # the camera that was supposed to be helping.
            #
            # A name may now overrule the thermal classifier only if it is
            # something large and static that a small airframe cannot be
            # confused with - a sofa, a car, a person. Everything else is
            # recorded as optical's opinion and changes nothing, because a
            # model asked a question outside its own class list has no way to
            # say so, and a confident wrong answer is worse than no answer.
            if recognitions and self.H is not None:
                named = best_overlap(recognitions,
                                     homography.map_box(self.H, det.box))
                # An aircraft name is held to the drone model's own bar,
                # not to the one that guards against picking the wrong COCO
                # class. There is no wrong class in a one-class model.
                trust = (DRONE_CONF if named is not None
                         and named.label in AIRCRAFT_NAMES
                         else OPTICAL_NAME_TRUST)
                if named is not None and named.confidence >= trust:
                    if named.label in AIRCRAFT_NAMES:
                        # A model trained on drones saying "drone" is the one
                        # optical answer that is evidence FOR an aircraft
                        # rather than against one. No COCO model can produce
                        # this word, so seeing it means somebody installed a
                        # model that has actually looked at quadcopters.
                        det.label = "drone"
                        det.confidence = max(det.confidence,
                                             named.confidence)
                        det.named_by_optical = True
                    elif named.label in CAN_OVERRULE:
                        det.label = named.label
                        det.confidence = named.confidence
                        det.named_by_optical = True
                    else:
                        det.optical_guess = (f"{named.label} "
                                             f"{named.confidence:.2f}")
            if tr is not None:
                tr.label, tr.confidence = det.label, det.confidence

        # --- the optical camera doing its own job, not just answering
        optical_dets, pairs, unmatched_optical = [], {}, []
        ran_optical = False

        # No homography yet: run the optical detector over the WHOLE frame -
        # there is no thermal footprint to confine it to until we have one -
        # and let the calibrator watch for frames it can pair unambiguously.
        if (self.autocal is not None and self.H is None
                and self.cross_cue and optical is not None):
            # Keep what it found. This used to be handed to the calibrator and
            # then dropped, so res.optical_detections stayed empty for as long
            # as the rig was uncalibrated - which left both optical panes
            # completely blank at exactly the moment an operator is watching
            # them to see whether anything is working at all.
            optical_dets = self.optical.update(optical)
            ran_optical = True
            H_new = self.autocal.observe(detections, optical_dets,
                                         res.thermal_c.shape)
            if H_new is not None:
                self.H = H_new
                self._roi_set = False
                self._save_calibration(H_new)

        if self.cross_cue and optical is not None and self.H is not None:
            if not self._roi_set:
                self.optical.set_roi_from_homography(
                    self.H, res.thermal_c.shape, optical.shape, pad=12)
                self._roi_set = True
            # On the frame the calibration lands, the detector has already run
            # once over the whole frame. Running it again here would put two
            # updates of the same frame into the background model.
            if not ran_optical:
                optical_dets = self.optical.update(optical)
            pairs, unmatched_optical = associate(detections, optical_dets,
                                                 self.H)

        stage["optical"] = time.monotonic() - t_stage
        t_stage = time.monotonic()

        self.frame_index += 1
        now = self.now()

        # --- learn the site, then judge against what has been learned.
        # Order matters: observe() first so the z-map the scorer reads is
        # this frame's, not the previous one's.
        # Always observe: a frozen model still has to be READ every frame, or
        # its persistence counters never move and a settled intruder is
        # invisible. `learn` gates only the update inside.
        # The optical camera learning the same place. Run at the optical
        # detector's reduced rate: a Sobel over 640x480 every frame is real
        # work on a Pi, and a scene does not change meaningfully in 120 ms.
        optical_z = None
        if optical is not None:
            if self.optical_site is None:
                self.optical_site = OpticalBaseline(shape=optical.shape[:2],
                                                    fps=self.fps)
            self._optical_tick += 1
            if self._optical_tick % max(1, self.optical.every_n) == 0:
                optical_z = self.optical_site.observe(optical, learn=self.learn)

        self.site.observe(res.thermal_c, tracks,
                          exclude_ids=self._alerting, learn=self.learn)
        if self.learn:
            for det in detections:
                self.site.note_hotspot(det.hotspot_c)
        self.dwell.update(tracks, now)

        stage["site"] = time.monotonic() - t_stage
        t_stage = time.monotonic()

        assessments = []
        evidence = {}
        for det in detections:
            tr = tracks.get(det.track_id)
            dwell_s = self.dwell.dwell_s(det.track_id, now)

            # Ask the optical camera whether this patch of the scene looks
            # the way it normally does. Needs the homography to know where to
            # look, so on an uncalibrated rig this stays None - which Novelty
            # reads as "no opinion" rather than as a vote of zero.
            opt_nov, opt_why = None, ()
            if (self.optical_site is not None and self.H is not None
                    and not self.optical_site.learning):
                opt_nov, opt_why = self.optical_site.score(
                    homography.map_box(self.H, det.box))

            novelty = self.site.score(det, tr, dwell_s,
                                      optical_novelty=opt_nov,
                                      optical_reasons=opt_why)

            # THERMAL -> OPTICAL. Only asked for objects with enough track
            # history to be worth a second opinion; asking every frame about
            # every two-pixel flicker would cost more than it is worth.
            # Only ask optical about objects that are actually interesting.
            # A settled track in learned traffic is not worth a crop, a
            # segmentation and a shape fit on every frame - and on a Pi that
            # saved work is the difference between headroom and none. The
            # trigger is the site model or the classifier showing something,
            # which is exactly "an anomaly starts the cross-cue".
            optical_ev = None
            interesting = (novelty.overall >= 0.30
                           or det.label == "drone"
                           or det.confidence == 0.0)
            if (self.cross_cue and optical is not None and self.H is not None
                    and tr is not None and tr.hits >= 4 and interesting):
                optical_ev = verify_optical(optical, self.H, det.box)
                evidence[det.track_id] = optical_ev
            judgement = None
            if self.policy is not None:
                judgement = self.policy.judge(
                    det.label, det.centroid, dwell_s=dwell_s,
                    speed_px=(tr.speed_px if tr is not None else 0.0),
                    when=now)
            a = assess_track(det, tr, novelty, dwell_s, optical_ev,
                             judgement=judgement)
            # Why it is NOT being called an aircraft, in the reasons where an
            # operator will read it. "unknown" with no explanation is the same
            # silence that made the previous confident answer untrustworthy.
            if det.not_airborne:
                a.reasons = list(a.reasons) + [det.not_airborne]
            elif det.named_by_optical:
                a.reasons = list(a.reasons) + [
                    f"optical camera recognises it as a {det.label} "
                    f"({det.confidence:.2f})"]
            elif det.optical_guess:
                # Said as a guess, because that is what it is. The list the
                # optical model was trained on does not contain the thing we
                # are looking for.
                a.reasons = list(a.reasons) + [
                    f"optical's nearest match is {det.optical_guess}, but "
                    f"its class list has no drone in it - not evidence"]
            if optical_ev is not None:
                named = None
                if recognitions and self.H is not None:
                    named = best_overlap(
                        recognitions, homography.map_box(self.H, det.box))
                self._note_cross(
                    "thermal asks optical", a.key, a.label,
                    f"warm mover, {det.peak_c:.1f} C peak, "
                    f"{det.contrast_c:+.1f} C above background - what is it?",
                    self._optical_answer(optical_ev, named), a.threat)
            self._maybe_escalate(a, det, tr, res, optical_ev)
            assessments.append(a)

        # Nothing is "settled" until the site has been learned.
        #
        # A settled object is defined by contrast with a baseline: it was not
        # here, the model knows what here looks like, and now it is here. A
        # model that has learned nothing has no such contrast to offer, and
        # every object in the room is equally new to it - which is why an
        # unlearned rig reports the furniture as arrivals. During a learning
        # run it is worse than useless: the operator is deliberately
        # rebuilding the definition of normal, and anything reported against
        # a half-built one is noise by construction.
        statics = ([] if not self.settled_ready
                   else self._unclaimed_statics(self.site.static_anomalies(),
                                                detections))
        assessments.extend(assess_static(a) for a in statics)

        # OPTICAL -> THERMAL. Everything the optical camera found that no
        # thermal detection claimed. This is the direction that covers
        # thermal's blind spots, so it runs even when thermal is silent.
        confirmed = 0
        optical_cues = {}
        dwells = self.optical_contacts.update(unmatched_optical, now)
        for odet in unmatched_optical:
            th = verify_thermal(res.thermal_c, self.H, odet.box)
            if th.warm:
                confirmed += 1
            key = self.optical_contacts.key_for(odet.centroid)
            oa = assess_optical_only(odet, th, key, dwells.get(key, 0.0))
            named = best_overlap(recognitions, odet.box)
            if named is not None:
                oa.label = named.label
                oa.reasons = list(oa.reasons) + [
                    f"optical recognises it as a {named.label} "
                    f"({named.confidence:.2f})"]
            assessments.append(oa)
            # Keyed by box because an optical detection has no id of its own
            # and the renderer has only the detection in hand.
            if len(optical_cues) < MAX_OPTICAL_SHOWN:
                optical_cues[tuple(odet.box)] = self.cues.number(key, now)
            dwell = dwells.get(key, 0.0)
            self._note_cross(
                "optical asks thermal", key, oa.label,
                f"something here thermal never claimed, still for "
                f"{dwell:.0f} s - is it warm?",
                self._thermal_answer(th), oa.threat)
            self._maybe_escalate_optical(oa, odet.box, res, th,
                                         "optical-only contact")

        # Uncalibrated: nothing above ran, because every one of those paths
        # needs a homography to know where thermal is looking. The contacts
        # still get numbers and still get drawn. They are deliberately NOT
        # assessed - with no mapping, thermal cannot be asked whether the
        # thing is warm, and a threat score built on an unaskable question is
        # worse than no score.
        if self.H is None:
            for odet in optical_dets[:MAX_OPTICAL_SHOWN]:
                key = self.optical_contacts.key_for(odet.centroid)
                optical_cues[tuple(odet.box)] = self.cues.number(key, now)

        # What the OPTICAL site model can see sitting there. The optical
        # detector is motion-based, so an object that arrives and stops falls
        # out of it within seconds; this is the only optical channel that
        # keeps reporting one. Needs no homography, so it works on an
        # uncalibrated rig - which is the whole point of drawing it.
        optical_regions = []
        if self.optical_site is not None and self.settled_ready:
            optical_regions = self.optical_site.settled_regions()
            if self.H is not None and detections:
                # Calibrated: a thermal track already carries this object and
                # already has a number. Reporting the patch as well would put
                # two different numbers on one thing.
                mapped = [homography.map_box(self.H, d.box) for d in detections]
                optical_regions = self._unclaimed_regions(optical_regions,
                                                          mapped)
            for r in optical_regions:
                self.cues.number(r.key, now)
                # A settled patch had no verdict at all until now: it was
                # drawn, numbered, and then dropped. That left the cold-soaked
                # airframe - the case this whole optical half was built for -
                # visible on a pane and absent from every alert.
                #
                # OpticalRegion carries box, centroid, key and dwell_s, which
                # is all assess_optical_only reads, so the reverse-cue verdict
                # already written for a moving optical contact applies here
                # unchanged. Its threat climbs with persistence, which is
                # exactly the evidence a stationary object has to offer.
                th = (verify_thermal(res.thermal_c, self.H, r.box)
                      if self.H is not None else None)
                ra = assess_optical_only(r, th, r.key, r.dwell_s)
                named = best_overlap(recognitions, r.box)
                if named is not None:
                    # Say what it is in the verdict itself. An operator
                    # reading "settled object, 90 s" has to go and look; one
                    # reading "handbag (0.81)" already knows.
                    ra.label = named.label
                    ra.reasons = list(ra.reasons) + [
                        f"optical recognises it as a {named.label} "
                        f"({named.confidence:.2f})"]
                assessments.append(ra)
                self._note_cross(
                    "optical asks thermal", r.key, ra.label,
                    (f"a {named.label} has been sitting here for "
                     f"{r.dwell_s:.0f} s - is it warm?" if named is not None
                     else f"this patch has not looked right for "
                          f"{r.dwell_s:.0f} s and nothing moved - is "
                          f"anything warm there?"),
                    self._thermal_answer(th), ra.threat)
                self._maybe_escalate_optical(ra, r.box, res, th,
                                             "settled optical patch")

        # Number everything on screen, then release the numbers of things
        # that have been gone a while. Pruning after the refresh, never
        # before, or an object present this very frame loses its number.
        for a in assessments:
            self.cues.number(a.key, now)
        self.cues.prune(now)

        assessments.sort(key=lambda a: a.threat, reverse=True)

        stage["assess"] = time.monotonic() - t_stage
        res.stage_ms = {k: round(v * 1000.0, 1) for k, v in stage.items()}

        res.detections = detections
        res.tracks = tracks
        res.mask = self.detector.last_mask
        res.optical_detections = optical_dets
        res.recognitions = recognitions
        res.recogniser_stats = self.recogniser.stats()
        if self.recogniser2 is not None:
            st2 = self.recogniser2.stats()
            res.recogniser_stats = dict(res.recogniser_stats, second=st2)
        res.optical_evidence = evidence
        res.optical_roi = self.optical.roi
        res.cross = {
            "optical_candidates": len(optical_dets),
            "optical_shape_rejects": self.optical.rejected_shape,
            "optical_hidden": max(0, len(optical_dets) - len(optical_cues)),
            "optical_noise": round(self.optical.noise, 1),
            "optical_threshold": round(self.optical.effective_threshold, 1),
            "paired_with_thermal": len(pairs),
            "optical_only": len(unmatched_optical),
            "optical_only_confirmed_warm": confirmed,
            "shape_checks": len(evidence),
            "shape_usable": sum(1 for e in evidence.values() if e.usable),
            "enabled": bool(self.cross_cue),
            "auto_calibration": (self.autocal.status()
                                 if self.autocal is not None else None),
        }
        res.assessments = assessments
        res.static_anomalies = statics
        res.site_stats = self.site.stats()
        res.optical_site_stats = (self.optical_site.stats()
                                  if self.optical_site is not None else None)
        res.cross_log = list(self.cross_log)
        res.watching = self._watch_list(assessments, now)
        res.cue_numbers = self.cues.snapshot()
        res.optical_cues = optical_cues
        # Stationary correspondences, for an operator who has moved about,
        # stopped, and is watching the pair counter sit short of the minimum.
        if (self.autocal is not None and self.H is None
                and optical_regions and statics):
            H_new = self.autocal.observe_settled(statics, optical_regions,
                                                 res.thermal_c.shape)
            if H_new is not None:
                self.H = H_new
                self._roi_set = False
                self._save_calibration(H_new)

        res.optical_regions = optical_regions
        res.learning = self.learning_status()
        res.identifications = {
            a.key: self.escalator.result_for(a.key).as_dict()
            for a in assessments
            if self.escalator.result_for(a.key) is not None}
        res.escalation = self.escalator.status()
        res.alerts = self.alerts.update(assessments, now)
        res.alert_history = self.alerts.history(now)

        res.frame_index = self.frame_index
        self._alerting = {a.track_id for a in assessments
                          if a.kind == "track" and a.level == "alert"}

        self._times.append((time.monotonic() - t0, capture_s))
        if len(self._times) > 30:
            self._times.pop(0)
        n = len(self._times)
        res.proc_ms = 1000.0 * sum(a for a, _ in self._times) / n
        res.capture_ms = 1000.0 * sum(b for _, b in self._times) / n

        self.result = res
        return res

    # ------------------------------------------------------- escalation
    def _maybe_escalate(self, assessment, det, track, res, optical_ev):
        """Hand a hard case to a bigger model, without waiting for it.

        Deliberately fires AFTER the local verdict is formed and attached to
        the assessment. The alert has already been raised or not raised on
        local evidence; whatever comes back is an annotation, and the system
        behaves identically when the network is gone.
        """
        esc = self.escalator
        if not esc.enabled:
            return
        # Two triggers. High threat, as before - and DOUBT, which is the case
        # a remote model is actually good for: something is there, it is not
        # nothing, and nothing local can name it.
        unsure = (assessment.label in ("unknown", "")
                  or assessment.confidence < UNSURE_CONF)
        if assessment.threat < self.escalate_at and not (
                unsure and assessment.threat >= ESCALATE_DOUBT_AT):
            return
        key = assessment.key
        if not esc.should_ask(key):
            return

        features = det.as_dict()
        features.update({
            "threat": round(assessment.threat, 3),
            "dwell_s": round(assessment.dwell_s, 1),
            "speed_px": round(track.speed_px, 2) if track else 0.0,
            "straightness": round(track.straightness, 2) if track else 0.0,
            "range_note": (
                "At 1.70 mrad/px, a 0.25 m airframe spans ~8 px at 18 m and "
                "~2 px at 73 m, so a small blob means distance, not a small "
                "aircraft."),
            "local_verdict": f"{assessment.label} at "
                             f"{assessment.confidence:.2f} confidence",
        })
        why = getattr(det, "not_airborne", None)
        if why:
            # Tell it what we already ruled out, so the reply is about the
            # open question rather than about the one that is settled.
            features["ruled_out"] = (
                f"local rules say this cannot be an aircraft: {why}")
        esc.ask(key, features,
                thermal_crop=self._thermal_crop(res, det),
                optical_crop=self.crop_optical(det, margin=12))

    def _maybe_escalate_optical(self, assessment, box, res, thermal_check,
                                note):
        """Ask a bigger model about something only the optical camera has.

        The existing escalation path runs inside the thermal detection loop
        and takes a thermal detection, so an optical contact could never
        reach it however strange it looked. The box here is in OPTICAL pixels
        and no homography is required, which matters: this has to work on an
        uncalibrated rig, where optical is the only sensor saying anything.

        Two triggers rather than one. Threat, as everywhere else - and cold,
        still and unidentified, which is the signature of the parked airframe
        and never produces a high local threat quickly enough to be useful.
        """
        esc = self.escalator
        if not esc.enabled or res.optical is None:
            return
        cold_and_still = (thermal_check is not None and thermal_check.found
                          and not thermal_check.warm
                          and assessment.dwell_s >= ESCALATE_COLD_DWELL_S)
        if assessment.threat < self.escalate_at and not cold_and_still:
            return
        if not esc.should_ask(assessment.key):
            return

        x, y, w, h = [int(v) for v in box]
        features = {
            "sensor": "optical only - this object has no thermal detection",
            "why_asked": note,
            "box": [x, y, w, h],
            "area_px": int(w * h),
            "aspect": round(w / max(h, 1), 2),
            "dwell_s": round(assessment.dwell_s, 1),
            "threat": round(assessment.threat, 3),
            "thermal_says": self._thermal_answer(thermal_check),
            "range_note": (
                "A stationary object with no heat signature that has been "
                "in place for a while is either scene clutter or an airframe "
                "that has landed and cooled to ambient. Shape is the only "
                "thing left to tell them apart."),
        }
        esc.ask(assessment.key, features, thermal_crop=None,
                optical_crop=self._optical_patch(res, box))

    @staticmethod
    def _optical_patch(res, box, margin=16):
        """Crop of the optical frame around an OPTICAL-space box.

        crop_optical() maps a thermal box through the homography; this is for
        objects that were found in the optical frame to begin with and must
        not require a calibration to be sent anywhere.
        """
        if res.optical is None:
            return None
        h, w = res.optical.shape[:2]
        x, y, bw, bh = [int(v) for v in box]
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
        if x1 <= x0 or y1 <= y0:
            return None
        return res.optical[y0:y1, x0:x1].copy()

    @staticmethod
    def _thermal_crop(res, det, margin=6, pad_c=None):
        """Colourised thermal patch around a detection, for the remote model."""
        if res.thermal_c is None:
            return None
        import cv2
        h, w = res.thermal_c.shape[:2]
        x, y, bw, bh = det.box
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
        if x1 <= x0 or y1 <= y0:
            return None
        patch = res.thermal_c[y0:y1, x0:x1]
        lo, hi = float(patch.min()), float(patch.max())
        span = max(hi - lo, 0.5)
        u8 = ((patch - lo) * (255.0 / span)).clip(0, 255).astype("uint8")
        img = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
        # Upscale: a 20x16 crop is unreadable to a vision model, and nearest
        # neighbour keeps the pixel structure honest rather than inventing
        # smooth edges that were never measured.
        scale = max(1, int(160 / max(img.shape[1], 1)))
        if scale > 1:
            img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                             interpolation=cv2.INTER_NEAREST)
        return img

    @staticmethod
    def _unclaimed_statics(anomalies, detections, pad=6):
        """Drop settled regions that a live track is already reporting.

        A drone that lands while we are watching is seen by both paths - the
        tracker never let go of it, and the site model watched it become part
        of the scene. Reporting both is two alarms for one aircraft, so the
        track wins: it carries a classification and the static region does not.
        """
        if not detections:
            return anomalies
        out = []
        for a in anomalies:
            ax, ay, aw, ah = a.box
            claimed = False
            for det in detections:
                dx, dy, dw, dh = det.box
                if (ax < dx + dw + pad and dx < ax + aw + pad
                        and ay < dy + dh + pad and dy < ay + ah + pad):
                    claimed = True
                    break
            if not claimed:
                out.append(a)
        return out

    @staticmethod
    def _unclaimed_regions(regions, boxes, pad=16):
        """Drop optical settled patches that a thermal detection already has.

        The optical mirror of _unclaimed_statics, taking already-mapped boxes
        because the caller has done the homography once for all of them. The
        pad is larger than the thermal one because these are optical pixels -
        the same physical slop covers more of them.
        """
        if not boxes:
            return regions
        out = []
        for r in regions:
            ax, ay, aw, ah = r.box
            if not any(ax < bx + bw + pad and bx < ax + aw + pad
                       and ay < by + bh + pad and by < ay + ah + pad
                       for bx, by, bw, bh in boxes):
                out.append(r)
        return out

    # ------------------------------------------------------------ site model
    def accept_assessment(self, key):
        """Operator marks one reported object as normal for this site.

        Routed by which camera found it. An optical contact's box is in
        OPTICAL pixels and handing it to the thermal site model taught the
        wrong cells an unrelated lesson - the object stayed reported, the
        operator clicked again, and somewhere across the frame a patch of
        thermal baseline quietly learned something false.
        """
        for a in self.result.assessments:
            if a.key != key:
                continue
            if a.kind == "optical" and self.optical_site is not None:
                cells = self.optical_site.accept(a.box)
                # And in the thermal model too, if there is a mapping - the
                # same object is in both fields of view and the operator
                # meant the object, not one camera's opinion of it.
                if self.H is not None:
                    cells += self.site.accept(
                        homography.map_box(np.linalg.inv(self.H), a.box))
            else:
                cells = self.site.accept(a.box)
            if a.kind == "track" and a.track_id in self.dwell.records:
                del self.dwell.records[a.track_id]
            self.optical_contacts.seen.pop(key, None)
            self.alerts.clear(key)
            return cells
        return 0

    def refresh_objects(self):
        """Throw away the held object names and take a fresh look.

        The names are held for a few passes so a confidence sitting on the
        threshold does not flicker. The cost of that is a stale name after
        somebody has actually moved the furniture, and no amount of waiting
        fixes it faster than asking again. This is the ask-again.
        """
        self.recogniser.forget()
        return {"cleared": True, "model": self.recogniser.stats()["model"]}

    def save_site(self, path):
        return self.site.save(path)

    # ------------------------------------------------------------ views
    def optical_box(self, det):
        """Where a thermal detection lands in the optical frame."""
        if self.H is None:
            return None
        return homography.map_box(self.H, det.box)

    def crop_optical(self, det, margin=8):
        """Optical patch for a thermal detection - the fusion handoff.

        This is what a shape classifier or a VLM would be handed. It is also
        the thing that silently breaks when the homography drifts: the crop
        still returns an image, just of the wrong place.
        """
        if self.H is None or self.result.optical is None:
            return None
        x, y, w, h = self.optical_box(det)
        H_img, W_img = self.result.optical.shape[:2]
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(W_img, x + w + margin)
        y1 = min(H_img, y + h + margin)
        if x1 <= x0 or y1 <= y0:
            return None
        return self.result.optical[y0:y1, x0:x1].copy()
