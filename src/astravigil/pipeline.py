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
import time

import numpy as np


from .alerting import AlertManager
from .calibration import homography
from .classification.rules import classify
from .detection.thermal import ThermalDetector
from .drivers.thermal.calibration import calibrate_frame
from .detection.optical import OpticalDetector
from .fusion import (OpticalContactLog, assess_optical_only, assess_static,
                     assess_track, associate, verify_optical, verify_thermal)
from .llm import Escalator
from .site_intelligence import DwellMonitor, SiteBaseline
from .site_intelligence.baseline import MIN_LEARNED_FRAMES
from .tracking.tracker import Tracker


class Result:
    __slots__ = ("thermal_raw", "thermal_c", "optical", "detections",
                 "tracks", "mask", "proc_ms", "capture_ms", "frame_index",
                 "healthy", "assessments", "alerts", "static_anomalies",
                 "site_stats", "novelty", "optical_detections",
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
        self.frame_index = 0
        self.healthy = True

        self.assessments = []       # one verdict per object, both paths
        self.alerts = []            # currently open alerts
        self.static_anomalies = []
        self.site_stats = {}
        self.novelty = None         # cell z-map upsampled to frame size

        self.optical_detections = []   # what the optical camera found alone
        self.optical_evidence = {}     # track_id -> optical's shape answer
        self.cross = {}                # cue counts, for the dashboard
        self.optical_roi = None        # where optical is allowed to look
        self.learning = {}             # operator-driven learning run
        self.identifications = {}      # key -> remote model's answer
        self.escalation = {}           # escalator health


class Pipeline:
    def __init__(self, source, H=None, threshold_c=1.5, site=None,
                 alerts=None, fps=25.0, clock="wall", learn=True,
                 cross_cue=True, optical_every_n=3, escalator=None,
                 escalate_at=0.75, policy=None, auto_calibrate=True,
                 calibration_path=None):
        self.source = source
        self.H = H
        # Authored site policy - what is ALLOWED here, as opposed to what the
        # baseline has learned is normal here. Optional: with no policy the
        # system behaves exactly as before, judging on statistics alone.
        self.policy = policy
        self.detector = ThermalDetector(threshold_c=threshold_c)
        # Optical frames carry six times the pixels of thermal ones and far
        # more clutter, so this runs at a fraction of the rate. Thermal keeps
        # every frame: it is the primary sensor and it is the cheap one.
        self.cross_cue = cross_cue
        self.optical = OpticalDetector(every_n=optical_every_n)
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

    def learning_status(self):
        s = self.site.stats()
        cov = float(np.clip(self.site.ref_n / MIN_LEARNED_FRAMES, 0, 1).mean())
        out = {"active": self._session is not None,
               "coverage": round(cov, 3),
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
            out["progress"] = round(min(elapsed / max(target, 1e-6), cov), 3)
            out["done"] = elapsed >= target
        return out

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

        detections = self.detector.update(res.thermal_c)
        tracks = self.tracker.update(detections)

        for det in detections:
            tr = tracks.get(det.track_id)
            det.label, det.confidence = classify(det, tr)
            if tr is not None:
                tr.label, tr.confidence = det.label, det.confidence

        # --- the optical camera doing its own job, not just answering
        optical_dets, pairs, unmatched_optical = [], {}, []

        # No homography yet: run the optical detector over the WHOLE frame -
        # there is no thermal footprint to confine it to until we have one -
        # and let the calibrator watch for frames it can pair unambiguously.
        if (self.autocal is not None and self.H is None
                and self.cross_cue and optical is not None):
            found = self.optical.update(optical)
            H_new = self.autocal.observe(detections, found,
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
            optical_dets = self.optical.update(optical)
            pairs, unmatched_optical = associate(detections, optical_dets,
                                                 self.H)

        self.frame_index += 1
        now = self.now()

        # --- learn the site, then judge against what has been learned.
        # Order matters: observe() first so the z-map the scorer reads is
        # this frame's, not the previous one's.
        # Always observe: a frozen model still has to be READ every frame, or
        # its persistence counters never move and a settled intruder is
        # invisible. `learn` gates only the update inside.
        self.site.observe(res.thermal_c, tracks,
                          exclude_ids=self._alerting, learn=self.learn)
        if self.learn:
            for det in detections:
                self.site.note_hotspot(det.hotspot_c)
        self.dwell.update(tracks, now)

        assessments = []
        evidence = {}
        for det in detections:
            tr = tracks.get(det.track_id)
            dwell_s = self.dwell.dwell_s(det.track_id, now)
            novelty = self.site.score(det, tr, dwell_s)

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
            self._maybe_escalate(a, det, tr, res, optical_ev)
            assessments.append(a)

        statics = self._unclaimed_statics(self.site.static_anomalies(),
                                          detections)
        assessments.extend(assess_static(a) for a in statics)

        # OPTICAL -> THERMAL. Everything the optical camera found that no
        # thermal detection claimed. This is the direction that covers
        # thermal's blind spots, so it runs even when thermal is silent.
        confirmed = 0
        dwells = self.optical_contacts.update(unmatched_optical, now)
        for odet in unmatched_optical:
            th = verify_thermal(res.thermal_c, self.H, odet.box)
            if th.warm:
                confirmed += 1
            key = self.optical_contacts.key_for(odet.centroid)
            assessments.append(
                assess_optical_only(odet, th, key, dwells.get(key, 0.0)))

        assessments.sort(key=lambda a: a.threat, reverse=True)

        res.detections = detections
        res.tracks = tracks
        res.mask = self.detector.last_mask
        res.optical_detections = optical_dets
        res.optical_evidence = evidence
        res.optical_roi = self.optical.roi
        res.cross = {
            "optical_candidates": len(optical_dets),
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
        res.learning = self.learning_status()
        res.identifications = {
            a.key: self.escalator.result_for(a.key).as_dict()
            for a in assessments
            if self.escalator.result_for(a.key) is not None}
        res.escalation = self.escalator.status()
        res.alerts = self.alerts.update(assessments, now)

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
        if not esc.enabled or assessment.threat < self.escalate_at:
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
        })
        esc.ask(key, features,
                thermal_crop=self._thermal_crop(res, det),
                optical_crop=self.crop_optical(det, margin=12))

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

    # ------------------------------------------------------------ site model
    def accept_assessment(self, key):
        """Operator marks one reported object as normal for this site."""
        for a in self.result.assessments:
            if a.key == key:
                cells = self.site.accept(a.box)
                if a.kind == "track" and a.track_id in self.dwell.records:
                    del self.dwell.records[a.track_id]
                self.alerts.clear(key)
                return cells
        return 0

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
