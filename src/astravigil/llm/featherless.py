"""Asking a hosted model what the object actually is.

The local system is deliberately offline and deterministic: rule-based
classification, statistical anomaly detection, nothing learned at runtime.
That is the right default for a perimeter sensor, and it stays that way. This
module is an ESCALATION layer bolted to the side of it - for the small number
of detections where the local answer is not good enough to act on and a slower,
larger model might say something useful.

Three rules keep it from becoming a dependency:

  IT NEVER BLOCKS THE PIPELINE. Requests go to a worker thread. Detection runs
  at frame rate whether the network is up, slow, or absent.

  IT NEVER GATES AN ALERT. The alert has already fired on local evidence by the
  time anything is sent. What comes back is an annotation on an existing
  alert, not permission to raise one. If the API is down, the system behaves
  exactly as it does now.

  IT IS RATE LIMITED AND ESCALATION-ONLY. One object, one question, with a
  cooldown. Sending every frame would be pointless, expensive, and would leak
  a continuous video feed of a protected site to a third party.

WHAT LEAVES THE MACHINE, and it is worth being blunt about this because this
is a security system: a cropped thermal patch and a cropped optical patch of
the detected object, plus the measured features. Not the whole frame, not the
site, not continuously. It is off unless explicitly enabled.

WHAT TO EXPECT BACK. Identifying a specific drone model from a 256x192 thermal
blob is not realistic and the prompt does not ask for it as though it were.
What is genuinely answerable from this imagery - and useful - is airframe
class, rotor count, rough size, and whether something is slung underneath.
That last one is the challenge brief's entire scenario.
"""
import base64
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("FEATHERLESS_BASE_URL",
                          "https://api.featherless.ai/v1")
DEFAULT_MODEL = os.environ.get("FEATHERLESS_MODEL",
                               "Qwen/Qwen2.5-VL-7B-Instruct")

# The API sits behind Cloudflare, which blocks requests by client signature.
# urllib identifies itself as "Python-urllib/3.x" by default, and that
# signature is refused: the reply is HTTP 403 carrying "error code: 1010",
# Cloudflare's "banned by browser signature", NOT anything Featherless said
# about the key or the model. Sending a real name is the whole fix.
#
# Overridable because a WAF rule is a moving target and the alternative to
# changing this string would be editing the source on the rig.
USER_AGENT = os.environ.get("FEATHERLESS_USER_AGENT", "AstraVigil/1.0")

# Statuses that mean "ask again", not "you asked wrong". A hosted model at
# capacity answers 503 with code "capacity_exhausted" and the words "please
# try again shortly" - it is busy, not misconfigured. Giving up on the first
# one wastes the whole escalation: the per-object cooldown is 60 s, so by the
# time we are allowed to ask about that track again the object has usually
# gone.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
RETRIES = int(os.environ.get("FEATHERLESS_RETRIES", "2"))
RETRY_BACKOFF_S = float(os.environ.get("FEATHERLESS_RETRY_BACKOFF_S", "2.0"))

# Other models to try when the first is at capacity or does not exist, most
# preferred first. Empty by default: a wrong guess at a model id is worse than
# no fallback, and which vision models an account can reach differs per plan.
#
#   FEATHERLESS_FALLBACK_MODELS=org/model-a,org/model-b
#
# scripts/check_featherless.py --list-models prints what this key can see.
FALLBACK_MODELS = tuple(
    m.strip() for m in
    os.environ.get("FEATHERLESS_FALLBACK_MODELS", "").split(",") if m.strip())


def _headers(api_key):
    return {"Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "Authorization": f"Bearer {api_key}"}
TIMEOUT_S = 45.0
COOLDOWN_S = 60.0          # per object, so one track cannot spam the API
MIN_GAP_S = 6.0            # global floor between any two calls
MAX_QUEUE = 4

SYSTEM_PROMPT = (
    "You are assisting a counter-UAS sensor that watches one fixed site. You "
    "are given measurements from a 256x192 uncooled LWIR thermal camera and, "
    "when available, a visible-light crop of the same object.\n\n"
    "Be calibrated. The imagery is low resolution and the object is usually "
    "small in frame. Identifying an exact commercial model is normally NOT "
    "possible from this data - say so rather than guessing a product name. "
    "What you can often judge: airframe class, rotor count, rough size, "
    "whether a payload is slung underneath, and whether the heat distribution "
    "is consistent with a powered multirotor.\n\n"
    "Reply with STRICT JSON only, no prose, no code fences:\n"
    "{\"class\": \"multirotor|fixed_wing|bird|person|vehicle|unknown\", "
    "\"rotors\": <int or null>, \"size_class\": \"micro|small|medium|large|unknown\", "
    "\"payload\": \"none|possible|likely|unclear\", "
    "\"candidate_models\": [\"...\"], "
    "\"confidence\": <0.0-1.0>, "
    "\"reasoning\": \"<one or two sentences>\"}"
)


class Identification:
    __slots__ = ("key", "ok", "cls", "rotors", "size_class", "payload",
                 "candidates", "confidence", "reasoning", "error", "model",
                 "latency_s", "at")

    def __init__(self, key, ok=False, error=None, model=None, latency_s=0.0,
                 **kw):
        self.key = key
        self.ok = ok
        self.error = error
        self.model = model
        self.latency_s = latency_s
        self.at = time.time()
        self.cls = kw.get("class") or kw.get("cls") or "unknown"
        self.rotors = kw.get("rotors")
        self.size_class = kw.get("size_class", "unknown")
        self.payload = kw.get("payload", "unclear")
        self.candidates = kw.get("candidate_models") or []
        self.confidence = float(kw.get("confidence") or 0.0)
        self.reasoning = kw.get("reasoning", "")

    def summary(self):
        if not self.ok:
            return f"identification unavailable ({self.error})"
        bits = [self.cls]
        if self.rotors:
            bits.append(f"{self.rotors} rotors")
        if self.size_class and self.size_class != "unknown":
            bits.append(self.size_class)
        if self.payload in ("possible", "likely"):
            bits.append(f"payload {self.payload}")
        head = ", ".join(bits)
        if self.candidates:
            head += f" - possibly {', '.join(self.candidates[:3])}"
        return f"{head} (remote model, {self.confidence:.2f})"

    def as_dict(self):
        return {"key": self.key, "ok": self.ok, "error": self.error,
                "class": self.cls, "rotors": self.rotors,
                "size_class": self.size_class, "payload": self.payload,
                "candidate_models": list(self.candidates),
                "confidence": round(self.confidence, 3),
                "reasoning": self.reasoning, "model": self.model,
                "latency_s": round(self.latency_s, 2),
                "summary": self.summary()}


def _png_b64(image_bgr):
    """BGR array to a base64 PNG data URI, or None."""
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return None
    import cv2
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def describe(features):
    """The measurements, as text. This is the part that always works.

    Vision support varies by model and by provider, so the numeric description
    is sent every time and the images are an extra. A text-only model still
    gets the geometry, the temperatures and the behaviour, which is most of
    what actually discriminates at these ranges.
    """
    f = features
    lines = [
        "Thermal measurements (HIKMICRO Mini2 Plus V2, 256x192, 1.70 mrad/px):",
        f"  bounding box      : {f.get('box')} px",
        f"  blob area         : {f.get('area_px')} px",
        f"  peak temperature  : {f.get('peak_c')} C",
        f"  mean temperature  : {f.get('mean_c')} C",
        f"  peak above own mean: {f.get('hotspot_c')} C  "
        f"(a powered airframe carries a hot core: battery/ESC/VTX)",
        f"  contrast vs background: {f.get('contrast_c')} C",
        f"  merged hot parts  : {f.get('parts')}",
        f"  aspect / extent / solidity: {f.get('aspect')} / "
        f"{f.get('extent')} / {f.get('solidity')}",
        "Behaviour:",
        f"  silhouette area variation (wingbeat cue): {f.get('flap_score')}  "
        f"(a flapping bird is high, a rigid airframe near zero)",
        f"  path straightness : {f.get('straightness')}",
        f"  speed             : {f.get('speed_px')} px/frame",
        f"  stationary for    : {f.get('dwell_s')} s",
        "Local system's own verdict:",
        f"  class {f.get('label')} at confidence {f.get('confidence')}, "
        f"threat {f.get('threat')}",
    ]
    if f.get("range_note"):
        lines.append(f"  {f['range_note']}")
    return "\n".join(lines)


class FeatherlessClient:
    """Minimal OpenAI-compatible client. No SDK, so nothing new to install."""

    def __init__(self, api_key=None, model=DEFAULT_MODEL, base_url=BASE_URL,
                 timeout=TIMEOUT_S):
        self.api_key = api_key or os.environ.get("FEATHERLESS_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Which model actually answered - may be a fallback, and the operator
        # should be told which model made the call they are reading.
        self.model_used = model

    @property
    def configured(self):
        return bool(self.api_key)

    def _post(self, payload):
        """One chat completion, with retries and model fallback.

        Retryable statuses are retried on the same model first, because a
        capacity blip usually clears in seconds and the first model is the one
        that was chosen. Only once that is exhausted - or the model turns out
        not to exist - do we move down FALLBACK_MODELS.

        Anything else is raised immediately and unchanged, so the caller's
        4xx handling still gets to strip the images and try text-only.
        """
        wanted = payload["model"]
        candidates = [wanted] + [m for m in FALLBACK_MODELS if m != wanted]
        last = None

        for model in candidates:
            body = dict(payload, model=model)
            for attempt in range(RETRIES + 1):
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=json.dumps(body).encode(),
                    headers=_headers(self.api_key),
                    method="POST")
                try:
                    with urllib.request.urlopen(
                            req, timeout=self.timeout) as r:
                        got = json.loads(r.read().decode())
                    self.model_used = model
                    return got["choices"][0]["message"]["content"]
                except urllib.error.HTTPError as exc:
                    last = exc
                    if exc.code in RETRY_STATUSES and attempt < RETRIES:
                        time.sleep(RETRY_BACKOFF_S * (attempt + 1))
                        continue
                    break
            if last is not None and last.code not in RETRY_STATUSES \
                    and last.code != 404:
                raise last
        raise last

    def identify(self, features, thermal_crop=None, optical_crop=None):
        if not self.configured:
            raise RuntimeError("FEATHERLESS_API_KEY is not set")

        content = [{"type": "text", "text": describe(features)}]
        for name, img in (("thermal", thermal_crop), ("optical", optical_crop)):
            uri = _png_b64(img)
            if uri:
                content.append({"type": "text",
                                "text": f"{name} crop of the object:"})
                content.append({"type": "image_url",
                                "image_url": {"url": uri}})

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": content}],
            "max_tokens": 400,
            "temperature": 0.2,
        }
        return self._post(payload)

    def identify_text_only(self, features):
        """Fallback for models that reject image content."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": describe(features)}],
            "max_tokens": 400,
            "temperature": 0.2,
        }
        return self._post(payload)


def _parse(text):
    """Pull the JSON object out of a reply, tolerating fences and chatter."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:160]!r}")
    return json.loads(t[start:end + 1])


class Escalator:
    """Queues identification requests and runs them off the hot path."""

    def __init__(self, client=None, cooldown_s=COOLDOWN_S,
                 min_gap_s=MIN_GAP_S, enabled=False):
        self.client = client or FeatherlessClient()
        self.enabled = enabled and self.client.configured
        self.cooldown_s = cooldown_s
        self.min_gap_s = min_gap_s
        self.results = {}          # key -> Identification
        self.pending = set()
        self._asked_at = {}
        self._last_call = 0.0
        self._q = queue.Queue(maxsize=MAX_QUEUE)
        self._lock = threading.Lock()
        self.calls = 0
        self.failures = 0
        if self.enabled:
            threading.Thread(target=self._worker, daemon=True).start()

    def status(self):
        return {"enabled": self.enabled,
                "configured": self.client.configured,
                "model": self.client.model,
                "calls": self.calls, "failures": self.failures,
                "pending": len(self.pending),
                "results": len(self.results)}

    def should_ask(self, key, now=None):
        if not self.enabled:
            return False
        now = time.time() if now is None else now
        with self._lock:
            if key in self.pending:
                return False
            last = self._asked_at.get(key)
            if last is not None and now - last < self.cooldown_s:
                return False
            if now - self._last_call < self.min_gap_s:
                return False
        return True

    def ask(self, key, features, thermal_crop=None, optical_crop=None):
        """Queue a question. Returns immediately; never raises on a full queue."""
        if not self.should_ask(key):
            return False
        with self._lock:
            now = time.time()
            self._asked_at[key] = now
            self._last_call = now
            self.pending.add(key)
        try:
            self._q.put_nowait((key, features, thermal_crop, optical_crop))
            return True
        except queue.Full:
            with self._lock:
                self.pending.discard(key)
            return False

    def result_for(self, key):
        return self.results.get(key)

    def _worker(self):
        while True:
            key, features, tc, oc = self._q.get()
            t0 = time.monotonic()
            try:
                try:
                    text = self.client.identify(features, tc, oc)
                except urllib.error.HTTPError as exc:
                    # Plenty of hosted models are text-only and reject image
                    # content with a 4xx. Retry without the images rather than
                    # reporting a failure - the measurements alone carry most
                    # of what discriminates at these ranges.
                    if 400 <= exc.code < 500:
                        text = self.client.identify_text_only(features)
                    else:
                        raise
                data = _parse(text)
                ident = Identification(key, ok=True, model=self.client.model,
                                       latency_s=time.monotonic() - t0, **data)
                self.calls += 1
            except Exception as exc:                    # never kill the thread
                self.failures += 1
                ident = Identification(key, ok=False, error=f"{type(exc).__name__}: {exc}",
                                       model=self.client.model,
                                       latency_s=time.monotonic() - t0)
            self.results[key] = ident
            with self._lock:
                self.pending.discard(key)
            self._q.task_done()
