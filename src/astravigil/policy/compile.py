"""Turn a site policy written in plain English into reviewable rules.

This is the one place a language model earns its keep in this system, and the
reason is when it runs rather than what it does.

    AUTHORING TIME (here)     an operator writes what the site is for, in
                              prose. A model turns that into structured rules
                              once. A human reads the result and edits it.

    RUNTIME (rules.py)        pure arithmetic over that file. No model, no
                              network, no variance. Same input, same output,
                              every frame, whether or not anything is reachable.

Put the model in the frame loop instead and you get: latency you cannot bound,
an answer that can differ between two identical frames, a hard dependency on a
third party for a perimeter sensor, and nothing to show a reviewer. Put it
here and a hallucination is a line in a JSON file that someone reads before it
governs anything. That is the whole argument.

Two backends, both optional:

  local     an Ollama model on the machine. Preferred. A site security policy
            is exactly the kind of document that should not be posted to a
            third-party API, and this runs offline.
  hosted    Featherless, reusing the existing client.

Neither is required. hardware/../policy examples ship as JSON and can be
written by hand - the model is a convenience for authoring, never a dependency.
"""
import json
import os
import re
import urllib.error
import urllib.request

from .rules import Rule, SitePolicy, Zone, validate

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# 3B is the smallest size that reliably emits valid JSON against a schema this
# fiddly. Below that the failure mode is silent - plausible rules with invented
# fields - which is worse here than refusing.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct")

SCHEMA_PROMPT = """You convert a site security policy written in English into \
strict JSON rules for a counter-drone sensor watching one fixed site.

Output STRICT JSON only. No prose, no markdown, no code fences.

{
  "site": "<short name for the site>",
  "zones": [
    {"name": "<lowercase_identifier>", "description": "<what this area is>"}
  ],
  "rules": [
    {
      "id": "<short-kebab-id>",
      "zone": "<zone name, or * for anywhere>",
      "classes": ["drone"|"bird"|"vehicle"|"person"|"aircraft"|"unknown"|"*"],
      "verdict": "permitted"|"prohibited"|"restricted",
      "severity": <0.0-1.0>,
      "hours": [<start hour 0-23>, <end hour 0-23>],
      "days": [0..6 , Monday is 0],
      "max_dwell_s": <seconds an object may stay still>,
      "reason": "<one short sentence an operator will read on an alert>"
    }
  ]
}

Rules for you to follow:
- Omit "hours", "days" and "max_dwell_s" entirely when the text does not \
state them. Do not invent limits.
- "permitted" means expected there and should not raise an alarm.
- "prohibited" means it must not be there at all.
- "restricted" means allowed only within a stated limit, so it REQUIRES \
"max_dwell_s". Use it for loitering limits.
- severity: 1.0 for a direct threat to the protected asset, 0.5 for something \
irregular, 0.0 for permitted things.
- Do not create zones the text does not mention. Use "*" if the text speaks \
about the whole site.
- "reason" is shown to a human during an incident. Write it as a statement of \
the rule, not a restatement of the detection.
"""


class CompileError(RuntimeError):
    pass


def _extract_json(text):
    """Models still wrap JSON in prose or fences no matter how firmly asked."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, depth = text.find("{"), 0
    if start < 0:
        raise CompileError(f"no JSON object in model reply: {text[:200]}")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as exc:
                    raise CompileError(f"malformed JSON: {exc}") from exc
    raise CompileError("unterminated JSON object in model reply")


def _ollama(prompt, model=None, url=None, timeout=180):
    url = (url or OLLAMA_URL).rstrip("/") + "/api/chat"
    body = json.dumps({
        "model": model or OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SCHEMA_PROMPT},
                     {"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())["message"]["content"]
    except urllib.error.URLError as exc:
        raise CompileError(
            f"cannot reach Ollama at {url} ({exc}). Start it with "
            f"'ollama serve' and pull a model: 'ollama pull {model or OLLAMA_MODEL}'"
        ) from exc


def _featherless(prompt, model=None, timeout=180):
    from ..llm.featherless import BASE_URL, FeatherlessClient  # noqa: F401
    key = os.environ.get("FEATHERLESS_API_KEY")
    if not key:
        raise CompileError("FEATHERLESS_API_KEY is not set")
    body = json.dumps({
        "model": model or os.environ.get("FEATHERLESS_MODEL",
                                         "Qwen/Qwen2.5-7B-Instruct"),
        "messages": [{"role": "system", "content": SCHEMA_PROMPT},
                     {"role": "user", "content": prompt}],
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return data["choices"][0]["message"]["content"]
    except urllib.error.URLError as exc:
        raise CompileError(f"Featherless request failed: {exc}") from exc


def compile_policy(text, backend="local", model=None, zones=None):
    """English policy -> SitePolicy. Review the result before trusting it.

    zones maps zone name -> polygon, supplied separately because a language
    model cannot know where the apron is in your camera's pixels. It names the
    areas; you draw them.
    """
    reply = (_ollama(text, model) if backend == "local"
             else _featherless(text, model))
    data = _extract_json(reply)

    known = dict(zones or {})
    zone_objs = []
    for z in data.get("zones", []):
        name = str(z.get("name", "")).strip().lower().replace(" ", "_")
        if not name:
            continue
        zone_objs.append(Zone(name, polygon=known.get(name),
                              description=z.get("description", "")))

    rules = []
    for i, r in enumerate(data.get("rules", [])):
        try:
            rules.append(Rule(
                id=str(r.get("id") or f"rule-{i + 1}"),
                zone=str(r.get("zone", "*")).strip().lower().replace(" ", "_"),
                classes=r.get("classes") or ["*"],
                verdict=str(r.get("verdict", "prohibited")),
                severity=float(r.get("severity", 0.8)),
                hours=r.get("hours"),
                days=r.get("days"),
                max_dwell_s=r.get("max_dwell_s"),
                min_speed_px=r.get("min_speed_px"),
                reason=str(r.get("reason", "")),
            ))
        except (TypeError, ValueError) as exc:
            raise CompileError(f"rule {i + 1} is malformed: {exc}") from exc

    policy = SitePolicy(site=data.get("site", ""), zones=zone_objs,
                        rules=rules, source_text=text)
    return policy, validate(policy)
