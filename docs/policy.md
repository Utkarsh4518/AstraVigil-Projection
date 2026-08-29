# Site policy — what is *allowed* here

The system already learns what is **normal** at a site. This is the other
half: what a human says **should** happen there.

```bash
# author it (local model, nothing leaves the machine)
ollama serve && ollama pull qwen2.5:3b-instruct
python3 scripts/compile_policy.py --example

# review before it governs anything
python3 scripts/compile_policy.py --review configs/policy.json

# run with it
python3 scripts/run_dashboard.py --policy configs/policy.json
```

## Why this is a separate signal

**Normal** is descriptive and learned. `site_intelligence/baseline.py` watches
the site and measures what usually happens. It is honest about *this* site.

**Allowed** is prescriptive and authored. Nothing can learn it, because it is
a decision rather than a fact.

They diverge in the case that matters most. `learn_site.py` says it in its own
help text: *"Anything present and stationary while this runs becomes part of
the definition of normal... and also the obvious way to blind the system on
purpose."* A learned baseline **structurally cannot** tell you an object should
not be there if it was there while the model learned. A policy is immune —
someone wrote it knowing what the site is for, before the sensor ever ran.

They also diverge the other way, which is where false-positive relief comes
from. A ground vehicle in the service lane is thermally novel *every single
time*, and alarming every time is how an operator learns to ignore the system.

Measured on the live pipeline, both directions:

| Object | identity | site | policy | threat |
|---|---|---|---|---|
| drone over apron | 0.31 | 0.91 | **1.00** prohibited | **1.00** |
| bird anywhere | 0.00 | 0.03 | permitted, suppresses | **0.03** |

The drone's classifier confidence was only 0.31 — four warm pixels is not much
to classify — and policy carried it to certain anyway. That is the point:
*where it is* can matter more than *what we could tell it was*.

## Where the language model runs, and where it does not

This is the design decision worth defending out loud.

```
AUTHORING TIME          RUNTIME
scripts/compile_policy  policy/rules.py
a model, once           arithmetic and set membership, every frame
reviewable output       deterministic, offline, no network
a human edits it        same input -> same output, always
```

Put a model in the frame loop and you get latency you cannot bound, answers
that differ between two identical frames, a third-party dependency inside a
perimeter sensor, and nothing to show a reviewer. Put it at authoring time and
a hallucination is **a line in a JSON file that someone reads before it
governs anything.**

So when a judge asks "what if the LLM gets it wrong" the answer is not "we
tuned the prompt" — it is that the model cannot reach runtime at all.

Prefer `--backend local`. A document naming your restricted areas and their
hours is exactly what should not be posted to a hosted API. `--backend hosted`
uses Featherless if you have no local model, and **neither is required** —
`configs/policy.json` can be written by hand and checked with `--review`.

## Rules

Three verdicts:

- **`permitted`** — expected here. Suppresses site novelty while *in scope*.
- **`prohibited`** — must not be here at all.
- **`restricted`** — allowed only within a stated limit; exceeding it is a
  violation. This is how "no loitering" is expressed, and it needs a
  `max_dwell_s`.

Scoping is by `zone` (a polygon in **thermal pixel** coordinates), `classes`,
`hours`, `days`, `max_dwell_s`, and `min_speed_px`.

### Scoped permission, never a blanket whitelist

A permission is a claim about *a class, in a place, at a time, behaving a
certain way* — never about an object. A vehicle permitted in the service lane
in daylight that sits motionless at 03:00 has left the terms of its permission
and is judged on site evidence again:

```
vehicle in lane, noon   permitted   0.00   ground vehicles use the lane in daylight
vehicle in lane, 3am    prohibited  0.20   outside permitted hours 06:00-22:00
unknown parked 300s     prohibited  0.80   stationary 300 s, over the 120 s limit
```

That follows the project's own safety principle: *known object + abnormal
behaviour* must stay flaggable. Whitelisting **objects** is what makes a
perimeter system safe to walk past.

Note the third line — a loiter rule catches the landed-drone case through
policy alone, independently of whether the baseline saw it arrive.

### How it joins the threat score

```
threat = 1 - (1 - identity) x (1 - site) x (1 - policy)
```

The same noisy-OR the other two signals use: **sufficient on its own, never
able to veto the others.** A policy that could cancel a detection would be a
way to switch the sensor off by editing a config file.

The one thing policy may do downward is damp *site novelty* for explicitly
expected activity, and only to `SITE_SUPPRESSION = 0.25` — not to zero.
Policy states that a class belongs in a place, not that a particular object is
harmless, so a permitted vehicle behaving like nothing ever seen in that lane
can still climb back to an alert on its own.

Identity risk is never suppressed. A drone in a lane where vehicles are
permitted still carries its full identity risk.

## Zones need coordinates

A model can *name* the apron. It cannot know where the apron is in your
camera's pixels. After compiling, add a polygon to each zone in the JSON —
thermal pixel coordinates, 256 x 192 — or rules scoped to that zone will never
match. `--review` warns about zones with fewer than three points.

## With no policy at all

The `--policy` flag is optional and the system behaves exactly as before
without it, judging on learned statistics alone. Policy is added evidence, not
a prerequisite.
