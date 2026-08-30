"""The dashboard page. Kept as one string so the app has no template dir."""

from .render import OPTICAL_VIEW_ROT, THERMAL_VIEW_ROT

_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AstraVigil</title>
<style>
  :root {
    --bg:#0d1014; --panel:#161b22; --line:#232b36;
    --fg:#e6edf3; --dim:#8b949e;
    --drone:#eb3c3c; --bird:#3caaeb; --unknown:#8b949e; --ok:#3fb950;
    --alert:#f0503c; --watch:#d29922; --nominal:#3fb950; --settled:#c964d6;
    /* A ceiling on pane height, not a fixed height. Each pane's width comes
       from a column sized to its own shape, so the heights already agree to
       within a gap's worth of rounding; the cap only stops a very wide
       screen turning five panes into five billboards. Stating the height
       outright looked equal on one monitor and letterboxed on a narrower
       one, because a fixed height a pane is too narrow to fill is exactly
       the black bar this was meant to remove. */
    --view-h:46vh;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif; }
  header { display:flex; align-items:baseline; gap:16px; flex-wrap:wrap;
           padding:12px 20px; border-bottom:1px solid var(--line); }
  h1 { font-size:16px; margin:0; letter-spacing:.06em; }
  .tag { font-size:11px; color:var(--dim); text-transform:uppercase;
         letter-spacing:.09em; }
  .stats { margin-left:auto; display:flex; gap:18px; flex-wrap:wrap;
           align-items:center; }
  .stat b { font-variant-numeric:tabular-nums; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%;
         background:var(--ok); margin-right:6px; }
  .dot.warn { background:var(--watch); }

  /* The uplink light. Four states, not two: three of the ways this fails
     leave the internet working perfectly and only the identification service
     broken, and an operator told "offline" would go and check a cable that
     has nothing wrong with it. */
  .dot.net-ok { background:var(--ok); }
  .dot.net-auth, .dot.net-busy { background:var(--watch); }
  .dot.net-down { background:var(--alert); }
  .dot.net-unknown { background:var(--dim); }
  main { padding:16px 20px 32px; }

  /* --- alerts ------------------------------------------------------- */
  /* min-height reserves a row of space. Without it, an alert opening or
     closing changes the height of this block and everything below it
     jumps - which is exactly the movement that made this panel hard to
     read while it sat above the camera panes. */
  /* Three fixed-height columns. The height is FIXED, not min- or max-, and
     that is the whole point: an alert arriving or clearing must not change
     the height of this block, or the four camera panes below it get shunted
     down the page every time something happens. Overflow scrolls inside the
     column instead. */
  .board { display:grid; grid-template-columns:repeat(3,1fr); gap:12px;
           margin-bottom:14px; }
  .col { background:var(--panel); border:1px solid var(--line);
         border-radius:8px; display:flex; flex-direction:column;
         height:186px; overflow:hidden; }
  .col-alert   { border-color:#5a2230; }
  .col-watch   { border-color:#5a4a1e; }
  .col-nominal { border-color:var(--line); }
  .col h3 { margin:0; padding:7px 10px; font-size:11px; font-weight:600;
            text-transform:uppercase; letter-spacing:.07em; color:var(--dim);
            border-bottom:1px solid var(--line); display:flex; gap:8px;
            align-items:center; flex:none; }
  .col h3 span:last-child { margin-left:auto; font-variant-numeric:tabular-nums;
            color:var(--fg); }
  .stack { flex:1; min-height:0; overflow-y:auto; padding:8px;
           display:flex; flex-direction:column; gap:7px; }
  .stack:empty::after { content:"none"; color:#4d5661; font-size:12px;
           padding:2px; }
  /* Compact form: two lines, fixed height, three to a column. The full
     reason text is on the title attribute rather than wrapped, because a card
     that grows with its reasons would put the column back to changing height
     - which is the thing this board exists to avoid. */
  .stack .alert { display:block; padding:5px 8px; font-size:11px;
                  border-radius:6px; }
  .stack .ln1 { display:flex; align-items:center; gap:6px; white-space:nowrap;
                overflow:hidden; }
  .stack .ln1 b { color:var(--fg); font-weight:600; }
  .stack .ln1 .thr { font-variant-numeric:tabular-nums; color:var(--fg); }
  .stack .ln1 .age { color:var(--dim); font-size:10px; }
  .stack .ln1 .badge { font-size:9px; padding:1px 5px; }
  .stack .acts { margin-left:auto; display:flex; gap:4px; flex:none; }
  .stack .acts button { font-size:9.5px; padding:1px 5px; border-radius:4px; }
  .stack .ln2 { color:var(--dim); font-size:10.5px; margin-top:2px;
                overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  /* Scrollbars only inside the columns, and unobtrusive. */
  .stack::-webkit-scrollbar { width:8px; }
  .stack::-webkit-scrollbar-thumb { background:#2c343e; border-radius:4px; }
  @media (max-width:900px) { .board { grid-template-columns:1fr; }
                             .views { grid-template-columns:1fr; } }
  .alert { display:grid; gap:2px 14px; padding:10px 14px; border-radius:8px;
           background:var(--panel); border:1px solid var(--line);
           border-left:4px solid var(--nominal);
           grid-template-columns:auto 1fr auto; align-items:center; }
  .alert.alert-alert  { border-left-color:var(--alert);
                        background:linear-gradient(90deg,#2a1614,var(--panel) 45%); }
  .alert.alert-watch  { border-left-color:var(--watch); }
  .alert.kind-static  { border-left-color:var(--settled); }
  .alert .who { font-weight:600; font-size:15px; letter-spacing:.02em; }
  .alert .when { font-variant-numeric:tabular-nums; color:var(--dim);
                 font-size:12px; }
  .alert .why { grid-column:1 / -1; color:var(--dim); font-size:12.5px; }
  .alert .why span::after { content:" · "; color:var(--line); }
  .ident { color:#9fe8e8; }
  .ident.bad { color:var(--dim); font-style:italic; }
  .ident .pay { color:var(--alert); }
  .alert .why span:last-child::after { content:""; }
  .alert .acts { display:flex; gap:6px; }
  .sensors { font-size:10px; letter-spacing:.08em; text-transform:uppercase;
             color:var(--dim); border:1px solid var(--line); padding:1px 6px;
             border-radius:99px; margin-left:6px; vertical-align:middle; }
  .badge { font-size:11px; padding:2px 8px; border-radius:99px;
           text-transform:uppercase; letter-spacing:.07em; font-weight:600; }
  .b-alert { background:#3a1a17; color:var(--alert); }
  .b-watch { background:#332813; color:var(--watch); }
  .b-nominal { background:#16281c; color:var(--nominal); }
  button { background:#20272f; color:var(--fg); border:1px solid var(--line);
           border-radius:6px; padding:5px 10px; font:inherit; font-size:12px;
           cursor:pointer; }
  button:hover { background:#2a323c; }
  .quiet { padding:14px; border:1px dashed var(--line); border-radius:8px;
           color:var(--dim); text-align:center; }

  /* --- learning meter ------------------------------------------------ */
  .learn { display:flex; align-items:center; gap:10px; font-size:12px;
           color:var(--dim); }
  .bar { width:110px; height:6px; border-radius:3px; background:#20272f;
         overflow:hidden; }
  .bar i { display:block; height:100%; background:var(--ok); width:0; }

  /* --- views --------------------------------------------------------- */
  /* Columns are sized in proportion to what each pane actually shows, so a
     portrait pane gets a portrait column and a landscape one a landscape
     column. Every image then fills its own box at the same row height with
     no letterboxing and nothing cropped - the black bars either side of the
     thermal panes were the cost of forcing one 4:3 box on both shapes. */
  .views { display:grid; gap:14px; align-items:stretch;
           grid-template-columns:__TH_FR__fr __TH_FR__fr __OP_FR__fr
                                 __OP_FR__fr __OP_FR__fr; }
  figure { margin:0; background:var(--panel); border:1px solid var(--line);
           border-radius:8px; overflow:hidden;
           display:flex; flex-direction:column; }
  /* Each pane declares its own shape, filled from the view rotation at
     import time, and its column is sized to match.
     Still contain, never cover: cropping a thermal pane would hide detections
     at the edge of frame, which is exactly where something entering the site
     first appears. With the column matched to the image there is nothing left
     to letterbox, so contain costs nothing here. */
  figure img { flex:none; width:100%; height:auto;
               max-height:var(--view-h);
               object-fit:contain; display:block; background:#000; }
  figure.thermal img { aspect-ratio:__TH_ASPECT__; }
  figure.optical img { aspect-ratio:__OP_ASPECT__; }
  /* One line, not a paragraph.
     Five panes each carrying four sentences is more text than picture, and
     an operator reads none of it after the first session. What they do need
     at a glance is which colour means what; the prose is still there, on
     hover, for the session where somebody is learning the console. */
  figcaption { flex:1; padding:6px 10px; font-size:11px; color:var(--dim);
               border-top:1px solid var(--line); cursor:help;
               display:flex; align-items:center; flex-wrap:wrap; gap:0; }
  figcaption b { color:var(--fg); font-weight:600; letter-spacing:.04em;
                 text-transform:uppercase; font-size:10px; margin-right:6px; }
  .k { display:inline-block; width:8px; height:8px; border-radius:2px;
       margin:0 3px 0 9px; }
  .k-alert { background:var(--alert); } .k-watch { background:var(--watch); }
  .k-ok { background:var(--nominal); } .k-set { background:var(--settled); }
  .k-opt { background:#3cebeb; } .k-name { background:#7bd88f; }

  .trail { background:var(--panel); border:1px solid var(--line);
           border-radius:8px; padding:6px 10px; margin-bottom:14px;
           max-height:190px; overflow-y:auto; }
  .trail .row { display:grid; grid-template-columns:30px 150px 1fr 1fr 54px;
                gap:10px; padding:5px 0; font-size:12px;
                border-bottom:1px solid var(--line); align-items:baseline; }
  .trail .row:last-child { border-bottom:0; }
  .trail .dirn { color:var(--dim); text-transform:uppercase;
                 letter-spacing:.06em; font-size:10px; }
  .trail .q { color:var(--fg); }
  .trail .a { color:var(--dim); }
  .trail .t { text-align:right; font-variant-numeric:tabular-nums; }
  .trail .empty { color:var(--dim); margin:6px 0; }

  /* Closed alerts. An alert that vanishes off the board with no trace makes
     an operator doubt they saw it, and doubt is the one thing a console
     cannot afford to produce. */
  .hist { display:flex; gap:8px; flex-wrap:wrap; margin:-4px 0 4px; }
  .hist .h { background:var(--panel); border:1px solid var(--line);
             border-left:3px solid var(--line); border-radius:6px;
             padding:4px 9px; font-size:11px; color:var(--dim);
             font-variant-numeric:tabular-nums; }
  .hist .h.lv-alert { border-left-color:var(--alert); }
  .hist .h.lv-watch { border-left-color:var(--watch); }
  .hist .h b { color:var(--fg); font-weight:600; }
  .hist .none { color:var(--dim); font-size:11px; }

  /* The object number drawn on the object itself, so a row in this list and
     a badge on the picture are visibly the same thing. Deliberately NOT called
     a "cue" anywhere an operator can read: this dashboard already uses
     "cross-cue" for one camera asking the other a question, and a column
     headed CUE read as a claim that a question had been asked. */
  .cue { display:inline-block; min-width:22px; text-align:center;
         padding:0 5px; border:1px solid var(--line); border-radius:5px;
         background:#0d1014; color:var(--fg); font-size:11px;
         font-variant-numeric:tabular-nums; line-height:17px; }
  .cue.none { color:var(--dim); border-style:dashed; }

  /* Things that arrived and stayed. Not alerts - most of them are somebody's
     bag - but each one is an open question, and the answer is a person
     looking at it once. Dismissing teaches the baseline, so the same bag is
     not asked about again tomorrow. */
  .watch { background:var(--panel); border:1px solid var(--line);
           border-radius:8px; margin-bottom:14px; overflow:hidden; }
  .watch .w { display:grid; align-items:center; gap:10px;
              grid-template-columns:34px 1fr 1.4fr 62px 74px;
              padding:7px 10px; font-size:12px;
              border-bottom:1px solid var(--line); }
  .watch .w:last-child { border-bottom:0; }
  .watch .w.lv-alert { background:#2a1414; }
  .watch .w.lv-watch { background:#241d10; }
  .watch .what b { font-weight:600; }
  .watch .said { color:var(--dim); overflow:hidden;
                 text-overflow:ellipsis; white-space:nowrap; }
  .watch .age { text-align:right; color:var(--dim);
                font-variant-numeric:tabular-nums; }
  .watch .empty { padding:14px; color:var(--dim); text-align:center;
                  font-size:12px; }

  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.09em;
       color:var(--dim); margin:26px 0 8px; display:flex; gap:10px;
       align-items:center; }

  table { width:100%; border-collapse:collapse; background:var(--panel);
          border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  th,td { padding:7px 10px; text-align:right; font-variant-numeric:tabular-nums;
          border-bottom:1px solid var(--line); }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2)
        { text-align:left; }
  th { font-size:11px; text-transform:uppercase; letter-spacing:.06em;
       color:var(--dim); font-weight:600; }
  tbody tr:last-child td { border-bottom:none; }
  .lbl { font-weight:600; }
  .drone { color:var(--drone); } .bird { color:var(--bird); }
  .unknown { color:var(--unknown); }
  .lv-alert { color:var(--alert); font-weight:600; }
  .lv-watch { color:var(--watch); }
  .lv-nominal { color:var(--dim); }
  .empty { padding:18px; color:var(--dim); text-align:center; }
  .note { color:var(--dim); font-size:12px; margin-top:10px; max-width:78ch; }

  /* --- site scan ------------------------------------------------------ */
  .scanwrap { position:relative; }
  #scan { position:absolute; inset:0; width:100%; height:100%;
          pointer-events:none; display:none; }
  #scan.on { display:block; }
  #oscan { position:absolute; inset:0; width:100%; height:100%;
           pointer-events:none; display:none; }
  #oscan.on { display:block; }
  .scanhud { position:absolute; left:0; right:0; bottom:34px; display:none;
             justify-content:center; pointer-events:none; }
  .scanhud.on { display:flex; }
  .scanhud div { background:#0b1220dd; border:1px solid #29d3d322;
                 border-radius:10px; padding:8px 16px; font-size:12px;
                 color:#9fe8e8; letter-spacing:.06em;
                 font-variant-numeric:tabular-nums; }
  .scanhud b { color:#e6ffff; }
  .learnbtn { background:#12313a; border-color:#29d3d244; color:#9fe8e8; }
  .learnbtn:hover { background:#17414d; }
  .learnbtn.busy { background:#3a2a12; border-color:#d2992244; color:#f0c674; }

  /* --- kiosk ---------------------------------------------------------- */
  body.kiosk { cursor:none; }
  body.kiosk .devonly { display:none !important; }
  #esc { position:fixed; inset:auto 0 0 0; display:none; justify-content:center;
         pointer-events:none; padding:0 0 26px; z-index:50; }
  #esc.show { display:flex; }
  #esc div { background:#161b22ee; border:1px solid var(--line);
             border-radius:10px; padding:12px 22px; font-size:14px;
             letter-spacing:.04em; box-shadow:0 8px 30px #0009; }
  #esc b { color:var(--watch); font-variant-numeric:tabular-nums; }
  #kioskbar { display:none; gap:8px; align-items:center; }
  body.kiosk #kioskbar { display:flex; }
  /* The way out has to be visible. Ctrl+Shift+Esc is grabbed by the desktop
     on Linux before the page ever sees it, which leaves an operator in a
     fullscreen window with no address bar and a shortcut that does nothing. */
  button.danger { border-color:#5a2230; color:#ff9d8a; }
  button.danger:hover { background:#3a1a17; }
  button.danger.armed { background:#5a2230; color:#fff; border-color:#8a3444; }
</style>
</head>
<body>
<header>
  <h1>ASTRAVIGIL</h1>
  <span class="tag" id="src">—</span>
  <span class="learn">
    <span class="tag">site</span>
    <span class="bar"><i id="scenebar"></i></span>
    <span id="scenetxt">—</span>
  </span>
  <div class="stats">
    <span class="stat"><span class="dot" id="health"></span><span id="status">starting</span></span>
    <span class="stat"><span class="tag">fps</span> <b id="fps">—</b></span>
    <span class="stat"><span class="tag">detect</span> <b id="proc">—</b> ms</span>
    <span class="stat"><span class="tag">draw</span> <b id="draw">—</b> ms</span>
    <span class="stat"><span class="tag">headroom</span> <b id="head">—</b></span>
    <span class="stat"><span class="tag">tracks</span> <b id="tracks">—</b></span>
    <span class="stat"><span class="tag">cross-cue</span> <b id="cross">—</b></span>
    <span class="stat"><span class="tag">frame</span> <b id="frame">—</b></span>
    <span class="stat" id="netstat" title="Checking the uplink…">
      <span class="dot net-unknown" id="net"></span><span id="nettxt">uplink</span>
    </span>
    <button id="learnbtn" class="learnbtn" onclick="toggleLearn()"
            title="Watch the site and build a model of what is normal here">learn this site</button>
    <button class="devonly" onclick="saveSite()" title="Write the learned site model to disk">save site model</button>
    <span id="kioskbar">
      <button onclick="kioskRestart()" title="Restart the console">restart</button>
    </span>
    <!-- Deliberately outside #kioskbar, so it shows in every mode. Closing a
         browser tab does not stop the pipeline: the sensor keeps running, the
         port stays bound and the camera stays claimed. Without a visible stop
         there is nothing on the page that actually ends a session. -->
    <button id="closebtn" class="danger" onclick="kioskClose()"
            title="Stop AstraVigil - releases the camera and frees the port">stop</button>
  </div>
</header>

<main>
  <h2>Situational picture</h2>
  <div class="board">
    <section class="col col-alert">
      <h3><span class="badge b-alert">alert</span><span id="n-alert">0</span></h3>
      <div class="stack" id="s-alert"></div>
    </section>
    <section class="col col-watch">
      <h3><span class="badge b-watch">watch</span><span id="n-watch">0</span></h3>
      <div class="stack" id="s-watch"></div>
    </section>
    <section class="col col-nominal">
      <h3><span class="badge b-nominal">nominal</span><span id="n-nominal">0</span></h3>
      <div class="stack" id="s-nominal"></div>
    </section>
  </div>

  <div class="views">
    <figure class="thermal">
      <img src="/stream/thermal" alt="thermal">
      <figcaption title="Warm movers against cold sky. Detection runs on this frame and every other pane is derived from it. Boxes are coloured by threat rather than by class, because what an operator needs at a glance is which of the things on screen matters."><b>Thermal</b>detection<span class="k k-alert"></span>alert<span class="k k-watch"></span>watch<span class="k k-set"></span>settled</figcaption>
    </figure>
    <figure class="scanwrap thermal">
      <img src="/stream/site" alt="site">
      <canvas id="scan"></canvas>
      <div class="scanhud" id="scanhud"><div></div></div>
      <figcaption title="What the thermal camera has learned about this place. Green is learned traffic - the brighter the cell, the more often something has moved through it. Magenta has been off its learned temperature long enough to be an object rather than a fly-past. Numbered boxes are what the model is being fed right now, and the same number appears on every other pane and in the tracks table."><b>Site</b>thermal<span class="k k-ok"></span>traffic<span class="k k-set"></span>settled</figcaption>
    </figure>
    <figure class="optical">
      <img src="/stream/optical" alt="optical">
      <figcaption title="Thermal detections mapped through the homography, plus what this camera found on its own: cyan is something moving that no thermal detection claimed, magenta is a patch its site model says has been sitting there. Those two need no homography, so they are still drawn while the rig is calibrating - but an object seen by only one sensor cannot share a number with the other, because nothing yet knows they are the same thing. Coloured labels are the object recogniser naming what is in the room, and those names are what answer thermal when it asks what something is."><b>Optical</b>mapped + own<span class="k k-opt"></span>optical-only<span class="k k-set"></span>settled<span class="k k-name"></span>named<span id="objnote"></span></figcaption>
    </figure>
    <figure class="scanwrap optical">
      <img src="/stream/optical_site" alt="optical site">
      <canvas id="oscan"></canvas>
      <figcaption title="What the optical camera has learned about this place, one square per cell of the model. While it is learning, green is how much history a cell has and an outlined square has none yet. Once learned, green is where change normally happens - an empty map means nothing has moved through, which is a real answer rather than a failure. Magenta has looked wrong long enough to be an object; amber is off baseline now."><b>Site</b>optical<span class="k k-ok"></span>learned<span class="k k-set"></span>settled<span class="k k-watch"></span>live</figcaption>
    </figure>
    <figure class="optical">
      <img src="/stream/overlay" alt="overlay">
      <figcaption title="Only the warm part of the thermal frame, warped into optical space. The white outline is its edge and it should sit on the object that is actually hot; registration is a judgement about whether two edges coincide. If a numbered box misses its warm patch, the frame matching has drifted and every optical crop downstream is being taken from the wrong place."><b>Overlay</b>registration</figcaption>
    </figure>
  </div>

  <h2>Watching <span class="tag" id="watchnote"></span></h2>
  <div class="watch" id="watch">
    <p class="empty">nothing has been left in view</p>
  </div>

  <h2>Closed <span class="tag" id="histnote"></span></h2>
  <div class="hist" id="hist"><span class="none">nothing has closed yet</span></div>

  <h2>Cross-cue — what each camera asked the other</h2>
  <div class="trail" id="trail"><p class="empty">nothing asked yet</p></div>

  <h2>Tracks</h2>
  <table>
    <thead><tr>
      <th>Obj</th><th>Track</th><th>Class</th><th>Conf</th><th>Threat</th>
      <th>Site</th>
      <th>Still</th><th>Peak °C</th><th>Hotspot</th>
      <th>Area px</th><th>Parts</th><th>Flap</th><th>Straight</th><th>Frames</th>
    </tr></thead>
    <tbody id="rows"><tr><td class="empty" colspan="14">no detections</td></tr></tbody>
  </table>

  <p class="note"><b>Threat</b> fuses two independent answers and lets either one
    raise the alarm on its own: what the object <i>is</i> (the classifier) and whether it
    <i>belongs</i> (the site baseline). They are combined with a noisy-OR, so a confident
    "bird" cannot talk down a screaming site anomaly and a quiet site cannot talk down a
    confident drone.<br><br>
    <b>Site</b> is that second number by itself — position the site has never seen traffic
    in, a patch that is off its learned temperature, a size or heat unlike anything
    recorded here, or an object that has stopped moving. <b>Still</b> is how long it has
    been stationary. An object with no motion cue at all — one that landed before the
    system booted — is caught only by the site model, and appears as a
    <span style="color:var(--settled)">settled object</span>.<br><br>
    <b>Hotspot</b> is peak temperature above the object's <i>own</i> mean —
    the motor signature. Contrast against the sky is not used, because a 31&nbsp;°C bird
    against −5&nbsp;°C sky is exactly as bright as a drone. <b>Flap</b> is the coefficient
    of variation of blob area over ~1.6&nbsp;s: a bird modulates its silhouette, a rigid
    airframe does not. Confidences come from hand-tuned rules, not from a model fitted to
    real footage.</p>
</main>

<div id="esc"><div></div></div>

<script>
const fmt = (v, d=2) => (v === undefined || v === null) ? "—" : Number(v).toFixed(d);

/* ---------------------------------------------------------------- kiosk
   Active only when the launcher opened the page with ?kiosk=1, so pressing
   the sequence during development does nothing.

   The counter is shown on screen on purpose. An escape hatch you cannot tell
   is working is one you will assume is broken, and the person testing it is
   standing in front of a fullscreen window with no other way out. */
const KIOSK = new URLSearchParams(location.search).get("kiosk") === "1";
const ESC_NEEDED = 3, ESC_WINDOW_MS = 3000;
let escHits = [];

function escHint(html, ms) {
  const box = document.getElementById("esc");
  box.firstElementChild.innerHTML = html;
  box.classList.add("show");
  clearTimeout(escHint._t);
  escHint._t = setTimeout(() => box.classList.remove("show"), ms);
}

/* Two clicks, not one. A single-click kill on an always-on console is one
   sleeve-brush away from taking the sensor down mid-demo. Two clicks inside
   four seconds is the same "you meant this" test the triple-Esc was for,
   without depending on a shortcut the desktop intercepts. */
let closeArmed = 0;

function kioskClose() {
  const b = document.getElementById("closebtn");
  const now = Date.now();
  if (now - closeArmed > 4000) {
    closeArmed = now;
    b.classList.add("armed");
    b.textContent = "click again to stop";
    escHint("Click <b>again</b> to stop AstraVigil.", 4000);
    setTimeout(() => {
      if (Date.now() - closeArmed >= 4000) {
        b.classList.remove("armed");
        b.textContent = "stop";
      }
    }, 4200);
    return;
  }
  closeArmed = 0;
  b.classList.remove("armed");
  b.textContent = "stopping…";
  escHint("Stopping AstraVigil — the camera is being released.", 15000);
  fetch("/api/kiosk/exit", {method:"POST"}).catch(() => {});
}

async function kioskRestart() {
  escHint("Restarting the console…", 8000);
  try { await fetch("/api/kiosk/restart", {method:"POST"}); } catch (e) {}
  setTimeout(() => location.reload(), 3500);
}

if (KIOSK) {
  document.body.classList.add("kiosk");
  addEventListener("keydown", e => {
    if (!(e.ctrlKey && e.shiftKey)) return;
    if (e.code !== "Escape" && e.key !== "Escape") return;
    e.preventDefault();
    e.stopPropagation();
    const now = Date.now();
    escHits = escHits.filter(t => now - t < ESC_WINDOW_MS);
    escHits.push(now);
    if (escHits.length >= ESC_NEEDED) {
      escHits = [];
      escHint("Stopping AstraVigil — the camera is being released.", 15000);
      fetch("/api/kiosk/exit", {method:"POST"}).catch(() => {});
      return;
    }
    escHint(`Exit kiosk: <b>${escHits.length}</b> of <b>${ESC_NEEDED}</b>` +
            ` — press Ctrl+Shift+Esc again`, ESC_WINDOW_MS);
  }, true);

  // A kiosk browser has no address bar, so a stray middle-click or drag that
  // navigates away leaves the operator staring at nothing with no way back.
  addEventListener("contextmenu", e => e.preventDefault());
  addEventListener("dragstart", e => e.preventDefault());
}

async function post(url, body) {
  try {
    await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"},
                      body: JSON.stringify(body || {})});
  } catch (e) {}
  poll();
}
const accept = key => post("/api/accept", {key});
const ack = id => post("/api/ack", {id});
const saveSite = () => post("/api/save_site");

function esc(t) {
  return String(t).replace(/&/g, "&amp;").replace(/"/g, "&quot;")
                  .replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function alertCard(a) {
  const dur = a.duration_s >= 60
    ? Math.floor(a.duration_s / 60) + "m" + Math.round(a.duration_s % 60) + "s"
    : Math.round(a.duration_s) + "s";
  const why = a.reasons.concat(identText(a)).filter(Boolean).join(" · ");
  // The cue number if the object still has one, so a card and a badge on
  // the picture are the same object. Falls back to the track id for an alert
  // whose object has since left the scene and given its number back.
  const who = a.label + (a.cue ? " #" + a.cue
                               : (a.track_id ? " track " + a.track_id : ""));
  // Everything an operator needs at a glance is on line one; the evidence is
  // on line two, truncated, with the whole of it on hover.
  return `<div class="alert alert-${a.level} kind-${a.kind}"
       title="${esc(who + " - since " + a.opened_hms + ", " + dur +
                    ", threat " + fmt(a.threat) + " (peak " +
                    fmt(a.peak_threat) + ") - " + why)}">
    <div class="ln1">
      <span class="badge b-${a.level}">${a.level}</span>
      <b>${who}</b>
      <span class="thr">${fmt(a.threat)}</span>
      <span class="age">${dur}${a.acked ? " · ack" : ""}</span>
      <span class="acts">
        <button onclick="ack(${a.id})" title="Acknowledge">ack</button>
        <button onclick="accept('${a.key}')" title="Teach the site model that this is normal here">normal</button>
      </span>
    </div>
    <div class="ln2">${why}</div>
  </div>`;
}

/* ------------------------------------------------------------ site scan
   The grid is not an animation over the video - it IS the model. Every cell
   is one 8x8 px block of the thermal frame, and its brightness is how much
   real history that block has. A corner the camera cannot see properly stays
   visibly empty instead of being quietly assumed fine, which is the whole
   reason to show it rather than a spinner. */
let scanState = null;

function drawScan(stats) {
  const hud = document.getElementById("scanhud");
  const L = stats.learning || {};
  // Both cameras are learning the site, so both panes show it. Watching one
  // fill while the other sits still is what makes an operator think only
  // half the system is doing anything - and until now they were right.
  drawGrid(document.getElementById("scan"), stats.grid,
           stats.grid_w, stats.grid_h, L.active);
  drawGrid(document.getElementById("oscan"), stats.ogrid,
           stats.ogrid_w, stats.ogrid_h, L.active);
  if (!L.active || !stats.grid) { hud.classList.remove("on"); return; }
  hud.classList.add("on");
  scanHud(L);
}

function drawGrid(cv, cov, gw, gh, active) {
  if (!cv) return;
  if (!active || !cov || !gw || !gh) { cv.classList.remove("on"); return; }
  cv.classList.add("on");

  const img = cv.previousElementSibling;
  const W = img.clientWidth, H = img.clientHeight;
  if (!W || !H) return;
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== W * dpr || cv.height !== H * dpr) {
    cv.width = W * dpr; cv.height = H * dpr;
  }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);

  const cw = W / gw, ch = H / gh;
  const t = Date.now() / 1000;

  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const v = cov[y * gw + x] / 100;
      if (v <= 0.02) continue;
      // A slow diagonal shimmer so filled cells read as actively surveyed
      // rather than as a static heat map.
      const wave = 0.10 * Math.sin(t * 2.2 - (x + y) * 0.30);
      g.fillStyle = `rgba(41,211,211,${(v * 0.30 + wave * v).toFixed(3)})`;
      g.fillRect(x * cw, y * ch, cw - 1, ch - 1);
      if (v > 0.9) {
        g.strokeStyle = "rgba(120,255,255,0.22)";
        g.lineWidth = 1;
        g.strokeRect(x * cw + 0.5, y * ch + 0.5, cw - 2, ch - 2);
      }
    }
  }

  // Leading edge of the sweep, so there is a sense of progress even when the
  // scene is uniform and every cell fills at the same rate.
  const sweep = ((t * 0.35) % 1) * H;
  const grad = g.createLinearGradient(0, sweep - 26, 0, sweep + 26);
  grad.addColorStop(0, "rgba(41,211,211,0)");
  grad.addColorStop(0.5, "rgba(160,255,255,0.16)");
  grad.addColorStop(1, "rgba(41,211,211,0)");
  g.fillStyle = grad;
  g.fillRect(0, sweep - 26, W, 52);

}

function scanHud(L) {
  const hud = document.getElementById("scanhud");
  const pct = Math.round((L.progress || 0) * 100);
  hud.firstElementChild.innerHTML =
    `LEARNING THIS SITE &nbsp; <b>${pct}%</b> &nbsp;·&nbsp; ` +
    `thermal <b>${Math.round((L.coverage || 0) * 100)}%</b> &nbsp;·&nbsp; ` +
    `optical <b>${Math.round((L.optical_coverage || 0) * 100)}%</b> ` +
    `&nbsp;·&nbsp; <b>${Math.round(L.remaining_s || 0)}s</b> left`;
}

async function toggleLearn() {
  const btn = document.getElementById("learnbtn");
  const active = btn.classList.contains("busy");
  const body = active ? {action: "stop"}
                      : {action: "start", seconds: 90, reset: true};
  btn.disabled = true;
  try { await fetch("/api/site/learn", {method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify(body)}); } catch (e) {}
  btn.disabled = false;
  poll();
}

// Plain-text form of the remote identification, for the compact card's
// single evidence line and its hover text.
function identText(a) {
  const id = a.identification;
  if (!id) return "";
  if (!id.ok) return "ID unavailable: " + id.error;
  const pay = (id.payload === "likely" || id.payload === "possible")
    ? ", payload " + id.payload : "";
  return "ID: " + id.summary + pay;
}

function identLine(a) {
  const id = a.identification;
  if (!id) return "";
  if (!id.ok) return `<span class="ident bad">ID: ${id.error}</span>`;
  const pay = (id.payload === "likely" || id.payload === "possible")
    ? ` <b class="pay">payload ${id.payload}</b>` : "";
  return `<span class="ident">ID: ${id.summary}${pay}</span>`;
}

async function poll() {
  try {
    const r = await fetch("/api/state");
    const { detections, alerts, stats } = await r.json();
    const site = stats.site || {};

    document.getElementById("fps").textContent = fmt(stats.fps, 1);
    document.getElementById("proc").textContent = fmt(stats.proc_ms, 2);
    document.getElementById("draw").textContent =
      fmt(stats.render_ms, 2) + (stats.view_fps ? " @" + stats.view_fps + "Hz" : "");
    document.getElementById("head").textContent =
      stats.headroom ? stats.headroom + "x" : "—";
    document.getElementById("tracks").textContent = stats.tracks ?? "—";
    const cx = stats.cross || {};
    document.getElementById("cross").textContent = cx.enabled
      ? `${cx.paired_with_thermal ?? 0}\u21c4 ${cx.optical_only ?? 0}opt`
      : "off";
    document.getElementById("frame").textContent = stats.frame ?? "—";

    // The uplink. Says WHICH failure, because "offline" and "your key was
    // rejected" send somebody to look at two completely different things.
    const net = (stats.escalation || {}).net || {};
    const nstate = net.state || "unknown";
    document.getElementById("net").className = "dot net-" + nstate;
    document.getElementById("nettxt").textContent = {
      ok: "online", auth: "key rejected", busy: "service busy",
      down: "offline", unknown: "checking"}[nstate] || nstate;
    document.getElementById("netstat").title =
      (net.detail || "checking the uplink") +
      (net.latency_ms != null ? ` · ${net.latency_ms} ms` : "") +
      (net.age_s != null ? ` · checked ${Math.round(net.age_s)}s ago` : "");
    document.getElementById("src").textContent =
      (stats.source ?? "") + (stats.calibrated ? " · calibrated" : " · uncalibrated");

    const scene = (site.scene_maturity ?? 0), act = (site.activity_maturity ?? 0);
    document.getElementById("scenebar").style.width = (scene * 100) + "%";
    document.getElementById("scenetxt").textContent =
      site.learning ? `learning scene ${(scene*100)|0}%`
                    : `learned · traffic ${(act*100)|0}%`;

    const dot = document.getElementById("health");
    const warm = stats.warmed_up;
    dot.className = "dot" + (warm ? "" : " warn");
    document.getElementById("status").textContent =
      warm ? "running" : "learning background";

    drawScan(stats);
    const lb = document.getElementById("learnbtn");
    if ((stats.learning || {}).active) {
      lb.classList.add("busy");
      lb.textContent = `learning ${Math.round((stats.learning.progress||0)*100)}%  ·  stop`;
    } else {
      lb.classList.remove("busy");
      lb.textContent = "learn this site";
    }

    // Split by level into three columns, newest first, three shown per column.
    // The count in each header is the TOTAL, so a column capped at three never
    // hides how many there really are - a board that silently shows 3 of 40
    // would be worse than one that scrolls.
    for (const level of ["alert", "watch", "nominal"]) {
      const of_level = alerts
        .filter(a => a.level === level)
        .sort((x, y) => (y.opened_at || 0) - (x.opened_at || 0));
      document.getElementById("n-" + level).textContent = of_level.length;
      const stack = document.getElementById("s-" + level);
      const html = of_level.slice(0, 3).map(alertCard).join("");
      // Only touch the DOM when it actually changed, or the browser throws
      // away scroll position and any half-made click four times a second.
      if (stack.dataset.sig !== html) {
        stack.dataset.sig = html;
        stack.innerHTML = html;
      }
    }

    // Everything that arrived and stayed. This is the settled channel's
    // whole output in one place: what it is, what the other camera said
    // about it, and a button that ends the question. Dismissing writes the
    // object into the baseline, so it is not raised again.
    const watch = document.getElementById("watch");
    const wl = stats.watching || [];
    const whtml = wl.length
      ? wl.map(w => `<div class="w lv-${w.level}">
           <span class="cue">${w.cue ?? "—"}</span>
           <span class="what"><b>${esc(w.label)}</b>
             <span class="said">${esc(w.reasons[0] || "")}</span></span>
           <span class="said" title="${esc(w.thermal)}">${esc(w.thermal)}</span>
           <span class="age">${Math.round(w.dwell_s)}s</span>
           <button onclick="accept('${w.key}')"
             title="This belongs here — fold it into the baseline and stop
                    asking">normal</button>
         </div>`).join("")
      : '<p class="empty">nothing has been left in view</p>';
    if (watch.dataset.sig !== whtml) {
      watch.dataset.sig = whtml;
      watch.innerHTML = whtml;
    }
    const orx = (stats.optical_site || {}).regions_found || 0;
    document.getElementById("watchnote").textContent = wl.length
      ? `settled objects · "normal" teaches the site and stops the question`
      : (orx > 6 ? `${orx} patches off baseline — the site model is stale, `
                 + `re-learn it` : "objects that arrived and stayed");

    // Where an alert goes when it disappears. It closes five seconds after
    // the object stops being seen, and every open and close is appended to
    // the log file named beside the heading - so a board that empties is a
    // scene that emptied, and it can be checked.
    const hist = document.getElementById("hist");
    const hl = stats.alert_history || [];
    const hhtml = hl.length
      ? hl.map(a => `<span class="h lv-${a.level}"
             title="${esc(a.label)} — opened ${a.opened_hms}, closed ${
               a.closed_hms || "—"}, peak threat ${fmt(a.peak_threat)}${
               a.acked ? ", acknowledged" : ""} · ${esc(a.reasons.join(" · "))}">
           <b>${esc(a.label)}</b> ${a.closed_hms || ""} ·
           ${Math.round(a.duration_s)}s · peak ${fmt(a.peak_threat)}
         </span>`).join("")
      : '<span class="none">nothing has closed yet</span>';
    if (hist.dataset.sig !== hhtml) {
      hist.dataset.sig = hhtml;
      hist.innerHTML = hhtml;
    }
    // Whether object naming is running at all, said out loud. A pane with no
    // labels on it looks identical whether the model found nothing or was
    // never installed.
    const rec = stats.recogniser || {};
    document.getElementById("objnote").textContent = !rec.available
      ? (rec.error ? " — object naming off: " + rec.error : "")
      : ` — ${rec.model} · ${(stats.objects || []).length} named · ` +
        `${fmt(rec.last_ms, 0)} ms every ${rec.every_n} frames`;

    document.getElementById("histnote").textContent = stats.alert_log
      ? `closes 5 s after the object is last seen · logged to ${stats.alert_log}`
      : "closes 5 s after the object is last seen";

    // The conversation between the two cameras. Question on the left,
    // answer in the middle, the threat it produced on the right - so the
    // reason a verdict moved is visible next to the verdict.
    const trail = document.getElementById("trail");
    const log = stats.cross_log || [];
    const thtml = log.length
      ? log.map(e => `<div class="row">
           <span class="cue${e.cue ? "" : " none"}">${e.cue ?? "—"}</span>
           <span class="dirn">${esc(e.dir)}</span>
           <span class="q">${esc(e.asked)}</span>
           <span class="a">${esc(e.answered)}</span>
           <span class="t ${e.threat >= 0.75 ? "lv-alert"
                            : e.threat >= 0.4 ? "lv-watch" : ""}"
             >${fmt(e.threat, 2)}</span>
         </div>`).join("")
      : '<p class="empty">nothing asked yet — both cameras need something to '
        + 'disagree about, and a calibration to compare through</p>';
    if (trail.dataset.sig !== thtml) {
      trail.dataset.sig = thtml;
      trail.innerHTML = thtml;
    }

    const body = document.getElementById("rows");
    if (!detections.length) {
      body.innerHTML = '<tr><td class="empty" colspan="14">no detections</td></tr>';
    } else {
      body.innerHTML = detections.map(d => `
        <tr>
          <td><span class="cue${d.cue ? "" : " none"}">${d.cue ?? "—"}</span></td>
          <td>#${d.track_id ?? "—"}</td>
          <td class="lbl ${d.label}">${d.label}</td>
          <td>${fmt(d.confidence)}</td>
          <td class="lv-${d.level}">${fmt(d.threat)}</td>
          <td>${fmt(d.site_risk)}</td>
          <td>${d.dwell_s >= 1 ? Math.round(d.dwell_s) + "s" : "—"}</td>
          <td>${fmt(d.peak_c, 1)}</td>
          <td>${fmt(d.hotspot_c)}</td>
          <td>${d.area_px}</td>
          <td>${d.parts}</td>
          <td>${fmt(d.flap_score, 3)}</td>
          <td>${fmt(d.straightness)}</td>
          <td>${d.hits ?? 0}</td>
        </tr>`).join("");
    }
  } catch (e) { /* keep polling through a hiccup */ }
}
setInterval(poll, 400);
poll();
</script>
</body>
</html>
"""


# A quarter or three-quarter turn swaps a pane's width and height, so the
# shape the browser must reserve depends on the view rotation - which is why
# this is filled in here rather than written into the stylesheet.
#
# The fr numbers are the aspect ratios themselves. Sizing each column in
# proportion to what it holds is what makes every pane come out the same
# height with none of them letterboxed.
def _shape(rot):
    return ("3 / 4", "0.75") if rot % 2 else ("4 / 3", "1.3333")


_TH_ASPECT, _TH_FR = _shape(THERMAL_VIEW_ROT)
_OP_ASPECT, _OP_FR = _shape(OPTICAL_VIEW_ROT)

PAGE = (_PAGE.replace("__TH_ASPECT__", _TH_ASPECT)
             .replace("__OP_ASPECT__", _OP_ASPECT)
             .replace("__TH_FR__", _TH_FR)
             .replace("__OP_FR__", _OP_FR))
