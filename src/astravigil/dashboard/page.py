"""The dashboard page. Kept as one string so the app has no template dir."""

PAGE = r"""<!doctype html>
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
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif; }
  header { display:flex; align-items:baseline; gap:16px; flex-wrap:wrap;
           padding:12px 20px; border-bottom:1px solid var(--line); }
  h1 { font-size:16px; margin:0; letter-spacing:.06em; }
  .tag { font-size:11px; color:var(--dim); text-transform:uppercase;
         letter-spacing:.09em; }
  .stats { margin-left:auto; display:flex; gap:18px; flex-wrap:wrap; }
  .stat b { font-variant-numeric:tabular-nums; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%;
         background:var(--ok); margin-right:6px; }
  .dot.warn { background:#d29922; }
  main { padding:16px 20px 32px; }
  .views { display:grid; gap:14px;
           grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }
  figure { margin:0; background:var(--panel); border:1px solid var(--line);
           border-radius:8px; overflow:hidden; }
  figure img { width:100%; display:block; background:#000; }
  figcaption { padding:8px 12px; font-size:12px; color:var(--dim);
               border-top:1px solid var(--line); }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.09em;
       color:var(--dim); margin:26px 0 8px; }
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
  .empty { padding:18px; color:var(--dim); text-align:center; }
  .note { color:var(--dim); font-size:12px; margin-top:10px; max-width:74ch; }
</style>
</head>
<body>
<header>
  <h1>ASTRAVIGIL</h1>
  <span class="tag" id="src">—</span>
  <div class="stats">
    <span class="stat"><span class="dot" id="health"></span><span id="status">starting</span></span>
    <span class="stat"><span class="tag">fps</span> <b id="fps">—</b></span>
    <span class="stat"><span class="tag">proc</span> <b id="proc">—</b> ms</span>
    <span class="stat"><span class="tag">capture</span> <b id="cap">—</b> ms</span>
    <span class="stat"><span class="tag">headroom</span> <b id="head">—</b></span>
    <span class="stat"><span class="tag">tracks</span> <b id="tracks">—</b></span>
    <span class="stat"><span class="tag">frame</span> <b id="frame">—</b></span>
  </div>
</header>

<main>
  <div class="views">
    <figure>
      <img src="/stream/thermal" alt="thermal">
      <figcaption>Thermal — warm movers against cold sky. Detection runs here.</figcaption>
    </figure>
    <figure>
      <img src="/stream/optical" alt="optical">
      <figcaption>Optical — thermal boxes mapped through the homography.</figcaption>
    </figure>
    <figure>
      <img src="/stream/overlay" alt="overlay">
      <figcaption>Overlay — thermal warped into optical space. Hot regions should
        sit on things that are actually hot; if they drift, recalibrate.</figcaption>
    </figure>
  </div>

  <h2>Tracks</h2>
  <table>
    <thead><tr>
      <th>ID</th><th>Class</th><th>Conf</th><th>Peak °C</th><th>Hotspot</th>
      <th>Area px</th><th>Parts</th><th>Aspect</th><th>Flap</th><th>Straight</th><th>Frames</th>
    </tr></thead>
    <tbody id="rows"><tr><td class="empty" colspan="11">no detections</td></tr></tbody>
  </table>
  <p class="note"><b>Hotspot</b> is peak temperature above the object's <i>own</i> mean —
    the motor signature. Contrast against the sky is not used, because a 31&nbsp;°C bird
    against −5&nbsp;°C sky is exactly as bright as a drone. <b>Parts</b> is how many blobs
    were merged into one object: a resolved quad breaks into a body plus four motors, and
    reporting those separately would be five alarms for one aircraft.<br><br>
    <b>Flap</b> is the coefficient of variation of blob area over ~1.6 s —
    a flapping bird modulates its silhouette, a rigid airframe does not. It is the
    strongest single cue at range, where a drone and a bird are both just a few
    warm pixels. Confidences come from hand-tuned rules, not from a model fitted
    to real footage.</p>
</main>

<script>
const fmt = (v, d=2) => (v === undefined || v === null) ? "—" : Number(v).toFixed(d);

async function poll() {
  try {
    const r = await fetch("/api/state");
    const { detections, stats } = await r.json();

    document.getElementById("fps").textContent = fmt(stats.fps, 1);
    document.getElementById("proc").textContent = fmt(stats.proc_ms, 2);
    document.getElementById("cap").textContent = fmt(stats.capture_ms, 1);
    document.getElementById("head").textContent =
      stats.headroom ? stats.headroom + "x" : "—";
    document.getElementById("tracks").textContent = stats.tracks ?? "—";
    document.getElementById("frame").textContent = stats.frame ?? "—";
    document.getElementById("src").textContent =
      (stats.source ?? "") + (stats.calibrated ? " · calibrated" : " · uncalibrated");

    const dot = document.getElementById("health");
    const warm = stats.warmed_up;
    dot.className = "dot" + (warm ? "" : " warn");
    document.getElementById("status").textContent =
      warm ? "running" : "learning background";

    const body = document.getElementById("rows");
    if (!detections.length) {
      body.innerHTML = '<tr><td class="empty" colspan="11">no detections</td></tr>';
    } else {
      body.innerHTML = detections.map(d => `
        <tr>
          <td>#${d.track_id ?? "—"}</td>
          <td class="lbl ${d.label}">${d.label}</td>
          <td>${fmt(d.confidence)}</td>
          <td>${fmt(d.peak_c, 1)}</td>
          <td>${fmt(d.hotspot_c)}</td>
          <td>${d.area_px}</td>
          <td>${d.parts}</td>
          <td>${fmt(d.aspect)}</td>
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
