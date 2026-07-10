/* NBA edge terminal — mirrors the MLB-edges architecture:
   manifest.json dates -> loadSlate(date) -> renderSlate(rows), grade chips,
   deep-analysis rows, health drawer, themes, PWA. No dependencies. */
"use strict";

const $ = (id) => document.getElementById(id);
let MANIFEST = null;
let CURRENT = { date: null, games: [] };
let FILTER = null;
let MODE = "nba";               // "nba" (graded archive) | "sl" (summer league)

/* ---------- utils ---------- */
function gradeClass(g) {
  return "grade-" + g.replace("+", "p").replace(/^([AB])-$/, "$1-").replace("B-", "Bm").replace("A-", "A-");
}
function fmtML(v) { return v > 0 ? "+" + v : String(v); }
function setStatus(msg) { $("status").textContent = msg || ""; }

/* ---------- boot ----------
   Runs immediately if the DOM is already parsed (script sits at the end of
   <body>), falls back to DOMContentLoaded otherwise — immune to load-order
   races. Each init is isolated so one failure cannot kill the rest. */
async function boot() {
  const inits = [initTheme, initVisits, initDrawer, initHelp, initPicker, initChips, initXfade, initLeagueToggle, initRadarTip];
  for (const f of inits) { try { f(); } catch (e) { console.error("[nbaedge]", f.name, e); } }
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }
  try {
    MANIFEST = await fetch("data/manifest.json", { cache: "no-store" }).then(r => r.json());
  } catch (e) { setStatus("manifest unavailable"); return; }
  try { paintHero(); paintDateStrip(); paintGradeRecord(); loadHealth(); } catch (e) { console.error("[nbaedge]", e); }
  try {
    const pm = (MANIFEST.props || {});
    if (pm.mae && $("props-mae")) $("props-mae").textContent =
      Object.keys(pm.mae).map((k) =>
        k.toUpperCase() + " MAE " + pm.mae[k] + " (naive " + pm.baseline_mae[k] + ")"
      ).join(" · ");
  } catch (e) { /* cosmetic */ }
  $("built-at").textContent = " · data built " + (MANIFEST.built_at || "").slice(0, 10);
  const initial = MANIFEST.dates[0];
  $("datePicker").value = initial;
  loadSlate(initial);
}
if (document.readyState !== "loading") { boot(); }
else { document.addEventListener("DOMContentLoaded", boot); }

/* ---------- hero counters ---------- */
function paintHero() {
  const rec = MANIFEST.record || {};
  animate($("hero-acc"), (rec.acc || 0) * 100, "%");
  animate($("hero-agrade"), ((rec.grades || {})["A"] || {}).hit * 100 || 0, "%");
  animate($("hero-slates"), (MANIFEST.dates || []).length, "");
}
function animate(el, target, suffix) {
  const t0 = performance.now(), dur = 900;
  function tick(t) {
    const k = Math.min(1, (t - t0) / dur);
    el.textContent = (target * (0.2 + 0.8 * k)).toFixed(suffix === "%" ? 1 : 0) + suffix;
    if (k < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

/* ---------- date strip + picker ---------- */
function modeDates() {
  return MODE === "sl" ? (MANIFEST.sl_dates || []) : MANIFEST.dates;
}
function initLeagueToggle() {
  document.querySelectorAll(".lg-chip").forEach((c) => {
    c.onclick = () => {
      if (MODE === c.dataset.mode) return;
      MODE = c.dataset.mode;
      document.querySelectorAll(".lg-chip").forEach((x) =>
        x.classList.toggle("on", x.dataset.mode === MODE));
      $("sl-banner").style.display = MODE === "sl" ? "block" : "none";
      const slm = (MANIFEST || {}).sl_model;
      if (slm && $("sl-acc")) $("sl-acc").textContent =
        (slm.eval_2025.acc * 100).toFixed(1) + "% accuracy (n=" + slm.eval_2025.n + ")";
      $("nba-head").style.display = MODE === "sl" ? "none" : "";
      $("sl-head").style.display = MODE === "sl" ? "" : "none";
      $("quick-chips").style.display = MODE === "sl" ? "none" : "flex";
      FILTER = null;
      document.querySelectorAll(".q-chip").forEach((x) => x.classList.remove("on"));
      paintDateStrip();
      const dates = modeDates();
      // SL runs live in July: land on TODAY's slate when it exists,
      // otherwise the nearest date (list is newest-first).
      const today = new Date().toISOString().slice(0, 10);
      const first = dates.includes(today) ? today : dates[0];
      if (first) { $("datePicker").value = first; loadSlate(first); }
      else { renderSlate([]); setStatus("no slates published"); }
    };
  });
}
function paintDateStrip() {
  const strip = $("dateStrip");
  strip.querySelectorAll(".date-chip").forEach((c) => c.remove());
  // Full season timeline, chronological left -> right (past .. future).
  // The active chip is kept in view by followActiveChip() on every load,
  // so stepping into the past never scrolls the selection out of sight.
  [...modeDates()].reverse().forEach((d) => {
    const b = document.createElement("button");
    b.className = "date-chip"; b.textContent = d.slice(5); b.dataset.date = d;
    b.title = d;
    b.onclick = () => { $("datePicker").value = d; loadSlate(d); };
    strip.appendChild(b);
  });
}
function followActiveChip() {
  // Deterministic positioning: smooth scrollIntoView proved slow and
  // interruptible with a 200-chip strip; direct scrollLeft always lands.
  const chip = document.querySelector(".date-chip.active");
  if (!chip) return;
  const strip = chip.parentElement;
  strip.scrollLeft = Math.max(
    0, chip.offsetLeft - (strip.clientWidth - chip.offsetWidth) / 2);
}
function initPicker() {
  $("loadBtn").onclick = () => loadSlate($("datePicker").value);
  $("refreshBtn").onclick = () => loadSlate($("datePicker").value, { bust: true });
  $("datePicker").addEventListener("change", () => loadSlate($("datePicker").value));
  $("prevBtn").onclick = () => step("past");
  $("nextBtn").onclick = () => step("future");
}
function step(where) {
  // date lists are newest-first: past = index+1, future = index-1.
  const dates = modeDates();
  const i = dates.indexOf(CURRENT.date);
  const j = i + (where === "past" ? 1 : -1);
  if (j >= 0 && j < dates.length) {
    $("datePicker").value = dates[j];
    loadSlate(dates[j]);
  }
}

/* ---------- slate ---------- */
async function loadSlate(date, { bust = false } = {}) {
  if (!date) return;
  setStatus("loading " + date + "…");
  let payload = null;
  try {
    const prefix = MODE === "sl" ? "data/sl_" : "data/picks_";
    const url = prefix + date + ".json" + (bust ? "?b=" + Date.now() : "");
    const r = await fetch(url, { cache: bust ? "no-store" : "default" });
    if (r.ok) payload = await r.json();
  } catch (e) { /* handled below */ }
  if (!payload) {
    CURRENT = { date, games: [] };
    renderSlate([]);
    setStatus("no slate published for " + date);
    return;
  }
  CURRENT = { date, games: payload.games };
  document.querySelectorAll(".date-chip").forEach(c =>
    c.classList.toggle("active", c.dataset.date === date));
  followActiveChip();
  const n = payload.games.length;
  if (MODE === "sl") {
    renderSlateSL(payload.games);
    const fin = payload.games.filter(g => g.is_final).length;
    setStatus(n + " games · " + fin + " final · " + date);
    startSlLive(date);
  } else {
    stopSlLive();
    renderSlate(applyFilter(payload.games));
    const hits = payload.games.filter(g => g.result && g.result.pick_correct).length;
    setStatus(n + " games · model " + hits + "/" + n + " on " + date);
  }
}

/* ---------- SL live refresh (client-side ESPN merge) ----------
   The static sl_*.json is a snapshot from export time; live scores are
   merged in-browser straight from ESPN's CORS-open scoreboard API and
   re-polled every 60s while the tab is visible and games are in window. */
let SL_TIMER = null;
const SL_LEAGUES = ["nba-summer-las-vegas", "nba-summer-utah", "nba-summer-california"];

function stopSlLive() { if (SL_TIMER) { clearInterval(SL_TIMER); SL_TIMER = null; } }

function startSlLive(date) {
  stopSlLive();
  const games = CURRENT.games || [];
  if (!games.length || games.every(g => g.is_final)) return;
  // Any non-final snapshot gets ONE immediate merge — this heals stale
  // past-date exports (frozen LIVE rows) the moment they're viewed.
  refreshSlLive(date);
  const tips = games.map(g => +new Date(g.tip_utc)).filter(Number.isFinite);
  const now = Date.now();
  // recurring poll only around the game window: 1h before first tip -> 5h after last
  if (!tips.length || now < Math.min(...tips) - 3600e3
      || now > Math.max(...tips) + 5 * 3600e3) return;
  const tick = () => { if (!document.hidden && MODE === "sl") refreshSlLive(date); };
  SL_TIMER = setInterval(tick, 60_000);
}

async function refreshSlLive(date) {
  const ymd = date.replaceAll("-", "");
  const live = {};
  await Promise.all(SL_LEAGUES.map(async (lg) => {
    try {
      const r = await fetch("https://site.api.espn.com/apis/site/v2/sports/basketball/"
        + lg + "/scoreboard?dates=" + ymd, { cache: "no-store" });
      if (!r.ok) return;
      const j = await r.json();
      (j.events || []).forEach((ev) => {
        const comp = (ev.competitions || [{}])[0];
        const sides = {};
        (comp.competitors || []).forEach((c) => { sides[c.homeAway] = c; });
        if (!sides.home || !sides.away) return;
        live[ev.id] = {
          state: ev.status.type.name,
          is_final: !!ev.status.type.completed,
          home_score: parseInt(sides.home.score || 0, 10),
          away_score: parseInt(sides.away.score || 0, 10),
        };
      });
    } catch (e) { /* offline: keep snapshot */ }
  }));
  if (MODE !== "sl" || CURRENT.date !== date) return;
  let changed = false;
  CURRENT.games.forEach((g) => {
    const u = live[g.espn_id];
    if (!u) return;
    if (u.state !== g.state || u.home_score !== g.home_score
        || u.away_score !== g.away_score || u.is_final !== g.is_final) {
      changed = true;
      Object.assign(g, u);
      if (g.is_final && g.pick && g.pick_correct == null) {
        const winner = g.home_score > g.away_score ? g.home_abbr : g.away_abbr;
        g.pick_correct = g.pick === winner ? 1 : 0;
      }
    }
  });
  const fin = CURRENT.games.filter(g => g.is_final).length;
  setStatus(CURRENT.games.length + " games · " + fin + " final · " + date
            + (CURRENT.games.every(g => g.is_final) ? "" : " · live ⟳60s"));
  if (CURRENT.games.every(g => g.is_final)) stopSlLive();
  if (changed) renderSlateSL(CURRENT.games);
}

function renderSlateSL(games) {
  $("card-grid").style.display = "none";
  const tb = $("slate-body");
  tb.innerHTML = "";
  $("slate").style.display = games.length ? "table" : "none";
  $("empty").style.display = games.length ? "none" : "block";
  games.forEach((g) => {
    const tr = document.createElement("tr");
    let tip = "—";
    if (g.tip_utc) {
      try { tip = new Date(g.tip_utc).toLocaleTimeString([],
        { hour: "numeric", minute: "2-digit" }); }
      catch (e) { tip = g.tip_utc.slice(11, 16); }
    }
    const stateTxt = g.is_final ? "FINAL"
      : (g.state === "STATUS_IN_PROGRESS" ? "LIVE" : "Scheduled");
    const score = (g.is_final || g.state === "STATUS_IN_PROGRESS")
      ? '<span class="mono sl-final">' + g.away_score + " – " + g.home_score + "</span>"
      : '<span class="mono sl-sched">—</span>';
    const pickCell = g.pick
      ? '<b class="mono">' + g.pick + '</b> <span class="mono" style="font-size:.64rem;color:var(--muted)">' + (g.tier || "") + "</span>"
      : '<span class="sl-sched">—</span>';
    const probCell = g.p_pick != null
      ? '<span class="mono">' + (g.p_pick * 100).toFixed(1) + "%</span>"
      : '<span class="sl-sched">—</span>';
    let verdict = "";
    if (g.is_final && g.pick_correct != null) {
      verdict = g.pick_correct ? ' <span class="res-w">✓</span>'
                               : ' <span class="res-l">✗</span>';
    }
    tr.innerHTML =
      '<td class="mono" style="color:var(--muted)">' + g.league + "</td>" +
      "<td><b>" + (g.away_abbr || g.away) + "</b> @ <b>" + (g.home_abbr || g.home) + "</b></td>" +
      '<td class="mono">' + tip + "</td>" +
      "<td>" + pickCell + "</td>" +
      "<td>" + probCell + "</td>" +
      "<td>" + (g.is_final ? '<span class="res-w">' + stateTxt + "</span>"
                : stateTxt === "LIVE" ? '<span style="color:var(--accent)">LIVE</span>'
                : '<span class="sl-sched">' + stateTxt + "</span>") + "</td>" +
      "<td>" + score + verdict + "</td>";
    tb.appendChild(tr);
  });
}

function applyFilter(games) {
  if (!FILTER) return games;
  if (FILTER === "agrade") return games.filter(g => g.grade === "A" || g.grade === "A-");
  if (FILTER === "best") {
    const s = [...games].sort((a, b) => b.pick_prob - a.pick_prob);
    return s.slice(0, 1);
  }
  if (FILTER === "upset") return games.filter(g => g.pick === g.away_abbr); // road wins are the minority call
  if (FILTER === "wrong") return games.filter(g => g.result && !g.result.pick_correct);
  return games;
}

function renderSlate(games) {
  // NBA mode renders the card grid; the table is Summer League's.
  $("slate").style.display = "none";
  const grid = $("card-grid");
  grid.innerHTML = "";
  grid.style.display = games.length ? "grid" : "none";
  $("empty").style.display = games.length ? "none" : "block";
  const rm = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

  games.forEach((g, i) => {
    const card = document.createElement("div");
    card.className = "game-card glass" + (g.grade === "A" || g.grade === "A-" ? " alit" : "");
    const p = g.pick_prob;
    const R = 26, C = 2 * Math.PI * R;
    const res = g.result;
    const resTxt = res
      ? (res.pick_correct ? '<span class="res-w">✓ HIT</span>' : '<span class="res-l">✗ MISS</span>')
        + ' <span style="color:var(--muted)">home ' + (res.margin_home > 0 ? "W" : "L")
        + " by " + Math.abs(res.margin_home) + "</span>"
      : '<span class="res-pend">pending</span>';
    // underdog edge: confident road call gets the pulse
    const pulse = (g.pick === g.away_abbr && p >= 0.55 && !rm) ? " pulse" : "";
    card.innerHTML =
      '<div class="gc-top"><div><div class="gc-matchup">' + g.away_abbr
      + ' @ ' + g.home_abbr
      + (g.season_type === "Playoffs" ? ' <span class="po">PO</span>' : "") + "</div>"
      + '<div class="gc-sub">' + g.away + " at " + g.home + "</div></div>"
      + '<div class="gc-dial' + (p >= 0.6 ? " hot" : "") + '">'
      + '<svg width="64" height="64" viewBox="0 0 64 64">'
      + '<circle class="ring" cx="32" cy="32" r="' + R + '"/>'
      + '<circle class="arc" cx="32" cy="32" r="' + R + '" stroke-dasharray="' + C.toFixed(1)
      + '" stroke-dashoffset="' + C.toFixed(1) + '"/></svg>'
      + '<span class="pct">0%</span></div></div>'
      + '<div class="gc-verdict"><span class="verdict-tag' + pulse + '">' + g.pick + "</span>"
      + '<span class="grade ' + gradeClass(g.grade) + '">' + g.grade + "</span>"
      + '<span class="gc-line">' + fmtML(g.fair_ml_home) + " / " + fmtML(g.fair_ml_away)
      + " · " + (g.pred_margin_home > 0 ? "H" : "A") + " −"
      + Math.abs(g.pred_margin_home).toFixed(1) + "</span>"
      + '<span class="gc-res">' + resTxt + "</span></div>"
      + '<div class="card-deep"><div class="deep-inner">' + deepPanel(g)
      + '<div class="props-box" data-gid="' + g.game_id + '">'
      + '<div class="deep-h">Player telemetry — projected vs actual</div>'
      + '<div class="props-slot mono" style="color:var(--muted);font-size:.75rem">…</div>'
      + "</div></div></div>";

    card.addEventListener("click", (e) => {
      if (e.target.closest(".card-deep")) return;   // interacting with content
      const opening = !card.classList.contains("open");
      card.classList.toggle("open");
      if (opening) {
        const cascade = () => card.querySelectorAll(".wf-seg").forEach((s, j) => {
          s.classList.remove("on");
          setTimeout(() => s.classList.add("on"), rm ? 0 : 90 + j * 120);
        });
        cascade();
        const wf = card.querySelector("details.wf");
        if (wf && !wf.dataset.bound) {
          wf.dataset.bound = "1";
          wf.addEventListener("toggle", () => { if (wf.open) cascade(); });
        }
        loadProps(CURRENT.date, card);
      }
    });
    grid.appendChild(card);

    // dial spin-up (next frame so the transition engages)
    const arc = card.querySelector(".arc");
    const pct = card.querySelector(".pct");
    const target = C * (1 - p);
    if (rm) { arc.style.strokeDashoffset = target; pct.textContent = (p * 100).toFixed(1) + "%"; }
    else {
      requestAnimationFrame(() => requestAnimationFrame(() => {
        arc.style.strokeDashoffset = target;
      }));
      const t0 = performance.now();
      (function tick(t) {
        const k = Math.min(1, (t - t0) / 950);
        pct.textContent = (p * 100 * (k < 1 ? k : 1)).toFixed(1) + "%";
        if (k < 1) requestAnimationFrame(tick);
      })(t0);
    }
  });
}

function deepPanel(g) {
  // Horizontal waterfall: segments build the cumulative top-factor total.
  const fs = g.factors || [];
  const maxAbs = Math.max(2, ...fs.map(f => Math.abs(f.impact_pp)));
  let run = 0;
  const span = maxAbs * 2.6;                       // px-free: percent scale
  const segs = fs.map((f) => {
    const a = run, b = run + f.impact_pp;
    run = b;
    const lo = Math.min(a, b), hi = Math.max(a, b);
    const left = 50 + (lo / span) * 100, width = ((hi - lo) / span) * 100;
    return '<div class="wf-row"><span class="wf-label">' + f.label + "</span>"
      + '<span class="wf-track"><span class="wf-zero" style="left:50%"></span>'
      + '<span class="wf-seg ' + (f.impact_pp >= 0 ? "pos" : "neg")
      + '" style="left:' + left.toFixed(2) + "%;width:" + Math.max(1.2, width).toFixed(2) + '%"></span></span>'
      + '<span class="wf-val" style="color:' + (f.impact_pp >= 0 ? "var(--green)" : "var(--red)") + '">'
      + (f.impact_pp >= 0 ? "+" : "") + f.impact_pp.toFixed(1) + " pp</span></div>";
  }).join("");
  const sum = fs.reduce((s, f) => s + f.impact_pp, 0);
  // progressive disclosure: open by default on desktop, tap-to-expand on mobile
  const openAttr = (window.matchMedia && matchMedia("(min-width: 760px)").matches)
    ? " open" : "";
  return (
    "<details class='dd wf'" + openAttr + "><summary>Why — top-factor waterfall (SHAP, → home)</summary>"
    + segs
    + "<div class='wf-row wf-sum'><span class='wf-label'>Σ top factors</span><span></span>"
    + "<span class='wf-val' style='color:" + (sum >= 0 ? "var(--green)" : "var(--red)") + "'>"
    + (sum >= 0 ? "+" : "") + sum.toFixed(1) + " pp</span></div></details>"
    + "<details class='dd deep-meta'" + openAttr + "><summary>Game meta</summary>"
    + "<div>Model home win prob: <b>" + (g.p_home * 100).toFixed(1) + "%</b></div>"
    + "<div>Fair line: <b>" + fmtML(g.fair_ml_home) + " home / " + fmtML(g.fair_ml_away) + " away</b></div>"
    + "<div>Projected margin: <b>home " + (g.pred_margin_home > 0 ? "+" : "") + g.pred_margin_home.toFixed(1) + "</b></div>"
    + "<div>Tier: <b>" + g.tier + "</b> · " + g.season_type + "</div>"
    + "<div class='mono' style='font-size:.68rem'>id " + g.game_id + "</div></details>"
  );
}

/* ---------- player props (lazy per-date) ---------- *//* ---------- player props (lazy per-date) ---------- */
const PROPS_CACHE = {};
const RADAR_AXES = [["pts", "PTS", 40], ["reb", "REB", 16], ["ast", "AST", 12],
                    ["fg3m", "3PM", 6], ["stl", "STL", 4], ["blk", "BLK", 4]];

function radarSVG(r) {
  // hexagonal radar, 120x120 viewBox, proj polygon (accent) vs actual (green)
  const cx = 60, cy = 62, R = 44, n = RADAR_AXES.length;
  const angle = (i) => -Math.PI / 2 + (i * 2 * Math.PI) / n;
  const px = (i, v, max) => {
    const k = Math.min(1, Math.max(0, v / max));
    return [(cx + Math.cos(angle(i)) * R * k).toFixed(1),
            (cy + Math.sin(angle(i)) * R * k).toFixed(1)];
  };
  const ring = (k) => RADAR_AXES.map((_, i) =>
    [(cx + Math.cos(angle(i)) * R * k).toFixed(1),
     (cy + Math.sin(angle(i)) * R * k).toFixed(1)].join(",")).join(" ");
  const poly = (get) => RADAR_AXES.map(([k, , max], i) => px(i, get(k), max).join(",")).join(" ");
  const labels = RADAR_AXES.map(([, name], i) => {
    const lx = cx + Math.cos(angle(i)) * (R + 9), ly = cy + Math.sin(angle(i)) * (R + 9);
    return '<text class="r-lab" x="' + lx.toFixed(1) + '" y="' + ly.toFixed(1)
      + '" text-anchor="middle" dominant-baseline="middle">' + name + "</text>";
  }).join("");
  const nodes = RADAR_AXES.map(([k, name, max], i) => {
    const [nx, ny] = px(i, r.actual[k], max);
    const d = (r.actual[k] - r.proj[k]);
    return '<circle class="r-node" cx="' + nx + '" cy="' + ny + '" r="7" '
      + 'data-tip="<b>' + r.player.split(" ").pop() + " · " + name + "</b><br>"
      + "PROJ: " + r.proj[k].toFixed(1) + " | ACT: " + r.actual[k]
      + " | Δ " + (d >= 0 ? "+" : "") + d.toFixed(1) + '"/>';
  }).join("");
  return '<svg width="138" height="132" viewBox="0 0 120 124">'
    + [0.33, 0.66, 1].map(k => '<polygon class="r-grid" points="' + ring(k) + '"/>').join("")
    + RADAR_AXES.map((_, i) => { const [ex, ey] = px(i, 1, 1);
        return '<line class="r-axis" x1="' + cx + '" y1="' + cy + '" x2="' + ex + '" y2="' + ey + '"/>'; }).join("")
    + '<polygon class="r-proj" points="' + poly(k => r.proj[k]) + '"/>'
    + '<polygon class="r-act" points="' + poly(k => r.actual[k]) + '"/>'
    + labels + nodes + "</svg>";
}

function initRadarTip() {
  const tip = $("radar-tip");
  document.addEventListener("mouseover", (e) => {
    const n = e.target.closest && e.target.closest(".r-node");
    if (!n) { tip.style.display = "none"; return; }
    tip.innerHTML = n.dataset.tip;
    tip.style.display = "block";
  });
  document.addEventListener("mousemove", (e) => {
    if (tip.style.display !== "block") return;
    const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
    const y = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8);
    tip.style.transform = "translate3d(" + x + "px," + y + "px,0)";
    tip.style.left = "0"; tip.style.top = "0";
  });
}

async function loadProps(date, cardEl) {
  const box = cardEl.querySelector(".props-box");
  const slot = cardEl.querySelector(".props-slot");
  if (!box || box.dataset.loaded) return;
  let payload = PROPS_CACHE[date];
  if (!payload) {
    slot.textContent = "loading telemetry…";
    try {
      const r = await fetch("data/props_" + date + ".json");
      payload = r.ok ? await r.json() : null;
    } catch (e) { payload = null; }
    PROPS_CACHE[date] = payload;
  }
  const rows = payload && payload.games && payload.games[box.dataset.gid];
  if (!rows || !rows.length) {
    slot.textContent = "no player projections for this game";
    box.dataset.loaded = "1";
    return;
  }
  box.dataset.loaded = "1";
  box.dataset.date = date;
  renderPropsInto(box, rows);
}

function propsView() {
  const saved = lsGet("nbaedge-props-view");
  if (saved) return saved;
  // default: data table on small screens (labels stay legible), radar on desktop
  return (window.matchMedia && matchMedia("(max-width: 640px)").matches)
    ? "table" : "radar";
}

function renderPropsInto(box, rows) {
  const methods = ((MANIFEST || {}).props || {}).method || {};
  const dagger = Object.values(methods).includes("baseline_r7")
    ? " · † BLK proj = trailing-7 avg (model gate)" : "";
  const view = propsView();
  const head =
    '<div class="props-head"><div class="deep-h" style="margin:0">Player telemetry — projected vs actual</div>'
    + '<button type="button" class="view-toggle">' + (view === "radar" ? "☰ table" : "⬡ radar")
    + "</button></div>";
  let bodyHTML;
  if (view === "radar") {
    const cells = rows.slice(0, 12).map((r) =>
      '<div class="radar-cell"><div class="radar-name">' + r.player + "</div>"
      + '<div class="radar-team">' + r.abbr + " · " + r.mp.toFixed(0) + " min</div>"
      + radarSVG(r) + "</div>").join("");
    bodyHTML = '<div class="radar-grid">' + cells + "</div>"
      + '<div class="radar-legend"><span><i style="background:var(--accent)"></i>projected</span>'
      + '<span><i style="background:var(--green)"></i>actual</span>'
      + '<span>hover nodes for exact deltas' + dagger + "</span></div>";
  } else {
    const STATS = RADAR_AXES.map(([k, n]) => [k, n]);
    const label = (k, n) => n + (methods[k] === "baseline_r7" ? " †" : "");
    const byTeam = {};
    rows.forEach((r) => (byTeam[r.abbr] = byTeam[r.abbr] || []).push(r));
    const nCols = 1 + STATS.length * 2;
    const trow = (r) => "<tr><td class='pn'>" + r.player + "</td>"
      + STATS.map(([k]) => "<td class='props-proj gs'>" + r.proj[k].toFixed(1)
        + "</td><td class='props-act'>" + r.actual[k] + "</td>").join("") + "</tr>";
    const body = Object.keys(byTeam).map((team) =>
      "<tr class='props-team'><td colspan='" + nCols + "'>" + team + "</td></tr>"
      + byTeam[team].sort((a, b) => b.proj.pts - a.proj.pts).slice(0, 5).map(trow).join("")).join("");
    bodyHTML = '<div class="props-scroll"><table class="props-table"><thead>'
      + "<tr><th class='ph' rowspan='2'>Player</th>"
      + STATS.map(([k, n]) => "<th class='gh gs' colspan='2'>" + label(k, n) + "</th>").join("")
      + "</tr><tr>" + STATS.map(() => "<th class='gs'>proj</th><th>act</th>").join("")
      + "</tr></thead><tbody>" + body + "</tbody></table></div>"
      + (dagger ? '<div class="props-note">' + dagger.slice(3) + "</div>" : "");
  }
  box.innerHTML = head + bodyHTML;
  box.querySelector(".view-toggle").addEventListener("click", (e) => {
    e.stopPropagation();
    lsSet("nbaedge-props-view", propsView() === "radar" ? "table" : "radar");
    renderPropsInto(box, rows);
  });
}

/* ---------- quick chips ---------- */
function initChips() {
  document.querySelectorAll(".q-chip").forEach(c => {
    c.onclick = () => {
      const f = c.dataset.filter;
      FILTER = (FILTER === f) ? null : f;
      document.querySelectorAll(".q-chip").forEach(x =>
        x.classList.toggle("on", x.dataset.filter === FILTER));
      renderSlate(applyFilter(CURRENT.games));
    };
  });
}

/* ---------- drawer + health ---------- */
function initDrawer() {
  const open = () => { $("side-drawer").classList.add("open"); $("drawer-backdrop").classList.add("show"); };
  const close = () => { $("side-drawer").classList.remove("open"); $("drawer-backdrop").classList.remove("show"); };
  $("drawer-btn").onclick = open;
  $("drawer-close").onclick = close;
  $("drawer-backdrop").onclick = close;
}
async function loadHealth() {
  let h = null;
  try { h = await fetch("data/health.json", { cache: "no-store" }).then(r => r.json()); }
  catch (e) { /* leave placeholder */ }
  const body = $("health-card-body");
  if (!h) { body.textContent = "health snapshot unavailable"; return; }
  body.innerHTML =
    '<div class="health-row"><span class="health-pill hp-' + h.overall + '"></span>' +
    "<b>overall: " + h.overall.toUpperCase() + "</b> · checked " + h.checked_at.slice(0, 16).replace("T", " ") + "</div>" +
    h.checks.map(c =>
      '<div class="health-row"><div class="hr-name">' + c.name + "</div>" +
      '<span class="health-pill hp-' + c.severity + '"></span>' + c.message + "</div>").join("");
}
function paintGradeRecord() {
  const g = ((MANIFEST || {}).record || {}).grades || {};
  $("grade-record").innerHTML = Object.keys(g).map(k =>
    '<div class="gr-row"><span>' + k + " (" + g[k].n + ")</span><span>" +
    (g[k].hit * 100).toFixed(1) + "%</span></div>").join("") || "…";
}

/* ---------- theme + visits + help ---------- */
function initTheme() {
  const saved = lsGet("nbaedge-theme") || "tactical";
  document.body.dataset.theme = saved;
  $("theme-label").textContent = saved.toUpperCase();
  $("theme-toggle").onclick = () => {
    const next = document.body.dataset.theme === "tactical" ? "courtside" : "tactical";
    document.body.dataset.theme = next;
    $("theme-label").textContent = next.toUpperCase();
    lsSet("nbaedge-theme", next);
  };
}
/* Unique-visit counter via counterapi.dev (anonymous, no fingerprinting).
   Dedup: one increment per browser, ever (localStorage flag) — returning
   browsers are read-only and NEVER fall back to /up (no double counting).
   Bot gating: crawlers without JS never reach this; webdriver/headless and
   bot UAs are read-only; prerendered pages (speculation rules) defer until
   real activation + tab visibility, so background prerenders don't count. */
/* Storage on gozorp.github.io is SHARED across both terminals and can be
   full (quota). Storage is an optimization here, never a dependency: all
   access goes through safe wrappers, the pill renders before any write is
   attempted, and the visited-flag falls back to a scoped cookie. */
function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); return true; } catch (e) { return false; } }
function markVisited() {
  if (!lsSet("nbaedge_visited_v1", "1")) {
    try { document.cookie = "nbaedge_v=1; max-age=31536000; path=/nba-edge/; SameSite=Lax"; } catch (e) {}
  }
}
function hasVisited() {
  if (lsGet("nbaedge_visited_v1")) return true;
  try { return document.cookie.indexOf("nbaedge_v=1") !== -1; } catch (e) { return false; }
}

function isLikelyBot() {
  if (navigator.webdriver === true) return true;
  return /bot|crawl|spider|slurp|headless|lighthouse|prerender|preview|fetch|monitor|scan/i
    .test(navigator.userAgent || "");
}
function whenTrulyVisible(fn) {
  const arm = () => {
    if (document.visibilityState === "visible") { fn(); return; }
    const h = () => {
      if (document.visibilityState === "visible") {
        document.removeEventListener("visibilitychange", h); fn();
      }
    };
    document.addEventListener("visibilitychange", h);
  };
  if (document.prerendering) {
    document.addEventListener("prerenderingchange", arm, { once: true });
  } else { arm(); }
}
function initVisits() {
  const el = $("visitText");
  const COUNT_KEY = "nbaedge_last_count_v1";
  const BASE = "https://api.counterapi.dev/v1/gozorp-nba-edge/unique_visits";
  const cached = parseInt(lsGet(COUNT_KEY) || "0", 10);
  if (cached > 0) el.textContent = cached.toLocaleString()
    + (cached === 1 ? " unique visit" : " unique visits");
  const show = (count) => {
    el.textContent = count.toLocaleString()
      + (count === 1 ? " unique visit" : " unique visits");  // text FIRST
    lsSet(COUNT_KEY, String(count));                         // storage best-effort
  };
  const hit = async (url) => {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) return null;
      const j = await r.json();
      return (typeof j.count === "number") ? j.count : null;
    } catch (e) { return null; }
  };
  // Display is read-only and immediate — even in hidden/prerendered tabs.
  hit(BASE + "/").then((c) => {
    if (c != null) show(c);
    else if (cached <= 0) el.textContent = "visits offline";
  });
  // The INCREMENT stays gated: real visibility + non-bot + first visit only.
  whenTrulyVisible(async () => {
    if (hasVisited() || isLikelyBot()) return;
    const c = await hit(BASE + "/up");
    if (c != null) { markVisited(); show(c); }
  });
}
function initXfade() {
  const vt = window.CSS && CSS.supports && CSS.supports("(view-transition-name: x)");
  const rm = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const a = $("mlb-return");
  if (a && !vt && !rm) {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      document.body.classList.add("xfade-leave");
      setTimeout(() => { location.href = a.href; }, 170);
    });
  }
  window.addEventListener("pageshow", () =>
    document.body.classList.remove("xfade-leave"));
}

function initHelp() {
  $("help-btn").onclick = () => $("help-backdrop").classList.add("show");
  $("help-close").onclick = () => $("help-backdrop").classList.remove("show");
  $("help-backdrop").addEventListener("click", (e) => {
    if (e.target === $("help-backdrop")) $("help-backdrop").classList.remove("show");
  });
}
