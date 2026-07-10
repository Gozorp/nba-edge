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
  const inits = [initTheme, initVisits, initDrawer, initHelp, initPicker, initChips, initXfade, initLeagueToggle];
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
      "PTS MAE " + pm.mae.pts + " (naive " + pm.baseline_mae.pts + "), REB "
      + pm.mae.reb + " (" + pm.baseline_mae.reb + "), AST "
      + pm.mae.ast + " (" + pm.baseline_mae.ast + ")";
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
  } else {
    renderSlate(applyFilter(payload.games));
    const hits = payload.games.filter(g => g.result && g.result.pick_correct).length;
    setStatus(n + " games · model " + hits + "/" + n + " on " + date);
  }
}

function renderSlateSL(games) {
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
  const tb = $("slate-body");
  tb.innerHTML = "";
  $("slate").style.display = games.length ? "table" : "none";
  $("empty").style.display = games.length ? "none" : "block";
  games.forEach((g, i) => {
    const tr = document.createElement("tr");
    tr.className = "row-clickable";
    const pickProbPct = (g.pick_prob * 100).toFixed(1);
    const res = g.result;
    const finalScore = res ? (res.margin_home > 0 ? "W " : "L ") + "by " + Math.abs(res.margin_home) : "—";
    const resCell = res
      ? (res.pick_correct ? '<span class="res-w">✓ HIT</span>' : '<span class="res-l">✗ MISS</span>')
        + ' <span class="mono" style="color:var(--muted);font-size:.72rem">(home ' + finalScore + ")</span>"
      : '<span class="res-pend">pending</span>';
    tr.innerHTML =
      '<td><span class="grade ' + gradeClass(g.grade) + '">' + g.grade + "</span></td>" +
      "<td><b>" + g.away_abbr + "</b> @ <b>" + g.home_abbr + "</b>" +
      (g.season_type === "Playoffs" ? ' <span class="mono" style="color:var(--accent);font-size:.66rem">PO</span>' : "") + "</td>" +
      '<td class="mono"><b>' + g.pick + "</b></td>" +
      '<td class="prob-cell"><span class="mono">' + pickProbPct + '%</span>' +
      '<div class="prob-bar"><i style="width:' + pickProbPct + '%"></i></div></td>' +
      '<td class="mono">' + fmtML(g.fair_ml_home) + " / " + fmtML(g.fair_ml_away) + "</td>" +
      '<td class="mono">' + (g.pred_margin_home > 0 ? "H −" : "A −") + Math.abs(g.pred_margin_home).toFixed(1) + "</td>" +
      "<td>" + resCell + "</td>";
    const deep = document.createElement("tr");
    deep.className = "deep";
    deep.innerHTML = '<td colspan="7"><div class="deep-inner">' + deepPanel(g)
      + '<div class="props-box" data-gid="' + g.game_id + '">'
      + '<div class="deep-h">Player props — projection vs actual</div>'
      + '<div class="props-slot mono" style="color:var(--muted);font-size:.75rem">click to load…</div>'
      + "</div></div></td>";
    tr.onclick = () => {
      deep.classList.toggle("open");
      if (deep.classList.contains("open")) loadProps(CURRENT.date, deep);
    };
    tb.appendChild(tr); tb.appendChild(deep);
  });
}

function deepPanel(g) {
  const maxImpact = Math.max(4, ...g.factors.map(f => Math.abs(f.impact_pp)));
  const bars = g.factors.map(f => {
    const w = Math.min(50, Math.abs(f.impact_pp) / maxImpact * 50);
    const cls = f.impact_pp >= 0 ? "pos" : "neg";
    return '<div class="factor"><span class="fl">' + f.label + "</span>" +
      '<span class="fbar"><i class="' + cls + '" style="width:' + w + '%"></i></span>' +
      '<span class="fv">' + (f.impact_pp >= 0 ? "+" : "") + f.impact_pp.toFixed(1) + " pp</span></div>";
  }).join("");
  return (
    "<div><div class='deep-h'>Why — top factors (SHAP, → home)</div>" + bars + "</div>" +
    "<div class='deep-meta'><div class='deep-h'>Game meta</div>" +
    "<div>Matchup: <b>" + g.away + " @ " + g.home + "</b></div>" +
    "<div>Model home win prob: <b>" + (g.p_home * 100).toFixed(1) + "%</b></div>" +
    "<div>Fair line: <b>" + fmtML(g.fair_ml_home) + " home / " + fmtML(g.fair_ml_away) + " away</b></div>" +
    "<div>Projected margin: <b>home " + (g.pred_margin_home > 0 ? "+" : "") + g.pred_margin_home.toFixed(1) + "</b></div>" +
    "<div>Tier: <b>" + g.tier + "</b> · " + g.season_type + "</div>" +
    "<div class='mono' style='font-size:.68rem'>id " + g.game_id + "</div></div>"
  );
}

/* ---------- player props (lazy per-date) ---------- */
const PROPS_CACHE = {};
async function loadProps(date, deepRow) {
  const box = deepRow.querySelector(".props-box");
  const slot = deepRow.querySelector(".props-slot");
  if (!box || box.dataset.loaded) return;
  let payload = PROPS_CACHE[date];
  if (!payload) {
    slot.textContent = "loading projections…";
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
  // Organize: block by team, sort by projected points inside each block,
  // and group stat pairs under a two-tier header (proj | act per stat).
  const byTeam = {};
  rows.forEach((r) => (byTeam[r.abbr] = byTeam[r.abbr] || []).push(r));
  const playerRow = (r) =>
    "<tr><td class='pn'>" + r.player + "</td>"
    + "<td class='props-proj gs'>" + r.proj.pts.toFixed(1) + "</td><td class='props-act'>" + r.actual.pts + "</td>"
    + "<td class='props-proj gs'>" + r.proj.reb.toFixed(1) + "</td><td class='props-act'>" + r.actual.reb + "</td>"
    + "<td class='props-proj gs'>" + r.proj.ast.toFixed(1) + "</td><td class='props-act'>" + r.actual.ast + "</td></tr>";
  const body = Object.keys(byTeam).map((team) =>
    "<tr class='props-team'><td colspan='7'>" + team + "</td></tr>"
    + byTeam[team]
        .sort((a, b) => b.proj.pts - a.proj.pts)
        .slice(0, 5)
        .map(playerRow).join("")
  ).join("");
  slot.outerHTML =
    "<table class='props-table'><thead>"
    + "<tr><th class='ph' rowspan='2'>Player</th>"
    + "<th class='gh gs' colspan='2'>Points</th>"
    + "<th class='gh gs' colspan='2'>Rebounds</th>"
    + "<th class='gh gs' colspan='2'>Assists</th></tr>"
    + "<tr><th class='gs'>proj</th><th>act</th>"
    + "<th class='gs'>proj</th><th>act</th>"
    + "<th class='gs'>proj</th><th>act</th></tr>"
    + "</thead><tbody>" + body + "</tbody></table>";
  box.dataset.loaded = "1";
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
