/* NBA edge terminal — mirrors the MLB-edges architecture:
   manifest.json dates -> loadSlate(date) -> renderSlate(rows), grade chips,
   deep-analysis rows, health drawer, themes, PWA. No dependencies. */
"use strict";

const $ = (id) => document.getElementById(id);
let MANIFEST = null;
let CURRENT = { date: null, games: [] };
let FILTER = null;

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
  const inits = [initTheme, initVisits, initDrawer, initHelp, initPicker, initChips, initXfade];
  for (const f of inits) { try { f(); } catch (e) { console.error("[nbaedge]", f.name, e); } }
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }
  try {
    MANIFEST = await fetch("data/manifest.json", { cache: "no-store" }).then(r => r.json());
  } catch (e) { setStatus("manifest unavailable"); return; }
  try { paintHero(); paintDateStrip(); paintGradeRecord(); loadHealth(); } catch (e) { console.error("[nbaedge]", e); }
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
function paintDateStrip() {
  const strip = $("dateStrip");
  MANIFEST.dates.slice(0, 14).forEach((d) => {
    const b = document.createElement("button");
    b.className = "date-chip"; b.textContent = d.slice(5); b.dataset.date = d;
    b.onclick = () => { $("datePicker").value = d; loadSlate(d); };
    strip.appendChild(b);
  });
}
function initPicker() {
  $("loadBtn").onclick = () => loadSlate($("datePicker").value);
  $("refreshBtn").onclick = () => loadSlate($("datePicker").value, { bust: true });
  $("datePicker").addEventListener("change", () => loadSlate($("datePicker").value));
  $("prevBtn").onclick = () => step(1);
  $("nextBtn").onclick = () => step(-1);
}
function step(dir) {
  const i = MANIFEST.dates.indexOf(CURRENT.date);
  const j = i + dir;
  if (j >= 0 && j < MANIFEST.dates.length) {
    $("datePicker").value = MANIFEST.dates[j];
    loadSlate(MANIFEST.dates[j]);
  }
}

/* ---------- slate ---------- */
async function loadSlate(date, { bust = false } = {}) {
  if (!date) return;
  setStatus("loading " + date + "…");
  let payload = null;
  try {
    const url = "data/picks_" + date + ".json" + (bust ? "?b=" + Date.now() : "");
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
  renderSlate(applyFilter(payload.games));
  const n = payload.games.length;
  const hits = payload.games.filter(g => g.result && g.result.pick_correct).length;
  setStatus(n + " games · model " + hits + "/" + n + " on " + date);
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
    deep.innerHTML = '<td colspan="7"><div class="deep-inner">' + deepPanel(g) + "</div></td>";
    tr.onclick = () => deep.classList.toggle("open");
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
  const saved = localStorage.getItem("nbaedge-theme") || "tactical";
  document.body.dataset.theme = saved;
  $("theme-label").textContent = saved.toUpperCase();
  $("theme-toggle").onclick = () => {
    const next = document.body.dataset.theme === "tactical" ? "courtside" : "tactical";
    document.body.dataset.theme = next;
    $("theme-label").textContent = next.toUpperCase();
    localStorage.setItem("nbaedge-theme", next);
  };
}
function initVisits() {
  const n = (parseInt(localStorage.getItem("nbaedge-visits") || "0", 10) || 0) + 1;
  localStorage.setItem("nbaedge-visits", String(n));
  $("visitText").textContent = n === 1 ? "first visit" : n + " visits";
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
