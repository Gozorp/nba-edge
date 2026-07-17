"""
contracts.py — Data-contract guard for every artifact the terminal serves.

WHY THIS EXISTS
---------------
The Summer League "stuck slate" bug was a silent contract violation: ESPN
emitted status values (STATUS_HALFTIME, STATUS_END_PERIOD, ...) outside the
set the renderer enumerated, and the failure mode was wrong-but-quiet UI.
This module makes that class of error IMPOSSIBLE to miss again:

  1. STRUCTURAL rules (missing keys, wrong types, out-of-range numbers)
     are RED: the validator exits non-zero and the publish gate in
     PUSH_NBA_SITE.bat refuses to ship the artifact.
  2. ENUM WATCHLISTS (upstream vocabularies we do not control) are
     fail-open: unknown values are allowed through — the UI is written to
     degrade safely — but every unknown is recorded, counted, and surfaced
     as a YELLOW check in health.json, which the site's TOOLS drawer shows.
     A new ESPN state now appears on the dashboard the day it first occurs,
     not the day a user screenshots a frozen slate.

Run:  python -m src.site.contracts          # validate docs/data, update health
Exit: 0 clean/yellow · 1 structural violations (blocks publish)
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import REPO_ROOT

DATA_DIR = REPO_ROOT / "docs" / "data"
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# ---- enum watchlists (upstream vocabularies; unknown => yellow, never red) --
KNOWN_ESPN_STATES = {
    "STATUS_SCHEDULED", "STATUS_IN_PROGRESS", "STATUS_HALFTIME",
    "STATUS_END_PERIOD", "STATUS_FINAL", "STATUS_POSTPONED",
    "STATUS_CANCELED", "STATUS_SUSPENDED", "STATUS_DELAYED",
    "STATUS_RAIN_DELAY", "STATUS_UNCONTESTED",
}
KNOWN_SL_LEAGUES = {"Las Vegas", "Salt Lake City", "California Classic"}
KNOWN_GRADES = {"A", "A-", "B+", "B", "B-", "C", "D"}
KNOWN_PICK_TIERS = {"DIAMOND", "PLATINUM", "GOLD", "SKIP"}   # conviction system
KNOWN_SL_TIERS = {"LEAN", "COINFLIP"}
KNOWN_SIGNALS = {"QUALITY", "FORM", "SCHEDULE", "AGREEMENT", "VALUE"}
KNOWN_PROP_METHODS = {"model", "baseline_r7"}
KNOWN_NOPROJ_REASONS = {"below_floor", "insufficient_history"}
PROP_MARKETS = {"pts", "reb", "ast", "stl", "blk", "fg3m"}


class Report:
    def __init__(self) -> None:
        self.red: list[str] = []
        self.unknown: dict[str, set] = {}     # watchlist -> unseen values
        self.files = 0

    def structural(self, msg: str) -> None:
        if len(self.red) < 50:
            self.red.append(msg)

    def watch(self, listname: str, value, known: set) -> None:
        if value not in known:
            self.unknown.setdefault(listname, set()).add(str(value))


def _num(x) -> bool: return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_sl(path: Path, rep: Report) -> None:
    d = json.loads(path.read_text())
    if not _ISO_DATE.match(d.get("date", "")):
        rep.structural(f"{path.name}: bad/missing date")
    for g in d.get("games", []):
        for k in ("espn_id", "league", "tip_utc", "state", "away", "home",
                  "away_abbr", "home_abbr"):
            if not g.get(k):
                rep.structural(f"{path.name}: game missing '{k}'"); break
        if not isinstance(g.get("is_final"), bool):
            rep.structural(f"{path.name}: is_final not bool")
        for k in ("away_score", "home_score"):
            if not (_num(g.get(k)) and g[k] >= 0):
                rep.structural(f"{path.name}: bad {k}")
        if "p_pick" in g and not (_num(g["p_pick"]) and 0.0 < g["p_pick"] <= 1.0):
            rep.structural(f"{path.name}: p_pick out of (0,1]")
        rep.watch("espn_state", g.get("state"), KNOWN_ESPN_STATES)
        rep.watch("sl_league", g.get("league"), KNOWN_SL_LEAGUES)
        if "tier" in g:
            rep.watch("sl_tier", g.get("tier"), KNOWN_SL_TIERS)


def check_picks(path: Path, rep: Report) -> None:
    d = json.loads(path.read_text())
    for g in d.get("games", []):
        p = g.get("p_home")
        if not (_num(p) and 0.0 < p < 1.0):
            rep.structural(f"{path.name}: p_home out of (0,1)")
        if not (_num(g.get("pred_margin_home"))):
            rep.structural(f"{path.name}: bad pred_margin_home")
        rep.watch("grade", g.get("grade"), KNOWN_GRADES)
        rep.watch("pick_tier", g.get("tier"), KNOWN_PICK_TIERS)
        for s in g.get("signals", []):
            rep.watch("signal", s, KNOWN_SIGNALS)
        sf = g.get("stake_frac", 0.0)
        if not (_num(sf) and 0.0 <= sf <= 0.05):
            rep.structural(f"{path.name}: stake_frac outside [0,0.05]")
        for k in ("edge_pp", "ev_per_dollar"):
            v = g.get(k)
            if v is not None and not _num(v):
                rep.structural(f"{path.name}: {k} non-numeric")
        for f in g.get("factors", []):
            if not (isinstance(f.get("label"), str) and _num(f.get("impact_pp"))):
                rep.structural(f"{path.name}: malformed factor"); break
        r = g.get("result")
        if r and not (isinstance(r.get("home_win"), int)
                      and _num(r.get("margin_home")) and r.get("margin_home") != 0):
            rep.structural(f"{path.name}: malformed result")


def check_props(path: Path, rep: Report) -> None:
    d = json.loads(path.read_text())
    for gid, rows in d.get("games", {}).items():
        for r in rows:
            if not r.get("player"):
                rep.structural(f"{path.name}: prop row missing player"); break
            if r.get("proj") is None:
                # no-projection rows must carry a known named reason
                rep.watch("noproj_reason", r.get("reason"), KNOWN_NOPROJ_REASONS)
                sides = ("actual",)
            else:
                sides = ("proj", "actual")
            for side in sides:
                block = r.get(side, {})
                if set(block) != PROP_MARKETS:
                    rep.structural(f"{path.name}: {side} markets != {sorted(PROP_MARKETS)}")
                    break
                if any(not (_num(v) and v >= 0) for v in block.values()):
                    rep.structural(f"{path.name}: negative/non-numeric {side}")
                    break


def check_manifest(rep: Report) -> dict:
    m = json.loads((DATA_DIR / "manifest.json").read_text())
    # cross-artifact integrity: if a family of files exists, its manifest
    # keys MUST exist (the 2026-07-10 SL-tab blanking, made structural)
    if list(DATA_DIR.glob("sl_*.json")):
        if not m.get("sl_dates"):
            rep.structural("manifest: sl_*.json exist but sl_dates missing/empty")
        if not m.get("sl_model"):
            rep.structural("manifest: sl files exist but sl_model block missing")
    if list(DATA_DIR.glob("props_*.json")) and not m.get("props", {}).get("mae"):
        rep.structural("manifest: props files exist but props.mae missing")
    for key in ("dates", "sl_dates"):
        ds = m.get(key, [])
        if any(not _ISO_DATE.match(x) for x in ds):
            rep.structural(f"manifest: non-ISO entry in {key}")
        if ds != sorted(ds, reverse=True):
            rep.structural(f"manifest: {key} not sorted desc")
    acc = m.get("record", {}).get("acc")
    if acc is not None and not (0.0 < acc < 1.0):
        rep.structural("manifest: record.acc out of (0,1)")
    for tgt, meth in (m.get("props", {}).get("method") or {}).items():
        rep.watch("prop_method", meth, KNOWN_PROP_METHODS)
    return m


def validate() -> int:
    rep = Report()
    check_manifest(rep)
    for p in sorted(DATA_DIR.glob("sl_*.json")):
        rep.files += 1; check_sl(p, rep)
    for p in sorted(DATA_DIR.glob("picks_*.json")):
        rep.files += 1; check_picks(p, rep)
    for p in sorted(DATA_DIR.glob("props_*.json")):
        rep.files += 1; check_props(p, rep)

    unknown = {k: sorted(v) for k, v in rep.unknown.items()}
    verdict = "red" if rep.red else ("yellow" if unknown else "green")
    (DATA_DIR / "contract_report.json").write_text(json.dumps({
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": rep.files, "verdict": verdict,
        "structural_violations": rep.red,
        "unknown_enum_values": unknown,
    }, indent=1))

    # surface as a health check the TOOLS drawer renders
    hpath = DATA_DIR / "health.json"
    if hpath.exists():
        h = json.loads(hpath.read_text())
        h["checks"] = [c for c in h.get("checks", [])
                       if c.get("name") != "data_contracts"]
        msg = (f"{rep.files} artifacts clean" if verdict == "green"
               else f"{len(rep.red)} structural violations" if verdict == "red"
               else "unknown upstream values: "
                    + "; ".join(f"{k}={v}" for k, v in unknown.items()))
        h["checks"].append({"name": "data_contracts", "severity": verdict,
                            "message": msg, "category": "data_flow"})
        sev_rank = {"green": 0, "yellow": 1, "red": 2}
        # current checks only — carrying the old overall forward lets a
        # stale red ratchet: it could never downgrade once set
        h["overall"] = max([c["severity"] for c in h["checks"]] or ["green"],
                           key=lambda s: sev_rank.get(s, 0))
        hpath.write_text(json.dumps(h, indent=1, default=str))

    print(f"[contracts] {rep.files} files -> {verdict.upper()}"
          + (f" | unknown: {unknown}" if unknown else "")
          + (f" | RED: {rep.red[:3]}" if rep.red else ""))
    return 1 if rep.red else 0


if __name__ == "__main__":
    sys.exit(validate())
