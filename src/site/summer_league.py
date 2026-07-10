"""
summer_league.py — Summer League slate exporter (schedule + results only).

DELIBERATELY UNGRADED. The game models are trained on NBA regular-season
regimes; Summer League is 5-game exhibition rosters with no stable team
identity — feeding it through the model would produce confidently labeled
noise. The terminal shows the SL slate and verified results, nothing more.

Source: ESPN public scoreboard API (no key), league codes:
    nba-summer-las-vegas | nba-summer-utah | nba-summer-california

Outputs:
    docs/data/sl_{date}.json          per-date merged slate across leagues
    manifest.json                     gains "sl_dates" (desc), rest preserved

Usage:
    python -m src.site.summer_league [--start 2026-07-01 --end 2026-07-25]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import REPO_ROOT

OUT_DIR = REPO_ROOT / "docs" / "data"
LEAGUES: dict[str, str] = {
    "nba-summer-las-vegas": "Las Vegas",
    "nba-summer-utah": "Salt Lake City",
    "nba-summer-california": "California Classic",
}
_API = ("https://site.api.espn.com/apis/site/v2/sports/basketball/"
        "{league}/scoreboard?dates={yyyymmdd}")


def _fetch(league: str, d: date) -> list[dict]:
    url = _API.format(league=league, yyyymmdd=d.strftime("%Y%m%d"))
    payload = None
    for attempt in range(3):                    # ESPN 5xx under parallel load
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                payload = json.load(r)
            break
        except Exception as e:
            if attempt == 2:
                print(f"[sl] WARN {league} {d}: {e}")
                return []
            import time as _t
            _t.sleep(1.2 * (attempt + 1))
    games = []
    for ev in payload.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if set(sides) != {"home", "away"}:
            continue
        status = ev.get("status", {}).get("type", {})
        games.append({
            "espn_id": ev.get("id"),
            "league": LEAGUES[league],
            # ESPN's ?dates= parameter buckets by US-Eastern slate date; keep
            # that as the game's slate identity so late ET tips (01:00+ UTC)
            # don't spill onto the next day's slate.
            "slate_date": d.isoformat(),
            "tip_utc": ev.get("date"),
            "state": status.get("name", "STATUS_SCHEDULED"),
            "detail": status.get("shortDetail", ""),
            "is_final": bool(status.get("completed")),
            "away": sides["away"]["team"].get("displayName"),
            "home": sides["home"]["team"].get("displayName"),
            "away_abbr": sides["away"]["team"].get("abbreviation", ""),
            "home_abbr": sides["home"]["team"].get("abbreviation", ""),
            "away_score": int(sides["away"].get("score") or 0),
            "home_score": int(sides["home"].get("score") or 0),
        })
    return games


# =============================================================================
# SL-native pick model.
# The NBA models are NOT used here (their features describe parent-club
# rosters). Instead: a 2-feature logistic regression on WITHIN-TOURNAMENT
# signal only, trained on complete SL tournaments 2023-2024 and verified
# out-of-sample on SL 2025 before grading the current year.
#     x1 = shrunk in-summer point-diff edge:
#          s(T) = mean_diff(T) * n/(n+2)   (k=2 shrinkage vs 1-2 game noise)
#          x1 = s(home) - s(away)
#     x2 = rest edge, capped at 3 days (first game := 3)
# Picks are labeled LEAN (|p-.5| >= .10) or COINFLIP — never A-D tiers.
# =============================================================================
def _summer_frame(year: int) -> list[dict]:
    """All SL games for a summer. Parallel fetch; past years disk-cached
    (a finished tournament is immutable)."""
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    cache = Path(tempfile.gettempdir()) / f"sl_frame_{year}.json"
    if year < date.today().year and cache.exists():
        return json.loads(cache.read_text())
    jobs = [(code, date(year, 7, 1) + timedelta(days=i))
            for i in range(25) for code in LEAGUES]
    games: list[dict] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for chunk in ex.map(lambda j: _fetch(*j), jobs):
            games.extend(chunk)
    games.sort(key=lambda g: g["tip_utc"])
    # cache only complete-looking tournaments (gaps would corrupt features)
    if year < date.today().year and len(games) >= 50:
        cache.write_text(json.dumps(games))
    return games


def _featurize(games: list[dict]) -> list[dict]:
    """Chronological replay: attach form/rest features BEFORE each game."""
    diff: dict[str, list[float]] = {}
    last: dict[str, str] = {}
    rows = []
    for g in games:
        h, a = g["home_abbr"], g["away_abbr"]
        def state(t):
            ds = diff.get(t, [])
            n = len(ds)
            shrunk = (sum(ds) / n) * n / (n + 2.0) if n else 0.0
            day = g["tip_utc"][:10]
            rest = 3.0
            if t in last:
                rest = min(3.0, max(0.0,
                    (date.fromisoformat(day) - date.fromisoformat(last[t])).days - 1))
            return shrunk, rest
        sh, rh = state(h)
        sa, ra = state(a)
        rows.append({**g, "x1": sh - sa, "x2": rh - ra})
        if g["is_final"]:
            margin = g["home_score"] - g["away_score"]
            diff.setdefault(h, []).append(float(margin))
            diff.setdefault(a, []).append(float(-margin))
            last[h] = last[a] = g["tip_utc"][:10]
    return rows


def fit_sl_model() -> dict:
    """Train 2023+2024, verify on 2025, refit on all three for deployment."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    frames = {y: [r for r in _featurize(_summer_frame(y)) if r["is_final"]]
              for y in (2023, 2024, 2025)}
    for y, f in frames.items():
        assert len(f) >= 50, f"FATAL: only {len(f)} finished SL games in {y}"

    def xy(rows):
        X = np.array([[r["x1"], r["x2"]] for r in rows])
        yv = np.array([1.0 if r["home_score"] > r["away_score"] else 0.0
                       for r in rows])
        return X, yv

    Xtr, ytr = xy(frames[2023] + frames[2024])
    Xev, yev = xy(frames[2025])
    m = LogisticRegression().fit(Xtr, ytr)
    p = m.predict_proba(Xev)[:, 1]
    acc = float(((p >= 0.5) == yev).mean())
    ll = float(log_loss(yev, p))

    Xall = np.vstack([Xtr, Xev])
    yall = np.concatenate([ytr, yev])
    final = LogisticRegression().fit(Xall, yall)
    return {"model": final, "eval_2025": {"n": int(len(yev)),
            "acc": round(acc, 4), "logloss": round(ll, 4)},
            "coef": [round(float(c), 4) for c in final.coef_[0]],
            "intercept": round(float(final.intercept_[0]), 4),
            "n_train": int(len(yall))}


def export(start: date, end: date) -> None:
    assert start <= end, "FATAL: start > end"
    fit = fit_sl_model()
    print(f"[sl-model] 2025 OOS: acc={fit['eval_2025']['acc']:.3f} "
          f"logloss={fit['eval_2025']['logloss']:.3f} "
          f"(n={fit['eval_2025']['n']}) coef={fit['coef']}")

    season = _featurize(_summer_frame(start.year))     # current tournament
    import numpy as np
    for g in season:
        p_home = float(fit["model"].predict_proba(
            np.array([[g["x1"], g["x2"]]]))[0, 1])
        pick_home = p_home >= 0.5
        g["p_home"] = round(p_home, 4)
        g["pick"] = g["home_abbr"] if pick_home else g["away_abbr"]
        g["p_pick"] = round(p_home if pick_home else 1 - p_home, 4)
        g["tier"] = "LEAN" if abs(p_home - 0.5) >= 0.10 else "COINFLIP"
        if g["is_final"]:
            g["pick_correct"] = int(pick_home ==
                                    (g["home_score"] > g["away_score"]))
        del g["x1"], g["x2"]

    by_date: dict[str, list[dict]] = {}
    for g in season:
        iso = g.get("slate_date") or g["tip_utc"][:10]
        if start <= date.fromisoformat(iso) <= end:
            by_date.setdefault(iso, []).append(g)

    sl_dates: list[str] = []
    for iso, games in sorted(by_date.items()):
        (OUT_DIR / f"sl_{iso}.json").write_text(json.dumps(
            {"date": iso, "games": games}, indent=0))
        sl_dates.append(iso)
        fin = sum(g["is_final"] for g in games)
        hits = sum(g.get("pick_correct", 0) for g in games)
        print(f"[sl] {iso}: {len(games)} games ({fin} final, model {hits}/{fin})")

    mpath = OUT_DIR / "manifest.json"
    manifest = json.loads(mpath.read_text())
    existing = set(manifest.get("sl_dates", []))
    merged = sorted(existing | set(sl_dates), reverse=True)
    manifest["sl_dates"] = merged
    manifest["sl_built_at"] = datetime.utcnow().isoformat(timespec="seconds")
    manifest["sl_model"] = {"eval_2025": fit["eval_2025"],
                            "coef": fit["coef"], "intercept": fit["intercept"],
                            "n_train": fit["n_train"],
                            "features": ["in_summer_form_edge", "rest_edge"]}
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"[sl] manifest: {len(merged)} SL dates")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    y = date.today().year
    ap.add_argument("--start", default=f"{y}-07-01")
    ap.add_argument("--end", default=f"{y}-07-25")
    a = ap.parse_args()
    export(date.fromisoformat(a.start), date.fromisoformat(a.end))
