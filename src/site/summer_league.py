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
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            payload = json.load(r)
    except Exception as e:                      # network is best-effort here
        print(f"[sl] WARN {league} {d}: {e}")
        return []
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
            "tip_utc": ev.get("date"),
            "state": status.get("name", "STATUS_SCHEDULED"),
            "is_final": bool(status.get("completed")),
            "away": sides["away"]["team"].get("displayName"),
            "home": sides["home"]["team"].get("displayName"),
            "away_abbr": sides["away"]["team"].get("abbreviation", ""),
            "home_abbr": sides["home"]["team"].get("abbreviation", ""),
            "away_score": int(sides["away"].get("score") or 0),
            "home_score": int(sides["home"].get("score") or 0),
        })
    return games


def export(start: date, end: date) -> None:
    assert start <= end, "FATAL: start > end"
    sl_dates: list[str] = []
    d = start
    while d <= end:
        all_games: list[dict] = []
        for code in LEAGUES:
            all_games.extend(_fetch(code, d))
        if all_games:
            all_games.sort(key=lambda g: g["tip_utc"])
            iso = d.isoformat()
            (OUT_DIR / f"sl_{iso}.json").write_text(json.dumps(
                {"date": iso, "games": all_games}, indent=0))
            sl_dates.append(iso)
            print(f"[sl] {iso}: {len(all_games)} games "
                  f"({sum(g['is_final'] for g in all_games)} final)")
        d += timedelta(days=1)

    mpath = OUT_DIR / "manifest.json"
    manifest = json.loads(mpath.read_text())
    existing = set(manifest.get("sl_dates", []))
    merged = sorted(existing | set(sl_dates), reverse=True)
    manifest["sl_dates"] = merged
    manifest["sl_built_at"] = datetime.utcnow().isoformat(timespec="seconds")
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"[sl] manifest: {len(merged)} SL dates")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    y = date.today().year
    ap.add_argument("--start", default=f"{y}-07-01")
    ap.add_argument("--end", default=f"{y}-07-25")
    a = ap.parse_args()
    export(date.fromisoformat(a.start), date.fromisoformat(a.end))
