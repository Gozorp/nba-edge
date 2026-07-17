"""
bref_scraper.py — Phase 1b: rate-limited, backoff-protected scraper for
basketball-reference.com. Closes the archive gap (post-2023-06-12) and
supplies ALL player-level game logs (absent from the archive), including
advanced per-game metrics (TS%, eFG%, USG%, ORtg, DRtg, BPM).

Engineering contract
--------------------
* Rate limit: token interval of MIN_REQUEST_INTERVAL_S + U(0, JITTER_MAX_S).
  Sports Reference's documented cap is 20 req/min; we run at 75% of cap.
* Exponential backoff on {429, 500, 502, 503, 504} and transport errors:
  delay_k = BACKOFF_BASE_S * BACKOFF_FACTOR**k, k in [0, BACKOFF_MAX_RETRIES),
  honoring any Retry-After header, capped at BACKOFF_CAP_S.
* Idempotent + resumable: game_ids already present in the raw parquet store
  are never re-fetched. Safe to kill and rerun.
* Every request is logged to LOG_DIR/scraper.jsonl (ts, url, status, latency,
  retries) — the audit trail for ban forensics.
* NULL POLICY (raw layer): the raw store preserves source fidelity, so
  structurally-absent values (e.g., FT% at 0 FTA) remain NaN here. The
  feature layer (Phase 2) is the enforcement boundary where every NaN must
  be explicitly routed or the pipeline halts.

Usage
-----
    python -m src.ingest.bref_scraper --start 2023-10-01 --end 2026-07-08
    python -m src.ingest.bref_scraper --season 2026            # whole season
    python -m src.ingest.bref_scraper --daily                  # yesterday only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    BACKOFF_BASE_S,
    BACKOFF_CAP_S,
    BACKOFF_FACTOR,
    BACKOFF_MAX_RETRIES,
    BREF_BASE_URL,
    HTTP_TIMEOUT_S,
    JITTER_MAX_S,
    LOG_DIR,
    MIN_REQUEST_INTERVAL_S,
    RAW_DIR,
    RETRYABLE_STATUS,
    SEASON_START_MONTH,
    USER_AGENT,
)

_MP_RE = re.compile(r"^(\d+):(\d{2})$")

# B-Ref data-stat name -> archive-aligned column name (unified schema).
_BREF_TO_ARCHIVE: dict[str, str] = {
    "fg": "fgm", "fga": "fga", "fg3": "fg3m", "fg3a": "fg3a",
    "ft": "ftm", "fta": "fta", "orb": "oreb", "drb": "dreb", "trb": "reb",
    "ast": "ast", "stl": "stl", "blk": "blk", "tov": "tov", "pf": "pf",
    "pts": "pts",
}
_ADV_STATS: tuple[str, ...] = (
    "ts_pct", "efg_pct", "fg3a_per_fga_pct", "fta_per_fga_pct",
    "orb_pct", "drb_pct", "trb_pct", "ast_pct", "stl_pct", "blk_pct",
    "tov_pct", "usg_pct", "off_rtg", "def_rtg", "bpm",
)


class FatalScrapeError(RuntimeError):
    """Raised when retries are exhausted — the loop must halt, not continue."""


# ----------------------------------------------------------------------------
# HTTP layer
# ----------------------------------------------------------------------------
class RateLimitedSession:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_request_t: float = 0.0
        self._log_path = LOG_DIR / "scraper.jsonl"

    def _wait(self) -> None:
        interval = MIN_REQUEST_INTERVAL_S + random.uniform(0.0, JITTER_MAX_S)
        elapsed = time.monotonic() - self._last_request_t
        if elapsed < interval:
            time.sleep(interval - elapsed)

    def _log(self, url: str, status: int | None, latency: float, retries: int) -> None:
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "url": url, "status": status,
            "latency_s": round(latency, 3), "retries": retries,
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def fetch(self, path: str) -> str | None:
        """GET BREF_BASE_URL+path. Returns HTML, or None on 404 (page absent).

        Raises FatalScrapeError after BACKOFF_MAX_RETRIES failed attempts.
        """
        url = BREF_BASE_URL + path
        for attempt in range(BACKOFF_MAX_RETRIES):
            self._wait()
            t0 = time.monotonic()
            status: int | None = None
            try:
                resp = self.session.get(url, timeout=HTTP_TIMEOUT_S)
                status = resp.status_code
                self._last_request_t = time.monotonic()
                self._log(url, status, time.monotonic() - t0, attempt)
                if status == 200:
                    return resp.text
                if status == 404:
                    return None
                if status not in RETRYABLE_STATUS:
                    raise FatalScrapeError(f"HTTP {status} (non-retryable): {url}")
                try:
                    retry_after = float(resp.headers.get("Retry-After", 0) or 0)
                except ValueError:      # HTTP-date form — use standard backoff
                    retry_after = 0.0
            except requests.RequestException:
                self._last_request_t = time.monotonic()
                self._log(url, status, time.monotonic() - t0, attempt)
                retry_after = 0.0
            delay = min(max(BACKOFF_BASE_S * BACKOFF_FACTOR**attempt, retry_after),
                        BACKOFF_CAP_S)
            time.sleep(delay)
        raise FatalScrapeError(f"retries exhausted: {url}")


# ----------------------------------------------------------------------------
# Parsing layer — driven entirely by data-stat attributes (markup-stable).
# B-Ref hides several tables inside HTML comments; _uncomment() inlines them.
# ----------------------------------------------------------------------------
def _uncomment(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            c.replace_with(BeautifulSoup(c, "lxml"))
    return soup


def _mp_to_float(raw: str) -> float | None:
    m = _MP_RE.match(raw.strip())
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60.0
    try:
        return float(raw)          # team totals render as plain integers
    except ValueError:
        return None                # DNP / DND / suspended rows


def _row_cells(tr) -> dict[str, str]:
    return {
        c["data-stat"]: c.get_text(strip=True)
        for c in tr.find_all(["th", "td"]) if c.has_attr("data-stat")
    }


def parse_box_page(html: str, bref_game_id: str) -> tuple[list[dict], list[dict]]:
    """Extract team totals and player rows (basic + advanced) from one page.

    Returns (team_rows, player_rows). Enforces: exactly 2 teams, exactly
    2 basic + 2 advanced tables, players present on both sides.
    """
    soup = _uncomment(html)
    tables = {
        t["id"]: t for t in soup.find_all("table")
        if t.has_attr("id") and re.match(r"^box-[A-Z]{3}-game-(basic|advanced)$", t["id"])
    }
    abbrs = sorted({tid.split("-")[1] for tid in tables})
    assert len(abbrs) == 2, f"FATAL: expected 2 teams, got {abbrs} in {bref_game_id}"
    assert len(tables) == 4, f"FATAL: expected 4 box tables in {bref_game_id}"

    team_rows: list[dict] = []
    player_rows: list[dict] = []

    for abbr in abbrs:
        merged: dict[str, dict] = {}       # player_id -> row dict
        totals: dict[str, float | str] = {"bref_game_id": bref_game_id, "abbr": abbr}

        for kind in ("basic", "advanced"):
            table = tables[f"box-{abbr}-game-{kind}"]
            for tr in table.find_all("tr"):
                th = tr.find("th", attrs={"data-stat": "player"})
                if th is None:
                    continue
                cells = _row_cells(tr)
                if th.get_text(strip=True) == "Team Totals":
                    src = _BREF_TO_ARCHIVE if kind == "basic" else {
                        s: s for s in _ADV_STATS}
                    for bref_name, out_name in src.items():
                        v = cells.get(bref_name, "")
                        totals[out_name] = float(v) if v not in ("", None) else None
                    if kind == "basic":
                        totals["mp"] = _mp_to_float(cells.get("mp", ""))
                    continue
                pid = th.get("data-append-csv")
                if pid is None:            # header/divider rows
                    continue
                mp = _mp_to_float(cells.get("mp", ""))
                if mp is None:             # DNP: route out, tracked by absence
                    continue
                row = merged.setdefault(pid, {
                    "bref_game_id": bref_game_id, "abbr": abbr,
                    "player_id": pid, "player": th.get_text(strip=True),
                    "mp": mp, "starter": len(merged) < 5,
                })
                names = _BREF_TO_ARCHIVE if kind == "basic" else {
                    s: s for s in _ADV_STATS}
                for bref_name, out_name in names.items():
                    v = cells.get(bref_name, "")
                    row[out_name] = float(v) if v not in ("", None) else None
                if kind == "basic":
                    pm = cells.get("plus_minus", "").replace("+", "")
                    row["plus_minus"] = float(pm) if pm not in ("", None) else None

        assert len(merged) >= 5, f"FATAL: <5 players for {abbr} in {bref_game_id}"
        player_rows.extend(merged.values())
        team_rows.append(totals)

    # Zero-sum sanity: both teams' points parsed and unequal.
    pts = [t.get("pts") for t in team_rows]
    assert all(p is not None for p in pts) and pts[0] != pts[1], (
        f"FATAL: bad team totals in {bref_game_id}: {pts}"
    )
    return team_rows, player_rows


def parse_schedule_month(html: str) -> list[dict]:
    """One month page -> rows with date, teams, and bref_game_id.

    A <tr class="thead"> whose text is 'Playoffs' partitions the page;
    is_playoffs toggles exactly once (verified monotone boolean state).
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="schedule")
    if table is None:
        return []
    rows: list[dict] = []
    is_playoffs = False
    for tr in table.find_all("tr"):
        if "thead" in (tr.get("class") or []) and "Playoffs" in tr.get_text():
            is_playoffs = True
            continue
        cells = _row_cells(tr)
        link_cell = tr.find("td", attrs={"data-stat": "box_score_text"})
        a = link_cell.find("a") if link_cell else None
        if a is None:                      # future / unplayed game
            continue
        bref_game_id = a["href"].split("/")[-1].removesuffix(".html")
        rows.append({
            "bref_game_id": bref_game_id,
            "game_date": pd.to_datetime(cells["date_game"]),
            "visitor": cells.get("visitor_team_name", ""),
            "home": cells.get("home_team_name", ""),
            "is_playoffs": is_playoffs,
        })
    return rows


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def _season_of(d: date) -> int:
    return d.year + 1 if d.month >= SEASON_START_MONTH else d.year


def _month_paths(sess: RateLimitedSession, season: int) -> list[str]:
    """Read the season index page and return its month-page paths, in order."""
    html = sess.fetch(f"/leagues/NBA_{season}_games.html")
    if html is None:
        return []
    soup = BeautifulSoup(html, "lxml")
    div = soup.find("div", class_="filter")
    if div is None:
        return [f"/leagues/NBA_{season}_games.html"]
    return [a["href"] for a in div.find_all("a") if a.has_attr("href")]


def _upsert(df: pd.DataFrame, path: Path, key: list[str]) -> None:
    if df.empty:                        # all-404 checkpoint: nothing to write
        return
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=key, keep="last")
    tmp = path.with_suffix(".parquet.tmp")
    df.sort_values(key).to_parquet(tmp, index=False)
    os.replace(tmp, path)               # atomic: never corrupt the store


def _done_ids() -> set[str]:
    p = RAW_DIR / "team_box.parquet"
    return set(pd.read_parquet(p, columns=["bref_game_id"])["bref_game_id"]) if p.exists() else set()


def scrape_range(start: date, end: date) -> None:
    assert start <= end, "FATAL: start > end"
    sess = RateLimitedSession()
    done = _done_ids()

    schedule: list[dict] = []
    for season in range(_season_of(start), _season_of(end) + 1):
        for path in _month_paths(sess, season):
            html = sess.fetch(path)
            if html:
                schedule.extend(parse_schedule_month(html))
    sched = pd.DataFrame(schedule)
    if sched.empty:
        print("[scraper] no games found in range")
        return
    sched = sched[(sched["game_date"].dt.date >= start)
                  & (sched["game_date"].dt.date <= end)]
    sched = sched.drop_duplicates(subset=["bref_game_id"])
    _upsert(sched, RAW_DIR / "schedule.parquet", ["bref_game_id"])

    todo = [g for g in sched["bref_game_id"] if g not in done]
    est_min = len(todo) * (MIN_REQUEST_INTERVAL_S + JITTER_MAX_S / 2) / 60
    print(f"[scraper] {len(sched)} games in range | {len(todo)} to fetch "
          f"| est {est_min:.0f} min at current rate limit")

    team_buf: list[dict] = []
    player_buf: list[dict] = []
    for i, gid in enumerate(todo, 1):
        html = sess.fetch(f"/boxscores/{gid}.html")
        if html is None:
            print(f"[scraper] WARN 404 box score {gid} — skipped")
            continue
        t_rows, p_rows = parse_box_page(html, gid)
        team_buf.extend(t_rows)
        player_buf.extend(p_rows)
        if i % 25 == 0:                     # checkpoint every 25 games
            # player rows FIRST: a game only becomes "done" (_done_ids reads
            # team_box) once its player rows are already on disk.
            _upsert(pd.DataFrame(player_buf), RAW_DIR / "player_box.parquet",
                    ["bref_game_id", "player_id"])
            _upsert(pd.DataFrame(team_buf), RAW_DIR / "team_box.parquet",
                    ["bref_game_id", "abbr"])
            team_buf, player_buf = [], []
            print(f"[scraper] checkpoint {i}/{len(todo)}")
    if team_buf or player_buf:              # final flush (a 404 `continue` on
        _upsert(pd.DataFrame(player_buf),   # the last game must not skip it)
                RAW_DIR / "player_box.parquet", ["bref_game_id", "player_id"])
        _upsert(pd.DataFrame(team_buf), RAW_DIR / "team_box.parquet",
                ["bref_game_id", "abbr"])
        print(f"[scraper] final flush ({len(todo)} games processed)")
    print("[scraper] complete")


def main() -> None:
    ap = argparse.ArgumentParser(description="basketball-reference scraper")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--daily", action="store_true", help="scrape yesterday")
    g.add_argument("--season", type=int, help="scrape one season (end year)")
    g.add_argument("--start", type=str, help="YYYY-MM-DD (requires --end)")
    ap.add_argument("--end", type=str, default=None)
    a = ap.parse_args()
    if a.daily:
        y = date.today() - timedelta(days=1)
        scrape_range(y, y)
    elif a.season:
        scrape_range(date(a.season - 1, SEASON_START_MONTH, 1),
                     date(a.season, 7, 31))
    else:
        assert a.end is not None, "--start requires --end"
        scrape_range(date.fromisoformat(a.start), date.fromisoformat(a.end))


if __name__ == "__main__":
    main()
