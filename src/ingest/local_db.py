"""
local_db.py — Phase 1a: deterministic ingestion of the local Kaggle archive
(nba.sqlite) into typed parquet base tables.

The archive is the immutable historical base layer (1946-11 -> 2023-06-12,
65,698 games, team-level only). The scraper (bref_scraper.py) supplies
everything after ARCHIVE_MAX_DATE plus all player-level game logs.

Outputs (PROCESSED_DIR):
    games_wide.parquet  — one row per game  (home/away paired; model targets)
    games_long.parquet  — one row per team-game (feature engineering base)

Null policy (NaN = fatal unless explicitly routed):
    * Percentage columns are RECOMPUTED from makes/attempts; x/0 := 0.0.
      Justification: FG%/3P%/FT% are undefined at 0 attempts; 0.0 is the
      unique value that keeps  pts = 2*fgm + fg3m + ftm  decompositions
      consistent while adding no phantom efficiency.
    * plus_minus is dropped from the long table (redundant: derivable as
      pts - opp_pts, which we compute exactly).
    * Rows failing any assertion are never silently dropped; counts are
      logged and asserted against explicit bounds.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from config import (
    ARCHIVE_GAME_COLS,
    ARCHIVE_GAME_ROWS_MIN,
    ARCHIVE_MAX_DATE,
    ARCHIVE_SQLITE,
    PROCESSED_DIR,
    SEASON_START_MONTH,
)

# Earliest season with every box-score field populated in THIS archive.
# Measured, not assumed: the last row with any null core stat is dated
# 1985-04-14; from 1985-10-01 onward 44,909/44,909 games are complete.
EPOCH_START: str = "1985-10-01"

# Explicit type registry for the game table (strong typing at ingestion).
_STAT_COLS: tuple[str, ...] = (
    "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts",
)
_KEEP_SEASON_TYPES: frozenset[str] = frozenset({"Regular Season", "Playoffs"})


def _season_end_year(dates: pd.Series) -> pd.Series:
    """f: datetime -> int. Season label = calendar year the season ends in."""
    y = dates.dt.year.astype("Int32")
    return (y + (dates.dt.month >= SEASON_START_MONTH).astype("Int32")).astype("Int32")


def load_archive_games() -> pd.DataFrame:
    """Read the archive game table; enforce shape and typing contracts."""
    assert ARCHIVE_SQLITE.exists(), f"FATAL: archive missing at {ARCHIVE_SQLITE}"
    con = sqlite3.connect(str(ARCHIVE_SQLITE))
    try:
        g = pd.read_sql_query("SELECT * FROM game", con)
    finally:
        con.close()

    # --- dimensionality assertions (fail fast if archive mutated) ----------
    assert g.shape[0] >= ARCHIVE_GAME_ROWS_MIN, (
        f"FATAL: game rows {g.shape[0]} < {ARCHIVE_GAME_ROWS_MIN}"
    )
    assert g.shape[1] == ARCHIVE_GAME_COLS, (
        f"FATAL: game cols {g.shape[1]} != {ARCHIVE_GAME_COLS}"
    )

    g["game_date"] = pd.to_datetime(g["game_date"], errors="raise")
    assert g["game_date"].max().strftime("%Y-%m-%d") == ARCHIVE_MAX_DATE, (
        "FATAL: archive coverage boundary moved; update config.ARCHIVE_MAX_DATE"
    )
    return g


def build_wide(g: pd.DataFrame) -> pd.DataFrame:
    """One row per game with paired home/away stats + model targets."""
    g = g[g["season_type"].isin(_KEEP_SEASON_TYPES)].copy()
    g = g[g["game_date"] >= EPOCH_START].copy()

    # Deduplicate on primary key (archive contains rare exact duplicates).
    n_before = len(g)
    g = g.drop_duplicates(subset=["game_id"], keep="first")
    n_dupes = n_before - len(g)
    assert n_dupes <= int(0.001 * n_before), (
        f"FATAL: {n_dupes} duplicate game_ids (> 0.1% of rows) — data corrupt"
    )

    for side in ("home", "away"):
        for c in _STAT_COLS:
            col = f"{c}_{side}"
            g[col] = pd.to_numeric(g[col], errors="coerce")
        # Null routing: a modern-era team-game with missing core counts is
        # unusable; drop with tracking rather than impute fabricated stats.
    core = [f"{c}_{s}" for c in _STAT_COLS for s in ("home", "away")]
    n_null = int(g[core].isna().any(axis=1).sum())
    g = g.dropna(subset=core)
    assert n_null <= int(0.005 * (len(g) + n_null)), (
        f"FATAL: {n_null} rows with null core stats exceeds 0.5% tolerance"
    )

    out = pd.DataFrame({
        "game_id": g["game_id"].astype("string"),
        "game_date": g["game_date"],
        "season": _season_end_year(g["game_date"]),
        "season_type": g["season_type"].astype("string"),
        "team_id_home": g["team_id_home"].astype("Int64"),
        "team_id_away": g["team_id_away"].astype("Int64"),
        "abbr_home": g["team_abbreviation_home"].astype("string"),
        "abbr_away": g["team_abbreviation_away"].astype("string"),
    })
    for c in _STAT_COLS:
        out[f"{c}_home"] = g[f"{c}_home"].astype("float64")
        out[f"{c}_away"] = g[f"{c}_away"].astype("float64")

    # Targets. Domain: margin in Z (integers, no ties in NBA); home_win {0,1}.
    out["margin_home"] = out["pts_home"] - out["pts_away"]
    assert (out["margin_home"] != 0).all(), "FATAL: tied final score encountered"
    out["home_win"] = (out["margin_home"] > 0).astype("int8")

    out = out.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    assert out["game_id"].is_unique, "FATAL: game_id not unique post-build"
    return out


def build_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (team perspective). Base table for rolling features."""
    frames: list[pd.DataFrame] = []
    for side, opp in (("home", "away"), ("away", "home")):
        f = pd.DataFrame({
            "game_id": wide["game_id"],
            "game_date": wide["game_date"],
            "season": wide["season"],
            "season_type": wide["season_type"],
            "team_id": wide[f"team_id_{side}"],
            "abbr": wide[f"abbr_{side}"],
            "opp_id": wide[f"team_id_{opp}"],
            "is_home": pd.Series(side == "home", index=wide.index, dtype="boolean"),
        })
        for c in _STAT_COLS:
            f[c] = wide[f"{c}_{side}"]
            f[f"opp_{c}"] = wide[f"{c}_{opp}"]
        frames.append(f)

    long = pd.concat(frames, ignore_index=True)
    # --- dimensionality assertion: long MUST be exactly 2x wide ------------
    assert long.shape[0] == 2 * wide.shape[0], (
        f"FATAL: long rows {long.shape[0]} != 2 * {wide.shape[0]}"
    )
    assert long["team_id"].notna().all(), "FATAL: null team_id in long table"

    # Recomputed efficiency (x/0 := 0.0 — see module docstring).
    with np.errstate(divide="ignore", invalid="ignore"):
        long["fg_pct"] = np.where(long["fga"] > 0, long["fgm"] / long["fga"], 0.0)
        long["fg3_pct"] = np.where(long["fg3a"] > 0, long["fg3m"] / long["fg3a"], 0.0)
        long["ft_pct"] = np.where(long["fta"] > 0, long["ftm"] / long["fta"], 0.0)

    # Possession estimate (Kubatko et al.):  POSS = FGA + 0.44*FTA - OREB + TOV
    # 0.44 = empirical share of FTA that end possessions (and-1s, technicals,
    # 3-shot fouls excluded) — the standard basketball-analytics constant.
    long["poss"] = long["fga"] + 0.44 * long["fta"] - long["oreb"] + long["tov"]
    assert (long["poss"] > 0).all(), "FATAL: non-positive possession estimate"
    long["off_rtg"] = 100.0 * long["pts"] / long["poss"]
    opp_poss = (
        long["opp_fga"] + 0.44 * long["opp_fta"] - long["opp_oreb"] + long["opp_tov"]
    )
    long["def_rtg"] = 100.0 * long["opp_pts"] / opp_poss
    long["net_rtg"] = long["off_rtg"] - long["def_rtg"]
    long["pace"] = 0.5 * (long["poss"] + opp_poss)
    long["win"] = (long["pts"] > long["opp_pts"]).astype("int8")

    long = long.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    assert not long[["off_rtg", "def_rtg", "pace"]].isna().any().any(), (
        "FATAL: NaN produced in derived ratings"
    )
    return long


def run() -> None:
    t0 = datetime.now()
    g = load_archive_games()
    wide = build_wide(g)
    long = build_long(wide)

    wide.to_parquet(PROCESSED_DIR / "games_wide.parquet", index=False)
    long.to_parquet(PROCESSED_DIR / "games_long.parquet", index=False)
    print(
        f"[local_db] wide={wide.shape} long={long.shape} "
        f"span=[{wide['game_date'].min().date()} .. {wide['game_date'].max().date()}] "
        f"elapsed={(datetime.now() - t0).total_seconds():.1f}s"
    )


if __name__ == "__main__":
    run()
