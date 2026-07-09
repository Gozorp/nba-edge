"""
crosswalk.py — Deterministic ID reconciliation between the two source systems.

The archive (Kaggle/nba_api lineage) keys teams by numeric team_id
(1610612xxx) and 3-letter abbreviations; basketball-reference uses its own
abbreviations (3 divergences) and slug player ids ('jamesle01').

POLICY (Constraint: no fuzzy matching):
  * Teams: exact abbreviation join + a hardcoded 3-entry exception map.
    The NBA has exactly 30 teams; the mapping is verified exhaustively.
  * Players: exact match on a deterministic normalization
    N(name) = lowercase(strip_accents(remove_punct(remove_suffix(name)))).
    Any B-Ref player whose normalized name matches 0 or >1 archive players
    is routed to LOG_DIR/player_crosswalk_unresolved.csv for one-time human
    adjudication — never guessed.
"""
from __future__ import annotations

import sqlite3
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ARCHIVE_SQLITE, LOG_DIR, PROCESSED_DIR, RAW_DIR

# B-Ref abbreviation -> archive (nba_api) abbreviation. All other 27 match.
_BREF_TO_API_ABBR: dict[str, str] = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}
_SUFFIXES: frozenset[str] = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})


def normalize_name(name: str) -> str:
    """N: str -> str. Deterministic, idempotent (N(N(x)) = N(x))."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = "".join(ch if (ch.isalnum() or ch == " ") else " " for ch in s)
    tokens = [t for t in s.split() if t not in _SUFFIXES]
    return " ".join(tokens)


def team_crosswalk() -> pd.DataFrame:
    """30-row table: bref_abbr <-> team_id. Asserted exhaustive."""
    con = sqlite3.connect(str(ARCHIVE_SQLITE))
    try:
        teams = pd.read_sql_query(
            "SELECT id AS team_id, abbreviation AS api_abbr, "
            "full_name FROM team", con
        )
    finally:
        con.close()
    assert teams.shape[0] == 30, f"FATAL: expected 30 teams, got {teams.shape[0]}"

    api_to_bref = {v: k for k, v in _BREF_TO_API_ABBR.items()}
    teams["bref_abbr"] = teams["api_abbr"].map(lambda a: api_to_bref.get(a, a))
    teams["team_id"] = teams["team_id"].astype("Int64")
    assert teams["bref_abbr"].is_unique and teams["team_id"].is_unique, (
        "FATAL: team crosswalk not bijective"
    )
    return teams[["team_id", "api_abbr", "bref_abbr", "full_name"]]


def player_crosswalk() -> pd.DataFrame:
    """Map scraped B-Ref player_ids to archive person_ids where provable.

    Output columns: player_id (bref), person_id (archive, Int64 or <NA>),
    match_state in {'exact', 'unresolved_none', 'unresolved_ambiguous'}.
    """
    pb_path = RAW_DIR / "player_box.parquet"
    assert pb_path.exists(), "FATAL: run bref_scraper before building crosswalk"
    scraped = (
        pd.read_parquet(pb_path, columns=["player_id", "player"])
        .drop_duplicates(subset=["player_id"])
    )

    con = sqlite3.connect(str(ARCHIVE_SQLITE))
    try:
        arch = pd.read_sql_query(
            "SELECT person_id, display_first_last FROM common_player_info", con
        )
    finally:
        con.close()

    arch["norm"] = arch["display_first_last"].map(normalize_name)
    # Archive-side collisions (distinct persons, same normalized name) are
    # excluded from the matchable universe — matching into them is unprovable.
    counts = arch.groupby("norm")["person_id"].nunique()
    unique_norms = set(counts[counts == 1].index)
    arch_unique = arch[arch["norm"].isin(unique_norms)].drop_duplicates("norm")

    scraped["norm"] = scraped["player"].map(normalize_name)
    merged = scraped.merge(
        arch_unique[["norm", "person_id"]], on="norm", how="left"
    )
    merged["person_id"] = merged["person_id"].astype("Int64")
    merged["match_state"] = "exact"
    merged.loc[merged["person_id"].isna(), "match_state"] = "unresolved_none"
    merged.loc[
        merged["norm"].isin(set(counts[counts > 1].index)), "match_state"
    ] = "unresolved_ambiguous"

    unresolved = merged[merged["match_state"] != "exact"]
    if not unresolved.empty:
        out = LOG_DIR / "player_crosswalk_unresolved.csv"
        unresolved.to_csv(out, index=False)
        print(f"[crosswalk] {len(unresolved)} unresolved players -> {out}")

    result = merged[["player_id", "player", "person_id", "match_state"]]
    result.to_parquet(PROCESSED_DIR / "player_crosswalk.parquet", index=False)
    print(f"[crosswalk] players: {len(result)} total, "
          f"{(result['match_state'] == 'exact').sum()} exact")
    return result


if __name__ == "__main__":
    team_crosswalk().to_parquet(PROCESSED_DIR / "team_crosswalk.parquet", index=False)
    print("[crosswalk] team crosswalk written (30 rows)")
    if (RAW_DIR / "player_box.parquet").exists():
        player_crosswalk()
