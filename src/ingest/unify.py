"""
unify.py — Phase 1c: fuse the archive base layer with scraped gap data into
one continuous, schema-identical timeline.

    games_wide_full.parquet  = archive_wide  ∪  scraped_wide
    games_long_full.parquet  = archive_long  ∪  scraped_long

Boundary rule: scraped rows are admitted iff game_date > ARCHIVE_MAX_DATE.
The two id spaces cannot collide (archive: zero-padded numeric strings;
B-Ref: 'YYYYMMDD0AAA'), which is asserted, not assumed.

Home-team resolution for scraped games is structural, not inferred: the
B-Ref game id's trailing 3 characters ARE the home team abbreviation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ARCHIVE_MAX_DATE, PROCESSED_DIR, RAW_DIR
from src.ingest.crosswalk import team_crosswalk
from src.ingest.local_db import _STAT_COLS, _season_end_year, build_long


def build_scraped_wide() -> pd.DataFrame:
    tb_path = RAW_DIR / "team_box.parquet"
    sc_path = RAW_DIR / "schedule.parquet"
    assert tb_path.exists() and sc_path.exists(), (
        "FATAL: scraped store empty — run bref_scraper first"
    )
    tb = pd.read_parquet(tb_path)
    sched = pd.read_parquet(sc_path)[["bref_game_id", "game_date", "is_playoffs"]]

    xwalk = team_crosswalk().set_index("bref_abbr")["team_id"]
    tb["team_id"] = tb["abbr"].map(xwalk).astype("Int64")
    assert tb["team_id"].notna().all(), (
        f"FATAL: unmapped abbreviations: {sorted(tb.loc[tb['team_id'].isna(), 'abbr'].unique())}"
    )

    tb["is_home"] = tb.apply(lambda r: r["bref_game_id"][-3:] == r["abbr"], axis=1)
    sides = tb.groupby("bref_game_id")["is_home"].agg(["sum", "count"])
    assert ((sides["sum"] == 1) & (sides["count"] == 2)).all(), (
        "FATAL: some games lack exactly one home and one away row"
    )

    home = tb[tb["is_home"]].set_index("bref_game_id")
    away = tb[~tb["is_home"]].set_index("bref_game_id")
    wide = pd.DataFrame(index=home.index)
    wide["team_id_home"] = home["team_id"]
    wide["team_id_away"] = away["team_id"]
    wide["abbr_home"] = home["abbr"].astype("string")
    wide["abbr_away"] = away["abbr"].astype("string")
    for c in _STAT_COLS:
        wide[f"{c}_home"] = home[c].astype("float64")
        wide[f"{c}_away"] = away[c].astype("float64")

    wide = wide.reset_index().rename(columns={"bref_game_id": "game_id"})
    wide["game_id"] = wide["game_id"].astype("string")
    wide = wide.merge(
        sched.rename(columns={"bref_game_id": "game_id"}), on="game_id", how="left"
    )
    assert wide["game_date"].notna().all(), "FATAL: box scores missing schedule rows"
    wide["game_date"] = pd.to_datetime(wide["game_date"])
    wide["season"] = _season_end_year(wide["game_date"])

    wide = wide.drop(columns=["is_playoffs"])
    nightly = wide.groupby("game_date")["game_id"].transform("size")
    rs_end = (
        wide.loc[nightly >= 12].groupby("season")["game_date"].max()
        .rename("rs_end")
    )
    wide = wide.merge(rs_end, on="season", how="left")
    assert wide["rs_end"].notna().all(), "FATAL: season without a full slate"
    wide["season_type"] = (
        (wide["game_date"] > wide["rs_end"])
        .map({True: "Playoffs", False: "Regular Season"}).astype("string")
    )
    wide = wide.drop(columns=["rs_end"])
    wide["margin_home"] = wide["pts_home"] - wide["pts_away"]
    assert (wide["margin_home"] != 0).all(), "FATAL: tied final score"
    wide["home_win"] = (wide["margin_home"] > 0).astype("int8")
    return wide.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def run() -> None:
    arch_wide = pd.read_parquet(PROCESSED_DIR / "games_wide.parquet")
    boundary = pd.Timestamp(ARCHIVE_MAX_DATE)

    if (RAW_DIR / "team_box.parquet").exists():
        scraped = build_scraped_wide()
        scraped = scraped[scraped["game_date"] > boundary]
        overlap = set(arch_wide["game_id"]) & set(scraped["game_id"])
        assert not overlap, f"FATAL: id-space collision: {sorted(overlap)[:5]}"
        wide_full = pd.concat(
            [arch_wide, scraped[arch_wide.columns]], ignore_index=True
        )
    else:
        print("[unify] no scraped data yet — passing archive through")
        wide_full = arch_wide

    wide_full = wide_full.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    assert wide_full["game_id"].is_unique, "FATAL: duplicate game_id post-union"

    long_full = build_long(wide_full)
    wide_full.to_parquet(PROCESSED_DIR / "games_wide_full.parquet", index=False)
    long_full.to_parquet(PROCESSED_DIR / "games_long_full.parquet", index=False)
    print(f"[unify] wide_full={wide_full.shape} long_full={long_full.shape} "
          f"span=[{wide_full['game_date'].min().date()} .. "
          f"{wide_full['game_date'].max().date()}]")


if __name__ == "__main__":
    run()
