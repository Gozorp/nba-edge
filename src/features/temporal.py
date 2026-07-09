"""
temporal.py — Phase 2a: leakage-free temporal feature construction.

THE ONE RULE: every feature at game t is a function of games [.., t-1] only.
Mechanically enforced by a single primitive — shift(1) BEFORE any window —
applied inside (team_id, season) groups so windows never cross seasons
(a 4-month offseason invalidates any stationarity assumption).

Feature groups
--------------
1. Form:       rolling mean over 3 / 7 games + expanding season-to-date of
               the Four Factors (Oliver) and ratings (ORtg/DRtg/NetRtg/pace).
2. Fatigue:    rest_days + discrete schedule-load states (truth table below).
3. Schedule:   opponent quality entering the game (opp season-to-date NetRtg,
               itself shifted), trailing SOS, and SOS-adjusted NetRtg.

Fatigue truth table
-------------------
Inputs (all deterministic functions of the team's game-date sequence):
    r      = rest days       ∈ {0,1,...,7}   (capped; see _rest_days)
    g4     = games in trailing 4 nights (incl. tonight) ∈ {1,2,3}
    g6     = games in trailing 6 nights (incl. tonight) ∈ {1,..,4}
Derived boolean states:
    is_b2b        := (r == 0)
    is_3in4       := (g4 >= 3)
    is_4in6       := (g6 >= 4)
Of the 2^3 = 8 joint states, only 6 are reachable under NBA scheduling
(is_4in6 -> is_3in4 is implied when is_b2b, and g4 >= 3 requires r <= 1);
we therefore emit the 3 minimal booleans rather than a one-hot of joint
states — the Karnaugh-reduced encoding.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    MIN_GAMES_FOR_FEATURES,
    PROCESSED_DIR,
    ROLLING_WINDOWS,
)

# Columns summarized by rolling windows. All are rate/per-possession stats —
# scale-free, so window means are comparable across eras and pace regimes.
FORM_COLS: tuple[str, ...] = (
    "net_rtg", "off_rtg", "def_rtg", "pace",
    "efg", "tov_rate", "oreb_rate", "ft_rate", "fg3_share",
)
MAX_REST_DAYS: int = 7   # fatigue effect saturates; literature finds no
                         # marginal effect beyond ~3 rest days. Cap prevents
                         # offseason gaps (r>100) acting as leverage points.


def add_rolling(
    df: pd.DataFrame,
    group_keys: list[str],
    cols: list[str],
    windows: tuple[int, ...] = ROLLING_WINDOWS,
    min_periods: int = MIN_GAMES_FOR_FEATURES,
) -> pd.DataFrame:
    """Generic leakage-free roller: shift(1) -> rolling/expanding mean.

    Reused by the player pipeline. df MUST be pre-sorted by date within
    groups (asserted).
    """
    assert df.groupby(group_keys, observed=True)["game_date"].apply(
        lambda s: s.is_monotonic_increasing
    ).all(), "FATAL: frame not date-sorted within groups"

    n_levels = len(group_keys)
    g = df.groupby(group_keys, observed=True, sort=False)
    for c in cols:
        shifted = g[c].shift(1)
        grouped = shifted.groupby(
            [df[k] for k in group_keys], observed=True)
        for w in windows:
            r = grouped.rolling(w, min_periods=min_periods).mean()
            df[f"{c}_r{w}"] = r.droplevel(list(range(n_levels)))  # index-aligned
        e = grouped.expanding(min_periods=min_periods).mean()
        df[f"{c}_std"] = e.droplevel(list(range(n_levels)))
    return df


def _rest_days(dates: pd.Series) -> pd.Series:
    r = (dates - dates.shift(1)).dt.days - 1
    return r.fillna(MAX_REST_DAYS).clip(upper=MAX_REST_DAYS).astype("float64")


def _games_in_window(dates: pd.Series, nights: int) -> pd.Series:
    ones = pd.Series(1.0, index=pd.DatetimeIndex(dates.values))
    counts = ones.rolling(f"{nights}D").sum()
    return pd.Series(counts.values, index=dates.index)


def build_team_features() -> pd.DataFrame:
    long = pd.read_parquet(PROCESSED_DIR / "games_long_full.parquet")
    n_in = len(long)

    # Four Factors + shot-profile rates (derived, all denominators asserted).
    assert (long["fga"] > 0).all(), "FATAL: zero FGA team-game"
    long["efg"] = (long["fgm"] + 0.5 * long["fg3m"]) / long["fga"]
    long["tov_rate"] = long["tov"] / long["poss"]
    long["oreb_rate"] = long["oreb"] / (long["oreb"] + long["opp_dreb"])
    long["ft_rate"] = long["fta"] / long["fga"]
    long["fg3_share"] = long["fg3a"] / long["fga"]

    long = long.sort_values(["team_id", "season", "game_date", "game_id"]
                            ).reset_index(drop=True)
    long = add_rolling(long, ["team_id", "season"], list(FORM_COLS))

    # Fatigue block (within team-season; dates strictly increasing asserted
    # inside add_rolling above).
    grp = long.groupby(["team_id", "season"], observed=True, sort=False)
    long["rest_days"] = grp["game_date"].transform(_rest_days)
    long["g4"] = grp["game_date"].transform(lambda s: _games_in_window(s, 4))
    long["g6"] = grp["game_date"].transform(lambda s: _games_in_window(s, 6))
    long["is_b2b"] = (long["rest_days"] == 0).astype("int8")
    long["is_3in4"] = (long["g4"] >= 3).astype("int8")
    long["is_4in6"] = (long["g6"] >= 4).astype("int8")
    long = long.drop(columns=["g4", "g6"])

    # Opponent quality entering this game = opponent's own std_net_rtg row.
    opp_lut = long[["game_id", "team_id", "net_rtg_std"]].rename(columns={
        "team_id": "opp_id", "net_rtg_std": "opp_net_rtg_std"})
    long = long.merge(opp_lut, on=["game_id", "opp_id"], how="left")
    assert len(long) == n_in, "FATAL: opponent join changed row count"

    # Trailing SOS (mean opponent quality of last 7 faced) + adjusted rating.
    long = long.sort_values(["team_id", "season", "game_date", "game_id"]
                            ).reset_index(drop=True)
    g = long.groupby(["team_id", "season"], observed=True, sort=False)
    shifted_oq = g["opp_net_rtg_std"].shift(1)
    sos = (
        shifted_oq.groupby([long["team_id"], long["season"]], observed=True)
        .rolling(7, min_periods=MIN_GAMES_FOR_FEATURES).mean()
    )
    long["sos_r7"] = sos.droplevel([0, 1])  # index-aligned assignment
    # One-step SRS-style correction: quality ≈ own margin + schedule strength.
    long["adj_net_rtg_std"] = long["net_rtg_std"] + long["sos_r7"]

    long.to_parquet(PROCESSED_DIR / "team_features_long.parquet", index=False)
    return long


TEAM_FEATURE_COLS: list[str] = (
    [f"{c}_r{w}" for c in FORM_COLS for w in ROLLING_WINDOWS]
    + [f"{c}_std" for c in FORM_COLS]
    + ["rest_days", "is_b2b", "is_3in4", "is_4in6",
       "opp_net_rtg_std", "sos_r7", "adj_net_rtg_std"]
)
DIFF_COLS: tuple[str, ...] = (       # h_x - a_x deltas given to the model
    "net_rtg_r3", "net_rtg_r7", "net_rtg_std", "adj_net_rtg_std",
    "pace_r7", "efg_r7", "tov_rate_r7", "oreb_rate_r7", "ft_rate_r7",
    "rest_days", "sos_r7",
)


def build_game_matrix() -> pd.DataFrame:
    """Assemble the model design matrix: one row per game, home-perspective.

    X ∈ R^(n×d), targets: home_win ∈ {0,1}, margin_home ∈ Z\\{0}.
    """
    long = build_team_features()
    wide = pd.read_parquet(
        PROCESSED_DIR / "games_wide_full.parquet",
        columns=["game_id", "game_date", "season", "season_type",
                 "team_id_home", "team_id_away", "home_win", "margin_home"],
    )
    feat = long[["game_id", "team_id"] + TEAM_FEATURE_COLS]

    m = wide.merge(
        feat.rename(columns={"team_id": "team_id_home"}).rename(
            columns={c: f"h_{c}" for c in TEAM_FEATURE_COLS}),
        on=["game_id", "team_id_home"], how="left",
    ).merge(
        feat.rename(columns={"team_id": "team_id_away"}).rename(
            columns={c: f"a_{c}" for c in TEAM_FEATURE_COLS}),
        on=["game_id", "team_id_away"], how="left",
    )
    assert len(m) == len(wide), "FATAL: home/away feature join fan-out"

    for c in DIFF_COLS:
        m[f"d_{c}"] = m[f"h_{c}"] - m[f"a_{c}"]

    feature_cols = (
        [f"h_{c}" for c in TEAM_FEATURE_COLS]
        + [f"a_{c}" for c in TEAM_FEATURE_COLS]
        + [f"d_{c}" for c in DIFF_COLS]
    )

    # Null routing: rows lacking full windows (first MIN_GAMES per team-season)
    # are structurally incomplete -> dropped, bounded, and logged. Any OTHER
    # source of NaN would breach the bound and halt the pipeline.
    n_before = len(m)
    m = m.dropna(subset=feature_cols).reset_index(drop=True)
    drop_frac = 1.0 - len(m) / n_before
    assert drop_frac < 0.15, (
        f"FATAL: {drop_frac:.1%} rows dropped for missing features (bound 15%)"
    )
    print(f"[temporal] game matrix {m.shape}; dropped {drop_frac:.1%} "
          f"(early-season warmup rows)")

    m.to_parquet(PROCESSED_DIR / "features_games.parquet", index=False)
    (PROCESSED_DIR / "feature_cols_games.json").write_text(
        json.dumps(feature_cols, indent=1))
    return m


if __name__ == "__main__":
    build_game_matrix()
