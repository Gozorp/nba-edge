"""
performance_index.py — Phase 2c: data-driven player impact target (PI).

Definition (no hand-picked weights)
-----------------------------------
Weights are LEARNED, not asserted. At the team-game level we fit

    net_rtg = w · x + b + eps ,   x = team box events per 100 possessions
    x = [ast, oreb, dreb, stl, blk, tov, fga, fta, fg3m, ftm, fgm, pf]

by ridge regression (L2; alpha chosen by generalized CV over a fixed grid).
The coefficient vector w prices each box event in net-margin units — scoring
events price offense, while stl/blk/dreb/pf must carry the defensive load,
which is exactly how defense enters the index. Note pts is EXCLUDED from x:
pts/100poss is (nearly) the identity component of net_rtg and would collapse
the regression onto a trivial solution; the make/attempt columns (fgm, fg3m,
ftm, fga, fta) span scoring efficiency without the degeneracy.

Weights are fit ONLY on the archive era (game_date <= ARCHIVE_MAX_DATE) and
frozen — the PI definition never sees post-boundary data, so downstream
models trained on scraped-era PI carry no future information by construction.

Player application
------------------
    f: R^12 -> R,   PI_g(player) = w · x_g(player)
    x_g(player) = per-100-exposure stats, exposure = team_poss * 5*mp/team_mp
    PI_off = w_off · x_off (ast, oreb, tov, fga, fta, fg3m, ftm, fgm)
    PI_def = w_def · x_def (dreb, stl, blk, pf)

Eligibility: mp >= 8.0 (≈ 17 possessions at league pace; below this the
standard error of a per-100 rate exceeds the cross-player signal s.d.).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import ARCHIVE_MAX_DATE, PROCESSED_DIR, RAW_DIR

PI_EVENT_COLS: tuple[str, ...] = (
    "ast", "oreb", "dreb", "stl", "blk", "tov",
    "fga", "fta", "fg3m", "ftm", "fgm", "pf",
)
_DEF_COLS: frozenset[str] = frozenset({"dreb", "stl", "blk", "pf"})
MIN_MP_FOR_PI: float = 8.0
RIDGE_ALPHAS: np.ndarray = np.logspace(-2, 3, 30)


def learn_event_weights() -> dict[str, float]:
    """Fit w on archive-era team-games; persist to JSON; return mapping."""
    long = pd.read_parquet(PROCESSED_DIR / "games_long_full.parquet")
    train = long[long["game_date"] <= pd.Timestamp(ARCHIVE_MAX_DATE)].copy()
    assert len(train) >= 50_000, (
        f"FATAL: only {len(train)} archive team-games for weight learning"
    )

    X = pd.DataFrame({c: 100.0 * train[c] / train["poss"]
                      for c in PI_EVENT_COLS}).to_numpy(np.float64)
    y = train["net_rtg"].to_numpy(np.float64)
    assert X.shape == (len(train), len(PI_EVENT_COLS)), "FATAL: X dims"
    assert np.isfinite(X).all() and np.isfinite(y).all(), "FATAL: non-finite"

    model = RidgeCV(alphas=RIDGE_ALPHAS).fit(X, y)
    r2 = float(model.score(X, y))
    assert r2 >= 0.45, f"FATAL: weight regression R^2={r2:.3f} < 0.45"

    w = {c: float(k) for c, k in zip(PI_EVENT_COLS, model.coef_)}
    payload = {"weights": w, "intercept": float(model.intercept_),
               "alpha": float(model.alpha_), "r2_train": r2,
               "fit_boundary": ARCHIVE_MAX_DATE, "n": int(len(train))}
    (PROCESSED_DIR / "pi_weights.json").write_text(json.dumps(payload, indent=1))
    print(f"[PI] ridge alpha={model.alpha_:.3g} R^2={r2:.3f} weights={w}")
    return w


def load_or_learn_weights() -> dict[str, float]:
    p = PROCESSED_DIR / "pi_weights.json"
    if p.exists():
        return json.loads(p.read_text())["weights"]   # frozen definition
    return learn_event_weights()


def compute_player_pi() -> pd.DataFrame:
    """Score every scraped player-game. Output: player_pi.parquet."""
    w = load_or_learn_weights()
    pb = pd.read_parquet(RAW_DIR / "player_box.parquet")
    tb = pd.read_parquet(RAW_DIR / "team_box.parquet")

    # Team possessions + team minutes for exposure scaling.
    tb = tb.copy()
    tb["team_poss"] = tb["fga"] + 0.44 * tb["fta"] - tb["oreb"] + tb["tov"]
    assert (tb["team_poss"] > 0).all(), "FATAL: non-positive team possessions"
    tmap = tb.set_index(["bref_game_id", "abbr"])[["team_poss", "mp"]]

    pb = pb.join(tmap, on=["bref_game_id", "abbr"], rsuffix="_team")
    pb = pb.rename(columns={"mp_team": "team_mp"})
    assert pb["team_poss"].notna().all(), "FATAL: player rows without team totals"

    pb = pb[pb["mp"] >= MIN_MP_FOR_PI].copy()
    # Exposure: possessions witnessed = team_poss * (share of floor time).
    # team_mp sums 5 on-floor slots => floor share = 5*mp/team_mp in [0,1].
    share = 5.0 * pb["mp"] / pb["team_mp"]
    assert ((share > 0) & (share <= 1.0 + 1e-9)).all(), "FATAL: share out of [0,1]"
    pb["exposure_poss"] = pb["team_poss"] * share

    counting = list(PI_EVENT_COLS)
    pb[counting] = pb[counting].fillna(0.0)          # true zeros (source omits)

    pi = np.zeros(len(pb))
    pi_off = np.zeros(len(pb))
    pi_def = np.zeros(len(pb))
    for c in PI_EVENT_COLS:
        per100 = 100.0 * pb[c].to_numpy() / pb["exposure_poss"].to_numpy()
        contrib = w[c] * per100
        pi += contrib
        if c in _DEF_COLS:
            pi_def += contrib
        else:
            pi_off += contrib
    pb["pi"] = pi
    pb["pi_off"] = pi_off
    pb["pi_def"] = pi_def
    assert np.isfinite(pb["pi"]).all(), "FATAL: non-finite PI"

    pb["reb"] = pb["oreb"] + pb["dreb"]           # prop targets
    out_cols = ["bref_game_id", "player_id", "player", "abbr", "mp",
                "exposure_poss", "pi", "pi_off", "pi_def", "pts", "reb",
                "ast", "stl", "blk", "fg3m", "usg_pct", "ts_pct"]
    out = pb[out_cols].copy()
    out.to_parquet(PROCESSED_DIR / "player_pi.parquet", index=False)
    print(f"[PI] scored {len(out)} player-games "
          f"(mean={out['pi'].mean():.2f}, sd={out['pi'].std():.2f})")
    return out


def build_player_matrix() -> pd.DataFrame:
    """Trailing-form player feature matrix; target = current-game PI.

    Features: rolling PI/mp/usg/ts (3, 7, season-to-date), rest days,
    is_home, opponent defensive quality entering the game, archetype id.
    """
    from src.features.temporal import _rest_days, add_rolling  # shared prims

    pi = pd.read_parquet(PROCESSED_DIR / "player_pi.parquet")
    sched = pd.read_parquet(RAW_DIR / "schedule.parquet",
                            columns=["bref_game_id", "game_date"])
    xwalk = pd.read_parquet(PROCESSED_DIR / "team_crosswalk.parquet")

    df = pi.merge(sched, on="bref_game_id", how="left")
    df["game_date"] = pd.to_datetime(df["game_date"])
    assert df["game_date"].notna().all(), "FATAL: unscheduled player rows"
    df["season"] = (df["game_date"].dt.year
                    + (df["game_date"].dt.month >= 8)).astype("Int32")
    df["is_home"] = (df["bref_game_id"].str.slice(-3) == df["abbr"]).astype("int8")

    # Opponent defensive quality entering the game (leakage-free: _std cols
    # in team_features_long are shift(1)-expanding by construction).
    tf = pd.read_parquet(PROCESSED_DIR / "team_features_long.parquet",
                         columns=["game_id", "team_id", "def_rtg_std",
                                  "net_rtg_std", "pace_r7"])
    df["team_id"] = df["abbr"].map(
        xwalk.set_index("bref_abbr")["team_id"]).astype("Int64")
    pair = tf[tf["game_id"].isin(set(df["bref_game_id"]))]
    opp = df[["bref_game_id", "team_id"]].merge(
        pair.rename(columns={"game_id": "bref_game_id"}),
        on="bref_game_id", how="left", suffixes=("", "_row"))
    opp = opp[opp["team_id_row"] != opp["team_id"]]   # keep the OTHER team
    opp = opp.drop_duplicates(subset=["bref_game_id", "team_id"]).rename(
        columns={"def_rtg_std": "opp_def_rtg_std",
                 "net_rtg_std": "opp_net_rtg_std", "pace_r7": "opp_pace_r7"})
    df = df.merge(
        opp[["bref_game_id", "team_id", "opp_def_rtg_std",
             "opp_net_rtg_std", "opp_pace_r7"]],
        on=["bref_game_id", "team_id"], how="left")

    df = df.sort_values(["player_id", "season", "game_date"]).reset_index(drop=True)
    df = add_rolling(df, ["player_id", "season"],
                     ["pi", "pi_off", "pi_def", "mp", "usg_pct", "ts_pct",
                      "pts", "reb", "ast", "stl", "blk", "fg3m"])
    df["rest_days"] = df.groupby(["player_id", "season"], observed=True)[
        "game_date"].transform(_rest_days)

    arch = pd.read_parquet(PROCESSED_DIR / "player_archetypes.parquet",
                           columns=["player_id", "season", "cluster"])
    df = df.merge(arch, on=["player_id", "season"], how="left")
    df["cluster"] = df["cluster"].fillna(-1).astype("int16")  # sub-500-MP tier

    feature_cols = (
        [f"{c}_r{w}" for c in ("pi", "pi_off", "pi_def", "mp", "usg_pct",
                               "ts_pct", "pts", "reb", "ast",
                               "stl", "blk", "fg3m") for w in (3, 7)]
        + [f"{c}_std" for c in ("pi", "mp", "usg_pct", "ts_pct", "pts",
                                "reb", "ast", "stl", "blk", "fg3m")]
        + ["rest_days", "is_home", "opp_def_rtg_std", "opp_net_rtg_std",
           "opp_pace_r7", "cluster"]
    )
    n_before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    assert len(df) > 0, "FATAL: empty player matrix"
    print(f"[PI] player matrix {df.shape} "
          f"(dropped {1 - len(df)/n_before:.1%} warmup rows)")

    df.to_parquet(PROCESSED_DIR / "features_players.parquet", index=False)
    (PROCESSED_DIR / "feature_cols_players.json").write_text(
        json.dumps(feature_cols, indent=1))
    return df


if __name__ == "__main__":
    learn_event_weights()
    compute_player_pi()
    build_player_matrix()
