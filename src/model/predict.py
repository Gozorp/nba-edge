"""
predict.py — Daily inference: tonight's slate -> P(home win), spread, fair odds,
plus next-game player PI projections.

State reconstruction
--------------------
Training features at game t are shift(1)-window stats over [.., t-1].
For an UPCOMING game, the analogous state is the unshifted tail of completed
games: mean of last w rows for *_rw, season mean for *_std, schedule-derived
fatigue vs. the slate date. Identical estimator, evaluated one step later —
the train/serve transformations are congruent by construction.

Fair odds transform (no vig):
    p >= 0.5 : american = -100 * p / (1 - p)
    p <  0.5 : american = +100 * (1 - p) / p

Usage:
    python -m src.model.predict --date 2026-07-08
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, SEASON_START_MONTH
from src.features.temporal import (
    DIFF_COLS, FORM_COLS, MAX_REST_DAYS, TEAM_FEATURE_COLS)
from src.ingest.bref_scraper import RateLimitedSession, _row_cells
from src.model.registry import load_model

PRED_DIR = PROCESSED_DIR / "predictions"
PRED_DIR.mkdir(exist_ok=True)


def fair_american_odds(p: float) -> int:
    assert 0.0 < p < 1.0, "FATAL: probability outside (0,1)"
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


# ----------------------------------------------------------------------------
# Slate discovery (unplayed rows on the month schedule page)
# ----------------------------------------------------------------------------
def fetch_slate(target: date) -> pd.DataFrame:
    season = target.year + 1 if target.month >= SEASON_START_MONTH else target.year
    month = target.strftime("%B").lower()
    sess = RateLimitedSession()
    html = sess.fetch(f"/leagues/NBA_{season}_games-{month}.html")
    assert html is not None, f"FATAL: no schedule page for {season}/{month}"
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="schedule")
    assert table is not None, "FATAL: schedule table missing"

    rows = []
    for tr in table.find_all("tr"):
        cells = _row_cells(tr)
        if "date_game" not in cells or cells["date_game"] in ("", "Date"):
            continue
        d = pd.to_datetime(cells["date_game"]).date()
        if d == target:
            rows.append({"visitor": cells["visitor_team_name"],
                         "home": cells["home_team_name"]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Team state vectors
# ----------------------------------------------------------------------------
def _team_state(tf: pd.DataFrame, team_id: int, asof: date) -> dict[str, float] | None:
    season = asof.year + 1 if asof.month >= SEASON_START_MONTH else asof.year
    h = tf[(tf["team_id"] == team_id) & (tf["season"] == season)
           & (tf["game_date"] < pd.Timestamp(asof))].sort_values("game_date")
    if len(h) < 3:                      # insufficient warmup — honest refusal
        return None

    s: dict[str, float] = {}
    for c in FORM_COLS:
        s[f"{c}_r3"] = float(h[c].tail(3).mean())
        s[f"{c}_r7"] = float(h[c].tail(7).mean())
        s[f"{c}_std"] = float(h[c].mean())
    last = h["game_date"].max().date()
    r = (asof - last).days - 1
    s["rest_days"] = float(min(max(r, 0), MAX_REST_DAYS))
    nights = h["game_date"].dt.date
    s["is_b2b"] = float(r == 0)
    s["is_3in4"] = float(sum(1 for d in nights if (asof - d).days <= 3) + 1 >= 3)
    s["is_4in6"] = float(sum(1 for d in nights if (asof - d).days <= 5) + 1 >= 4)
    s["sos_r7"] = float(h["opp_net_rtg_std"].tail(7).mean())
    s["adj_net_rtg_std"] = s["net_rtg_std"] + (
        0.0 if np.isnan(s["sos_r7"]) else s["sos_r7"])
    return s


def predict_games(target: date) -> pd.DataFrame:
    slate = fetch_slate(target)
    if slate.empty:
        print(f"[predict] no games on {target}")
        return slate

    xwalk = pd.read_parquet(PROCESSED_DIR / "team_crosswalk.parquet")
    name_to_id = dict(zip(xwalk["full_name"], xwalk["team_id"]))
    unknown = (set(slate["visitor"]) | set(slate["home"])) - set(name_to_id)
    assert not unknown, f"FATAL: unmapped team names: {sorted(unknown)}"

    tf = pd.read_parquet(PROCESSED_DIR / "team_features_long.parquet")
    clf, clf_meta = load_model("game_clf")
    reg, reg_meta = load_model("game_margin")
    cols: list[str] = clf_meta["feature_cols"]
    assert cols == reg_meta["feature_cols"], "FATAL: model schema divergence"

    out_rows = []
    for _, g in slate.iterrows():
        hid, aid = int(name_to_id[g["home"]]), int(name_to_id[g["visitor"]])
        hs, as_ = _team_state(tf, hid, target), _team_state(tf, aid, target)
        if hs is None or as_ is None:
            print(f"[predict] SKIP {g['visitor']} @ {g['home']} — warmup")
            continue
        hs["opp_net_rtg_std"] = as_["net_rtg_std"]
        as_["opp_net_rtg_std"] = hs["net_rtg_std"]

        feat = {f"h_{k}": v for k, v in hs.items() if k in TEAM_FEATURE_COLS}
        feat |= {f"a_{k}": v for k, v in as_.items() if k in TEAM_FEATURE_COLS}
        for c in DIFF_COLS:
            feat[f"d_{c}"] = feat[f"h_{c}"] - feat[f"a_{c}"]
        X = pd.DataFrame([feat])[cols].astype("float64")
        assert X.shape == (1, len(cols)), "FATAL: inference vector shape"

        dm = xgb.DMatrix(X, feature_names=cols)
        p_home = float(clf.predict(dm)[0])
        margin = float(reg.predict(dm)[0])
        out_rows.append({
            "date": str(target), "away": g["visitor"], "home": g["home"],
            "p_home_win": round(p_home, 4),
            "fair_ml_home": fair_american_odds(p_home),
            "fair_ml_away": fair_american_odds(1 - p_home),
            "pred_margin_home": round(margin, 2),
            "model_clf": clf_meta["version"], "model_margin": reg_meta["version"],
        })

    preds = pd.DataFrame(out_rows)
    if not preds.empty:
        out = PRED_DIR / f"games_{target}.csv"
        preds.to_csv(out, index=False)
        print(preds.to_string(index=False))
        print(f"[predict] written -> {out}")
    return preds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str,
                    default=str(date.today()), help="YYYY-MM-DD slate date")
    predict_games(date.fromisoformat(ap.parse_args().date))
