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
    if html is None:                    # offseason: month page absent
        print(f"[predict] no schedule page for {season}/{month} — skipping")
        return pd.DataFrame()
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
    """State entering `asof` (unshifted tails). Extra keys prefixed '_' carry
    conviction inputs (games played, rest, b2b) — excluded from model features."""

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
    s["_games"] = float(len(h))
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
    # serve with the same tree count every validated metric used
    clf_ir = ((0, int(clf_meta["best_iteration"]) + 1)
              if clf_meta.get("best_iteration") is not None else (0, 0))
    reg_ir = ((0, int(reg_meta["best_iteration"]) + 1)
              if reg_meta.get("best_iteration") is not None else (0, 0))

    out_rows = []
    site_inputs = []
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
        p_home = float(clf.predict(dm, iteration_range=clf_ir)[0])
        margin = float(reg.predict(dm, iteration_range=reg_ir)[0])
        out_rows.append({
            "date": str(target), "away": g["visitor"], "home": g["home"],
            "p_home_win": round(p_home, 4),
            "fair_ml_home": fair_american_odds(p_home),
            "fair_ml_away": fair_american_odds(1 - p_home),
            "pred_margin_home": round(margin, 2),
            "model_clf": clf_meta["version"], "model_margin": reg_meta["version"],
        })
        site_inputs.append((g, hid, aid, hs, as_, X, dm, p_home, margin))

    preds = pd.DataFrame(out_rows)
    if not preds.empty:
        out = PRED_DIR / f"games_{target}.csv"
        preds.to_csv(out, index=False)
        print(preds.to_string(index=False))
        print(f"[predict] written -> {out}")
        _export_site_picks(target, site_inputs, clf, cols, xwalk)
    return preds


def _export_site_picks(target: date, site_inputs: list, clf, cols, xwalk) -> None:
    """Publish today's slate to the terminal (result: null until the live
    layer verifies it client-side and the next daily export archives it)."""
    import json as _json

    from src.model.conviction import score as conviction_score
    from src.site.export_site_data import _label, _write_atomic, grade_for

    id2abbr = dict(zip(xwalk["team_id"].astype(int), xwalk["bref_abbr"]))
    games = []
    for g, hid, aid, hs, as_, X, dm, p_home, margin in site_inputs:
        contribs = clf.predict(dm, pred_contribs=True)[0, :-1]
        slope = p_home * (1 - p_home) * 100.0
        top = np.argsort(-np.abs(contribs))[:5]
        factors = [{"label": _label(cols[j]),
                    "impact_pp": round(float(contribs[j]) * slope, 1)}
                   for j in top if abs(contribs[j]) > 1e-6]
        pick_home = p_home >= 0.5
        gr, _ = grade_for(p_home, margin)
        conv = conviction_score(dict(
            pick_is_home=pick_home, p_pick=p_home if pick_home else 1 - p_home,
            d_net_rtg_std=float(X["d_net_rtg_std"].iloc[0]),
            d_net_rtg_r7=float(X["d_net_rtg_r7"].iloc[0]),
            games_h=int(hs["_games"]), games_a=int(as_["_games"]),
            h_rest=hs["rest_days"], a_rest=as_["rest_days"],
            h_b2b=bool(hs["is_b2b"]), a_b2b=bool(as_["is_b2b"]),
            pred_margin_home=margin))
        games.append({
            "game_id": f"{target.isoformat()}_{id2abbr[aid]}@{id2abbr[hid]}",
            "away": g["visitor"], "home": g["home"],
            "away_abbr": id2abbr[aid], "home_abbr": id2abbr[hid],
            "p_home": round(p_home, 4),
            "pick": id2abbr[hid] if pick_home else id2abbr[aid],
            "pick_prob": round(p_home if pick_home else 1 - p_home, 4),
            "fair_ml_home": fair_american_odds(p_home),
            "fair_ml_away": fair_american_odds(1 - p_home),
            "pred_margin_home": round(margin, 1),
            "grade": gr, "tier": conv.tier, "signals": conv.signals,
            "why_skipped": conv.why_skipped, "stake_frac": conv.stake_frac,
            "edge_pp": conv.edge_pp, "ev_per_dollar": conv.ev_per_dollar,
            "season_type": "Regular Season", "factors": factors,
            "result": None,
        })
    if not games:
        return
    from config import REPO_ROOT
    ddir = REPO_ROOT / "docs" / "data"
    iso = target.isoformat()
    _write_atomic(ddir / f"picks_{iso}.json",
                  _json.dumps({"date": iso, "games": games}, indent=0))
    mpath = ddir / "manifest.json"
    m = _json.loads(mpath.read_text())
    if iso not in m["dates"]:
        m["dates"] = sorted(set(m["dates"]) | {iso}, reverse=True)
    m["mode"] = "live"
    _write_atomic(mpath, _json.dumps(m, indent=1))
    print(f"[predict] site slate published: picks_{iso}.json ({len(games)} games)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=str,
                    default=str(date.today()), help="YYYY-MM-DD slate date")
    predict_games(date.fromisoformat(ap.parse_args().date))
