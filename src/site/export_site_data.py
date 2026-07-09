"""
export_site_data.py — Publish layer: pipeline artifacts -> docs/data/*.json
for the GitHub Pages terminal (mirrors the MLB-edges site architecture:
manifest.json dates + per-date pick files + health.json).

Outputs (docs/data/):
    manifest.json          {"dates": [...desc], "season": "...", "built_at": ...}
    picks_{date}.json      per-game: teams, model prob, fair ML, spread pick,
                           grade + tier, top SHAP factors, final result
    health.json            model versions/metrics/drift in MLB-health shape

Grade rubric (deterministic; published in the site glossary):
    conf = |p_home - 0.5| ; agree = (margin model side == prob model side)
    A  conf>=.20 & agree | A- conf>=.15 & agree | B+ conf>=.10 & agree
    B  conf>=.06        | B- conf>=.03         | C  conf<.03
    D  models disagree with conf>=.10 (conflicted signal)

Run after training:  python -m src.site.export_site_data
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, REPO_ROOT
from src.model.predict import fair_american_odds
from src.model.registry import load_model, metrics_history

OUT_DIR = REPO_ROOT / "docs" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FACTOR_LABELS: dict[str, str] = {
    "d_net_rtg_std": "Season net-rating edge",
    "d_adj_net_rtg_std": "SOS-adjusted quality edge",
    "d_net_rtg_r7": "Last-7 form edge",
    "d_net_rtg_r3": "Last-3 form edge",
    "d_rest_days": "Rest edge",
    "d_pace_r7": "Pace differential",
    "d_efg_r7": "Shooting-efficiency edge",
    "d_tov_rate_r7": "Turnover-rate edge",
    "d_oreb_rate_r7": "Off-rebounding edge",
    "d_ft_rate_r7": "FT-rate edge",
    "d_sos_r7": "Schedule-strength edge",
}


def _label(col: str) -> str:
    if col in FACTOR_LABELS:
        return FACTOR_LABELS[col]
    side = "Home" if col.startswith("h_") else "Away" if col.startswith("a_") else ""
    base = col[2:] if col[:2] in ("h_", "a_") else col
    names = {
        "rest_days": "rest days", "is_b2b": "back-to-back", "is_3in4": "3-in-4",
        "is_4in6": "4-in-6", "sos_r7": "recent schedule", "opp_net_rtg_std":
        "opponent quality faced", "adj_net_rtg_std": "adjusted quality",
        "net_rtg_std": "season net rating", "net_rtg_r7": "last-7 net rating",
        "net_rtg_r3": "last-3 net rating", "off_rtg_std": "season offense",
        "def_rtg_std": "season defense", "pace_std": "season pace",
        "pace_r7": "recent pace", "efg_std": "season shooting",
        "efg_r7": "recent shooting", "efg_r3": "recent shooting (3g)",
        "tov_rate_std": "turnover rate", "tov_rate_r7": "recent turnovers",
        "tov_rate_r3": "recent turnovers (3g)",
        "oreb_rate_std": "off-rebound rate", "oreb_rate_r7": "recent o-boards",
        "oreb_rate_r3": "recent o-boards (3g)",
        "ft_rate_std": "FT rate", "ft_rate_r7": "recent FT rate",
        "ft_rate_r3": "recent FT rate (3g)",
        "fg3_share_std": "3-point profile", "fg3_share_r7": "recent 3P profile",
        "fg3_share_r3": "recent 3P profile (3g)",
        "off_rtg_r7": "recent offense", "off_rtg_r3": "recent offense (3g)",
        "def_rtg_r7": "recent defense", "def_rtg_r3": "recent defense (3g)",
    }
    return f"{side} {names.get(base, base)}".strip()


def grade_for(p_home: float, margin: float) -> tuple[str, str]:
    conf = abs(p_home - 0.5)
    agree = (margin > 0) == (p_home >= 0.5)
    if not agree and conf >= 0.10:
        return "D", "CONFLICT"
    if agree and conf >= 0.20:
        return "A", "PLATINUM"
    if agree and conf >= 0.15:
        return "A-", "GOLD"
    if agree and conf >= 0.10:
        return "B+", "SILVER"
    if conf >= 0.06:
        return "B", "BRONZE"
    if conf >= 0.03:
        return "B-", "LEAN"
    return "C", "COINFLIP"


def export(season: int = 2026) -> None:
    clf, meta = load_model("game_clf")
    reg, rmeta = load_model("game_margin")
    cols = meta["feature_cols"]

    df = pd.read_parquet(PROCESSED_DIR / "features_games.parquet")
    df = df[df["season"] == season].sort_values(["game_date", "game_id"])
    assert len(df) > 0, "FATAL: no games for season"
    xwalk = pd.read_parquet(PROCESSED_DIR / "team_crosswalk.parquet")
    id2name = dict(zip(xwalk["team_id"].astype(int), xwalk["full_name"]))
    id2abbr = dict(zip(xwalk["team_id"].astype(int), xwalk["bref_abbr"]))

    X = df[cols].astype("float64")
    dm = xgb.DMatrix(X, feature_names=cols)
    p = clf.predict(dm)
    mar = reg.predict(dm)
    contribs = clf.predict(dm, pred_contribs=True)[:, :-1]

    df = df.reset_index(drop=True)
    dates: list[str] = []
    for date, idx in df.groupby(df["game_date"].dt.strftime("%Y-%m-%d")).groups.items():
        rows = []
        for i in idx:
            r = df.loc[i]
            pi_, mi_ = float(p[i]), float(mar[i])
            g, tier = grade_for(pi_, mi_)
            # local-slope conversion: phi (log-odds) -> approx prob points
            slope = pi_ * (1 - pi_) * 100.0
            phi = contribs[i]
            top = np.argsort(-np.abs(phi))[:5]
            factors = [{
                "label": _label(cols[j]),
                "impact_pp": round(float(phi[j]) * slope, 1),
            } for j in top if abs(phi[j]) > 1e-6]
            hid, aid = int(r["team_id_home"]), int(r["team_id_away"])
            pick_home = pi_ >= 0.5
            rows.append({
                "game_id": str(r["game_id"]),
                "away": id2name[aid], "home": id2name[hid],
                "away_abbr": id2abbr[aid], "home_abbr": id2abbr[hid],
                "p_home": round(pi_, 4),
                "pick": id2abbr[hid] if pick_home else id2abbr[aid],
                "pick_prob": round(pi_ if pick_home else 1 - pi_, 4),
                "fair_ml_home": fair_american_odds(pi_),
                "fair_ml_away": fair_american_odds(1 - pi_),
                "pred_margin_home": round(mi_, 1),
                "grade": g, "tier": tier,
                "season_type": str(r["season_type"]),
                "factors": factors,
                "result": {
                    "home_win": int(r["home_win"]),
                    "margin_home": int(r["margin_home"]),
                    "pick_correct": int(pick_home == bool(r["home_win"])),
                },
            })
        (OUT_DIR / f"picks_{date}.json").write_text(
            json.dumps({"date": date, "games": rows}, indent=0))
        dates.append(date)

    dates.sort(reverse=True)
    picks_ok = int(((p >= 0.5) == df["home_win"].to_numpy(bool)).sum())
    grades_all = [grade_for(float(pi_), float(mi_))[0]
                  for pi_, mi_ in zip(p, mar)]
    correct_all = ((p >= 0.5) == df["home_win"].to_numpy(bool))
    gtab = {}
    for g in ("A", "A-", "B+", "B", "B-", "C", "D"):
        mask = np.array([x == g for x in grades_all])
        if mask.any():
            gtab[g] = {"n": int(mask.sum()),
                       "hit": round(float(correct_all[mask].mean()), 3)}
    (OUT_DIR / "manifest.json").write_text(json.dumps({
        "dates": dates, "season": f"{season-1}-{str(season)[2:]}",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "archive",   # flips to "live" in-season via daily loop
        "record": {"games": int(len(df)), "correct": picks_ok,
                   "acc": round(picks_ok / len(df), 4), "grades": gtab},
    }, indent=1))

    # ---- health.json (MLB-health shape) ------------------------------------
    from src.recalibrate.drift import full_report
    rep = full_report("game_clf")
    acc = float(np.mean((p >= 0.5) == df["home_win"].to_numpy(bool)))
    checks = [
        {"name": "model_versions", "severity": "green",
         "message": f"clf {meta['version']} / margin {rmeta['version']} promoted",
         "category": "model"},
        {"name": "oos_accuracy", "severity": "green" if acc >= 0.62 else "yellow",
         "message": f"{season-1}-{str(season)[2:]} out-of-sample accuracy {acc:.1%}",
         "category": "model"},
        {"name": "data_drift",
         "severity": "green" if not rep["feature_drift"]["data_drift"] else "yellow",
         "message": f"max PSI {rep['feature_drift']['max_psi']:.3f} "
                    f"({len(rep['feature_drift']['critical_features'])} critical)",
         "category": "data_flow"},
        {"name": "concept_drift",
         "severity": "green" if not rep["performance_drift"].get("concept_drift")
                     else "red",
         "message": rep["performance_drift"].get(
             "reason", f"z={rep['performance_drift'].get('z')}"),
         "category": "model"},
    ]
    sev = ("red" if any(c["severity"] == "red" for c in checks)
           else "yellow" if any(c["severity"] == "yellow" for c in checks)
           else "green")
    (OUT_DIR / "health.json").write_text(json.dumps({
        "version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "overall": sev,
        "categories": {"model": sev, "data_flow": "green", "deployment": "green"},
        "checks": checks,
        "holdout_metrics": meta["holdout_metrics"],
    }, indent=1, default=str))
    print(f"[site] exported {len(dates)} slates, health={sev}, acc={acc:.3f}")


if __name__ == "__main__":
    export()
