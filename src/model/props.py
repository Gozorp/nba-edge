"""
props.py — Player prop projections (PTS / REB / AST), honestly split.

The player matrix spans 2023-10 -> 2026-06, so the site's out-of-sample bar
(2025-26 untouched by training) is enforced HERE, not by fraction holdout:

    train : game_date <  2025-08-01   (seasons 2023-24, 2024-25)
    eval  : game_date >= 2025-08-01   (season 2025-26 — published on site)

Each target reports MAE against the naive baseline (trailing 7-game mean):
a prop model earns its place only by beating the number any bettor could
compute in their head. Models are versioned via the registry like the rest.

Usage:
    python -m src.model.props            # train + eval + export site JSONs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import EARLY_STOPPING_ROUNDS, PROCESSED_DIR, REPO_ROOT, XGB_SEED

SPLIT = pd.Timestamp("2025-08-01")
TARGETS = {"player_pts": "pts", "player_reb": "reb", "player_ast": "ast"}
OUT_DIR = REPO_ROOT / "docs" / "data"

PARAMS = {
    "objective": "reg:squarederror", "eval_metric": ["mae"],
    "eta": 0.05, "max_depth": 5, "min_child_weight": 10,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "tree_method": "hist", "seed": XGB_SEED,
}


def _load() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(PROCESSED_DIR / "features_players.parquet")
    cols = json.loads((PROCESSED_DIR / "feature_cols_players.json").read_text())
    df = df.sort_values("game_date").reset_index(drop=True)
    assert np.isfinite(df[cols].to_numpy()).all(), "FATAL: non-finite features"
    return df, cols


def train_props() -> dict:
    from src.model.registry import save_model

    df, cols = _load()
    tr = df[df["game_date"] < SPLIT]
    ev = df[df["game_date"] >= SPLIT]
    assert len(tr) > 20_000 and len(ev) > 10_000, (
        f"FATAL: bad split sizes train={len(tr)} eval={len(ev)}"
    )
    # early-stopping slice = chronological tail of TRAIN (never touches eval)
    cut = int(len(tr) * 0.9)

    report: dict = {"n_train": int(len(tr)), "n_eval": int(len(ev)),
                    "split": str(SPLIT.date()), "mae": {}, "baseline_mae": {}}
    preds: dict[str, np.ndarray] = {}
    for kind, tgt in TARGETS.items():
        dtr = xgb.DMatrix(tr[cols].iloc[:cut], label=tr[tgt].iloc[:cut],
                          feature_names=cols)
        dva = xgb.DMatrix(tr[cols].iloc[cut:], label=tr[tgt].iloc[cut:],
                          feature_names=cols)
        dev = xgb.DMatrix(ev[cols], feature_names=cols)
        bst = xgb.train(PARAMS, dtr, num_boost_round=2000,
                        evals=[(dva, "va")],
                        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                        verbose_eval=False)
        p = bst.predict(dev, iteration_range=(0, bst.best_iteration + 1))
        y = ev[tgt].to_numpy(np.float64)
        mae = float(np.abs(p - y).mean())
        base = float(np.abs(ev[f"{tgt}_r7"].to_numpy(np.float64) - y).mean())
        assert mae < base, f"FATAL: {kind} does not beat trailing-mean baseline"
        report["mae"][tgt] = round(mae, 3)
        report["baseline_mae"][tgt] = round(base, 3)
        preds[tgt] = p
        save_model(bst, {
            "kind": kind, "params": PARAMS, "feature_cols": cols,
            "n_train": int(cut), "best_iteration": int(bst.best_iteration),
            "train_span": [str(tr["game_date"].iloc[0].date()),
                           str(tr["game_date"].iloc[cut - 1].date())],
            "holdout_span": [str(ev["game_date"].iloc[0].date()),
                             str(ev["game_date"].iloc[-1].date())],
            "holdout_metrics": {"mae": report["mae"][tgt],
                                "baseline_mae": report["baseline_mae"][tgt]},
        })
        print(f"[props:{tgt}] eval MAE {mae:.3f} vs baseline {base:.3f} "
              f"(iter {bst.best_iteration})")

    # ---- site export: per-date, per-game projections vs actuals ------------
    out = ev[["game_date", "bref_game_id", "player", "abbr", "mp",
              "pts", "reb", "ast"]].copy()
    for tgt in TARGETS.values():
        out[f"proj_{tgt}"] = np.round(preds[tgt], 1)
    n_files = 0
    for iso, day in out.groupby(out["game_date"].dt.strftime("%Y-%m-%d")):
        games: dict[str, list] = {}
        for gid, gframe in day.groupby("bref_game_id"):
            gframe = gframe.sort_values("proj_pts", ascending=False).head(16)
            games[str(gid)] = [{
                "player": r["player"], "abbr": r["abbr"],
                "mp": round(float(r["mp"]), 1),
                "proj": {"pts": float(r["proj_pts"]), "reb": float(r["proj_reb"]),
                         "ast": float(r["proj_ast"])},
                "actual": {"pts": int(r["pts"]), "reb": int(r["reb"]),
                           "ast": int(r["ast"])},
            } for _, r in gframe.iterrows()]
        (OUT_DIR / f"props_{iso}.json").write_text(
            json.dumps({"date": iso, "games": games}, indent=0))
        n_files += 1

    mpath = OUT_DIR / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["props"] = report
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"[props] {n_files} prop files; manifest updated")
    return report


if __name__ == "__main__":
    train_props()
