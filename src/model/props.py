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
TARGETS = {"player_pts": "pts", "player_reb": "reb", "player_ast": "ast",
           "player_stl": "stl", "player_blk": "blk", "player_fg3m": "fg3m"}
OUT_DIR = REPO_ROOT / "docs" / "data"

PARAMS = {
    "objective": "reg:squarederror", "eval_metric": ["mae"],
    "eta": 0.05, "max_depth": 5, "min_child_weight": 10,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "tree_method": "hist", "seed": XGB_SEED,
}
# Sparse counting stats are Poisson-distributed; squared error over-smooths
# them. Per-target objective overrides:
TARGET_PARAMS = {"stl": {"objective": "count:poisson"},
                 "blk": {"objective": "count:poisson"},
                 "fg3m": {"objective": "count:poisson"}}


def _load() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(PROCESSED_DIR / "features_players.parquet")
    cols = json.loads((PROCESSED_DIR / "feature_cols_players.json").read_text())
    df = df.sort_values("game_date").reset_index(drop=True)
    assert np.isfinite(df[cols].to_numpy()).all(), "FATAL: non-finite features"
    return df, cols


def train_props(only: list[str] | None = None) -> dict:
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
                    "split": str(SPLIT.date()), "mae": {}, "baseline_mae": {},
                    "skipped": []}
    preds: dict[str, np.ndarray] = {}
    todo = {k: t for k, t in TARGETS.items() if only is None or t in only}
    for kind, tgt in todo.items():
        dtr = xgb.DMatrix(tr[cols].iloc[:cut], label=tr[tgt].iloc[:cut],
                          feature_names=cols)
        dva = xgb.DMatrix(tr[cols].iloc[cut:], label=tr[tgt].iloc[cut:],
                          feature_names=cols)
        dev = xgb.DMatrix(ev[cols], feature_names=cols)
        params = PARAMS | TARGET_PARAMS.get(tgt, {})
        bst = xgb.train(params, dtr, num_boost_round=2000,
                        evals=[(dva, "va")],
                        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                        verbose_eval=False)
        p = bst.predict(dev, iteration_range=(0, bst.best_iteration + 1))
        y = ev[tgt].to_numpy(np.float64)
        mae = float(np.abs(p - y).mean())
        base = float(np.abs(ev[f"{tgt}_r7"].to_numpy(np.float64) - y).mean())
        if mae >= base:
            # the ship-gate: a market that cannot beat the naive trailing
            # average is excluded, loudly, rather than dressed up.
            report["skipped"].append({"target": tgt, "mae": round(mae, 3),
                                      "baseline_mae": round(base, 3)})
            print(f"[props:{tgt}] EXCLUDED — MAE {mae:.3f} >= baseline {base:.3f}")
            continue
        report["mae"][tgt] = round(mae, 3)
        report["baseline_mae"][tgt] = round(base, 3)
        preds[tgt] = p
        save_model(bst, {
            "kind": kind, "params": params, "feature_cols": cols,
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

    return report


def export_props() -> None:
    """Write props_{date}.json for the eval season from PROMOTED models.

    Markets whose model was gate-excluded (couldn't beat the trailing-7
    baseline) are published WITH the baseline itself as the projection and
    flagged in manifest.props.method — an honest number beats a missing one.
    """
    from src.model.registry import current_version, load_model

    df, cols = _load()
    ev = df[df["game_date"] >= SPLIT]
    method: dict[str, str] = {}
    proj: dict[str, np.ndarray] = {}
    dev = xgb.DMatrix(ev[cols], feature_names=cols)
    for kind, tgt in TARGETS.items():
        if current_version(kind):
            bst, meta = load_model(kind)
            proj[tgt] = bst.predict(
                dev, iteration_range=(0, meta.get("best_iteration", 0) + 1))
            method[tgt] = "model"
        else:
            proj[tgt] = ev[f"{tgt}_r7"].to_numpy(np.float64)
            method[tgt] = "baseline_r7"

    out = ev[["game_date", "bref_game_id", "player", "abbr", "mp"]
             + list(TARGETS.values())].copy()
    for tgt in TARGETS.values():
        out[f"proj_{tgt}"] = np.round(np.clip(proj[tgt], 0, None), 1)

    # FULL ROSTER: every player in the box score ships. Players without a
    # model row (below the 8-min PI floor, or without 3 prior games of
    # history) appear with actuals and an explicit no-projection reason —
    # named exclusion, never silent omission.
    from config import RAW_DIR
    from src.features.performance_index import MIN_MP_FOR_PI
    pb = pd.read_parquet(RAW_DIR / "player_box.parquet",
                         columns=["bref_game_id", "player_id", "player",
                                  "abbr", "mp", "pts", "oreb", "dreb",
                                  "ast", "stl", "blk", "fg3m"])
    eval_gids = set(out["bref_game_id"])
    pb = pb[pb["bref_game_id"].isin(eval_gids)].copy()
    counting = ["pts", "oreb", "dreb", "ast", "stl", "blk", "fg3m"]
    pb[counting] = pb[counting].fillna(0.0)
    pb["reb"] = pb["oreb"] + pb["dreb"]
    modeled = set(zip(out["bref_game_id"], out["player"]))
    extras = pb[~pb.apply(lambda r: (r["bref_game_id"], r["player"]) in modeled,
                          axis=1)].copy()
    extras["reason"] = np.where(extras["mp"] < MIN_MP_FOR_PI,
                                "below_floor", "insufficient_history")
    gid2date = dict(zip(out["bref_game_id"],
                        out["game_date"].dt.strftime("%Y-%m-%d")))

    n_files = 0
    extras_by = {k: v for k, v in extras.groupby("bref_game_id")}
    for iso, day in out.groupby(out["game_date"].dt.strftime("%Y-%m-%d")):
        games: dict[str, list] = {}
        for gid, gframe in day.groupby("bref_game_id"):
            gframe = gframe.sort_values("proj_pts", ascending=False)
            rows = [{
                "player": r["player"], "abbr": r["abbr"],
                "mp": round(float(r["mp"]), 1),
                "proj": {t: float(r[f"proj_{t}"]) for t in TARGETS.values()},
                "actual": {t: int(r[t]) for t in TARGETS.values()},
            } for _, r in gframe.iterrows()]
            for _, r in extras_by.get(gid, pd.DataFrame()).iterrows():
                rows.append({
                    "player": r["player"], "abbr": r["abbr"],
                    "mp": round(float(r["mp"]), 1),
                    "proj": None, "reason": str(r["reason"]),
                    "actual": {t: int(r[t]) for t in TARGETS.values()},
                })
            games[str(gid)] = rows
        (OUT_DIR / f"props_{iso}.json").write_text(
            json.dumps({"date": iso, "games": games}, indent=0))
        n_files += 1

    mpath = OUT_DIR / "manifest.json"
    manifest = json.loads(mpath.read_text())
    props_block = manifest.get("props", {})
    props_block["method"] = method
    # refresh mae blocks from the promoted models' stored holdout metrics
    mae, base = {}, {}
    for kind, tgt in TARGETS.items():
        if current_version(kind):
            _, meta = load_model(kind)
            hm = meta.get("holdout_metrics", {})
            if "mae" in hm:
                mae[tgt] = hm["mae"]; base[tgt] = hm.get("baseline_mae")
    props_block["mae"], props_block["baseline_mae"] = mae, base
    manifest["props"] = props_block
    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"[props] exported {n_files} files; methods={method}")


if __name__ == "__main__":
    train_props()
    export_props()
