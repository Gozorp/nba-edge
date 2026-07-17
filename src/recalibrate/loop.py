"""
loop.py — Phase 4b: the daily continuous-learning loop.

Sequence (run_daily.py entrypoint):
    1. INGEST     scrape yesterday's box scores; rebuild unified store +
                  feature matrices (full deterministic rebuild — idempotent,
                  seconds at this scale; no incremental-state bugs possible).
    2. MONITOR    drift.full_report(): PSI (data drift) + Brier z-test
                  (concept drift) against the promoted champion.
    3. RETRAIN    gated. Fires iff
                     (retrain_signal OR n_new >= MIN_NEW_GAMES_RETRAIN)
                  The n_new floor is the anti-overfit guard: retraining on
                  a handful of fresh games moves weights on noise
                  (SE(Brier) ~ sigma/sqrt(n); at n=150, ±0.02).
    4. PROMOTE    challenger vs champion on the challenger's temporal
                  holdout (data neither model saw in training for the
                  challenger; strictly post-train for any older champion).
                  Promotion iff challenger loss <= champion loss. A worse
                  retrain can never take production traffic.
    5. PREDICT    score tonight's slate with whatever model is promoted.

Every run appends one JSON line to LOG_DIR/loop_runs.jsonl.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import LOG_DIR, MIN_NEW_GAMES_RETRAIN, PROCESSED_DIR

GAME_KINDS: tuple[str, ...] = ("game_clf", "game_margin")


def _n_new_games(kind: str) -> int:
    from src.model.registry import load_model
    _, meta = load_model(kind)
    df = pd.read_parquet(PROCESSED_DIR / "features_games.parquet",
                         columns=["game_date"])
    return int((df["game_date"] > pd.Timestamp(meta["holdout_span"][1])).sum())


def _challenge(kind: str) -> dict:
    """Train a challenger; promote iff it beats the champion on its holdout."""
    from src.model.registry import load_model, promote, save_model
    from src.model.train import _fold_metrics, cross_validate, fit_final, _load_matrix

    champion, champ_meta = load_model(kind)
    cv_metrics, oof = cross_validate(kind)
    challenger, meta = fit_final(kind)
    meta["cv_metrics"] = cv_metrics
    version = save_model(challenger, meta, oof, promote_now=False)

    # Head-to-head on the challenger's holdout slice.
    X, y, cols, dates = _load_matrix(kind)
    lo = pd.Timestamp(meta["holdout_span"][0])
    mask = (dates > lo).to_numpy()
    dm = xgb.DMatrix(X[mask], feature_names=cols)
    key = "brier" if kind == "game_clf" else "rmse"
    # symmetric comparison: both models truncated to their own best_iteration
    champ_ir = ((0, int(champ_meta["best_iteration"]) + 1)
                if champ_meta.get("best_iteration") is not None else (0, 0))
    m_champ = _fold_metrics(kind, y[mask], champion.predict(
        dm, iteration_range=champ_ir))
    m_chall = _fold_metrics(kind, y[mask], challenger.predict(
        dm, iteration_range=(0, challenger.best_iteration + 1)))

    promoted = m_chall[key] <= m_champ[key]
    if promoted:
        promote(kind, version)
    return {"kind": kind, "challenger": version, "promoted": promoted,
            "champion_" + key: round(m_champ[key], 5),
            "challenger_" + key: round(m_chall[key], 5),
            "n_compare": int(mask.sum())}


def daily_update(scrape: bool = True, predict: bool = True,
                 force_retrain: bool = False) -> dict:
    from src.features.temporal import build_game_matrix
    from src.ingest import unify
    from src.ingest.bref_scraper import scrape_range
    from src.recalibrate.drift import full_report

    run: dict = {"ts": datetime.now().isoformat(timespec="seconds")}
    try:
        # 1. INGEST ----------------------------------------------------------
        if scrape:
            y = date.today() - timedelta(days=1)
            # trailing window: the done-set makes re-scraping idempotent, so
            # this backfills holes left by missed runs (machine off/asleep)
            scrape_range(y - timedelta(days=6), y)
        unify.run()
        build_game_matrix()

        # 2. MONITOR ---------------------------------------------------------
        report = full_report("game_clf")
        n_new = _n_new_games("game_clf")
        run["drift"] = report
        run["n_new_games"] = n_new

        # 3-4. RETRAIN + PROMOTE (gated) --------------------------------------
        should = (force_retrain or report["retrain_signal"]
                  or n_new >= MIN_NEW_GAMES_RETRAIN)
        run["retrain_triggered"] = bool(should)
        if should:
            run["challenges"] = [_challenge(k) for k in GAME_KINDS]

        # 5. PREDICT ----------------------------------------------------------
        if predict:
            from src.model.predict import predict_games
            preds = predict_games(date.today())
            run["n_predictions"] = int(len(preds))
        run["status"] = "ok"
    except Exception as e:                  # noqa: BLE001 — ledger must record
        run["status"] = "error"
        run["error"] = f"{type(e).__name__}: {e}"
        run["trace"] = traceback.format_exc(limit=8)
        raise
    finally:
        with open(LOG_DIR / "loop_runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(run, default=str) + "\n")
        print(json.dumps({k: v for k, v in run.items() if k != "trace"},
                         indent=2, default=str))
    return run
