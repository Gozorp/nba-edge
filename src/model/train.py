"""
train.py — Phase 3: XGBoost training with walk-forward validation.

Models
------
    game_clf   : P(home_win)      objective binary:logistic
                 J(theta) = -(1/N) * sum_i [ y_i*log(p_i) + (1-y_i)*log(1-p_i) ]
                 y_i in {0,1},  p_i = sigmoid(f(x_i)) in (0,1)
    game_margin: E[margin_home]   objective reg:squarederror (spread model)
                 J(theta) = (1/N) * sum_i (y_i - f(x_i))^2 ,  y_i in Z\\{0}
    player_pi  : E[PI_next]       objective reg:squarederror

Validation
----------
sklearn TimeSeriesSplit over the DATE-SORTED matrix: every fold trains on
a strict past and validates on a strict future — the only admissible CV
geometry for forecasting. Random K-fold on temporal data leaks future
opponent form into the past and is prohibited here.

Imbalance
---------
Home teams win ~55-58% of NBA games. scale_pos_weight = N_neg / N_pos is
computed from the training fold (never hardcoded), which restores the
gradient balance of the minority class in J(theta).

Final artifact = trained on all data except the last HOLDOUT_FRACTION,
early-stopped on that temporal holdout, then persisted via registry.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    brier_score_loss, log_loss, mean_absolute_error, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import EARLY_STOPPING_ROUNDS, TSCV_SPLITS, XGB_SEED

HOLDOUT_FRACTION: float = 0.10   # final temporal holdout for early stopping
NUM_BOOST_ROUND: int = 2000      # upper bound; early stopping governs

BASE_PARAMS: dict[str, Any] = {
    "eta": 0.05,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "tree_method": "hist",
    "seed": XGB_SEED,
}


def _dmatrix(X: pd.DataFrame, y: np.ndarray | None = None) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y, feature_names=list(X.columns))


def _load_matrix(kind: str) -> tuple[pd.DataFrame, np.ndarray, list[str], pd.Series]:
    import json

    from config import PROCESSED_DIR
    if kind in ("game_clf", "game_margin"):
        df = pd.read_parquet(PROCESSED_DIR / "features_games.parquet")
        cols = json.loads((PROCESSED_DIR / "feature_cols_games.json").read_text())
        target = "home_win" if kind == "game_clf" else "margin_home"
    elif kind == "player_pi":
        df = pd.read_parquet(PROCESSED_DIR / "features_players.parquet")
        cols = json.loads((PROCESSED_DIR / "feature_cols_players.json").read_text())
        target = "pi"
    else:
        raise ValueError(f"unknown model kind: {kind}")

    if kind in ("game_clf", "game_margin"):
        from config import TRAIN_START_SEASON
        df = df[df["season"] >= TRAIN_START_SEASON]   # modern era only
    df = df.sort_values("game_date").reset_index(drop=True)  # temporal order
    assert df["game_date"].is_monotonic_increasing, "FATAL: unsorted matrix"
    assert len(df) > 5_000, "FATAL: era filter left too few games"
    X = df[cols].astype("float64")
    y = df[target].to_numpy(np.float64)
    assert np.isfinite(X.to_numpy()).all(), "FATAL: non-finite features"
    assert len(X) == len(y) and len(X) > 0, "FATAL: X/y dimension mismatch"
    return X, y, cols, df["game_date"]


def _params_for(kind: str, y_train: np.ndarray) -> dict[str, Any]:
    p = dict(BASE_PARAMS)
    if kind == "game_clf":
        n_pos = float(y_train.sum())
        n_neg = float(len(y_train) - n_pos)
        assert n_pos > 0 and n_neg > 0, "FATAL: degenerate class balance"
        p |= {"objective": "binary:logistic",
              "eval_metric": ["logloss"],
              "scale_pos_weight": n_neg / n_pos}
    else:
        p |= {"objective": "reg:squarederror", "eval_metric": ["rmse"]}
    return p


def _fold_metrics(kind: str, y_va: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if kind == "game_clf":
        return {
            "logloss": float(log_loss(y_va, pred, labels=[0, 1])),
            "brier": float(brier_score_loss(y_va, pred)),
            "auc": float(roc_auc_score(y_va, pred)),
            "acc": float(((pred >= 0.5).astype(int) == y_va).mean()),
        }
    return {
        "rmse": float(np.sqrt(np.mean((y_va - pred) ** 2))),
        "mae": float(mean_absolute_error(y_va, pred)),
    }


def cross_validate(kind: str) -> tuple[list[dict[str, float]], pd.DataFrame]:
    """Walk-forward CV. Returns per-fold metrics + out-of-fold predictions."""
    X, y, _, dates = _load_matrix(kind)
    tscv = TimeSeriesSplit(n_splits=TSCV_SPLITS)
    fold_metrics: list[dict[str, float]] = []
    oof: list[pd.DataFrame] = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        # Geometry assertion: strict past -> strict future, zero overlap.
        assert tr_idx.max() < va_idx.min(), "FATAL: temporal fold overlap"
        params = _params_for(kind, y[tr_idx])
        dtr = _dmatrix(X.iloc[tr_idx], y[tr_idx])
        dva = _dmatrix(X.iloc[va_idx], y[va_idx])
        bst = xgb.train(
            params, dtr, num_boost_round=NUM_BOOST_ROUND,
            evals=[(dva, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=False,
        )
        pred = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        m = _fold_metrics(kind, y[va_idx], pred)
        m["fold"] = fold
        m["n_train"], m["n_val"] = len(tr_idx), len(va_idx)
        m["best_iter"] = int(bst.best_iteration)
        fold_metrics.append(m)
        oof.append(pd.DataFrame({
            "game_date": dates.iloc[va_idx].values, "y": y[va_idx],
            "pred": pred, "fold": fold}))
        print(f"[cv:{kind}] fold {fold}: {m}")

    return fold_metrics, pd.concat(oof, ignore_index=True)


def fit_final(kind: str) -> tuple[xgb.Booster, dict[str, Any]]:
    """Train production booster on all-but-holdout; early stop on holdout."""
    X, y, cols, dates = _load_matrix(kind)
    cut = int(len(X) * (1.0 - HOLDOUT_FRACTION))
    assert 0 < cut < len(X), "FATAL: bad holdout cut"

    params = _params_for(kind, y[:cut])
    dtr = _dmatrix(X.iloc[:cut], y[:cut])
    dho = _dmatrix(X.iloc[cut:], y[cut:])
    bst = xgb.train(
        params, dtr, num_boost_round=NUM_BOOST_ROUND,
        evals=[(dho, "holdout")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=False,
    )
    pred = bst.predict(dho, iteration_range=(0, bst.best_iteration + 1))
    holdout = _fold_metrics(kind, y[cut:], pred)

    meta: dict[str, Any] = {
        "kind": kind,
        "params": {k: v for k, v in params.items() if k != "eval_metric"},
        "n_train": cut, "n_holdout": len(X) - cut,
        "best_iteration": int(bst.best_iteration),
        "feature_cols": cols,
        "train_span": [str(dates.iloc[0].date()), str(dates.iloc[cut - 1].date())],
        "holdout_span": [str(dates.iloc[cut].date()), str(dates.iloc[-1].date())],
        "holdout_metrics": holdout,
    }
    print(f"[final:{kind}] best_iter={bst.best_iteration} holdout={holdout}")
    return bst, meta


def train_all(kinds: tuple[str, ...] = ("game_clf", "game_margin")) -> None:
    from src.model.registry import save_model
    for kind in kinds:
        cv_metrics, oof = cross_validate(kind)
        bst, meta = fit_final(kind)
        meta["cv_metrics"] = cv_metrics
        save_model(bst, meta, oof)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+",
                    default=["game_clf", "game_margin"],
                    choices=["game_clf", "game_margin", "player_pi"])
    train_all(tuple(ap.parse_args().kinds))
