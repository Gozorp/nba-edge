"""
drift.py — Phase 4a: statistical monitoring of the deployed model.

Two orthogonal failure modes are tested:

1. DATA DRIFT (covariate shift) — Population Stability Index per feature.
       PSI = sum_i (a_i - e_i) * ln(a_i / e_i)
   over 10 quantile bins of the model's own training distribution
   (e = expected/train shares, a = actual/recent shares, both floored at
   1e-4 to keep ln finite). Convention (Siddiqi): <0.10 stable,
   0.10-0.25 moderate, >0.25 action.

2. CONCEPT DRIFT (performance decay) — one-sided z-test on the mean
   per-game Brier contribution:
       b_i = (p_i - y_i)^2
       z = (mean_recent(b) - mean_holdout(b)) / (sd_holdout(b)/sqrt(n_recent))
   Trigger at z > DRIFT_Z_THRESHOLD (=2.0, alpha ≈ 0.023 one-sided).
   The holdout ledger (oof/holdout predictions stored at save time) supplies
   the baseline; no distributional assumption beyond CLT on means of
   bounded [0,1] variables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    DRIFT_Z_THRESHOLD,
    MONITOR_WINDOW_GAMES,
    PROCESSED_DIR,
    PSI_CRITICAL,
    PSI_MODERATE,
)

_EPS = 1e-4
_N_BINS = 10


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """f: (R^n, R^m) -> R+. Quantile-binned Population Stability Index."""
    assert expected.size >= 100 and actual.size >= 30, "FATAL: PSI sample floor"
    qs = np.quantile(expected, np.linspace(0, 1, _N_BINS + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    qs = np.unique(qs)                      # constant features collapse bins
    if qs.size < 3:
        return 0.0                          # degenerate: no variation to shift
    e = np.clip(np.histogram(expected, qs)[0] / expected.size, _EPS, None)
    a = np.clip(np.histogram(actual, qs)[0] / actual.size, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_drift(kind: str = "game_clf") -> dict[str, Any]:
    from src.model.registry import load_model

    _, meta = load_model(kind)
    cols: list[str] = meta["feature_cols"]
    df = pd.read_parquet(PROCESSED_DIR / "features_games.parquet"
                         ).sort_values("game_date")
    df = df[df["season_type"] == "Regular Season"]
    from config import PSI_REF_SEASONS
    train_end = pd.Timestamp(meta["train_span"][1])
    ref_start = max(pd.Timestamp(meta["train_span"][0]),
                    train_end - pd.DateOffset(years=PSI_REF_SEASONS))
    ref = df[(df["game_date"] >= ref_start) & (df["game_date"] <= train_end)]
    recent = df.tail(MONITOR_WINDOW_GAMES)
    assert len(ref) >= 100, "FATAL: reference window too small for PSI"

    monitored = [c for c in cols if c.startswith("d_")
                 or c.endswith(("rest_days", "is_b2b", "is_3in4", "is_4in6"))]
    assert len(monitored) >= 10, "FATAL: monitored feature subset too small"
    scores = {c: psi(ref[c].to_numpy(np.float64),
                     recent[c].to_numpy(np.float64)) for c in monitored}
    critical = sorted((c for c, v in scores.items() if v > PSI_CRITICAL),
                      key=scores.get, reverse=True)
    moderate = [c for c, v in scores.items()
                if PSI_MODERATE < v <= PSI_CRITICAL]
    return {
        "max_psi": max(scores.values()),
        "critical_features": critical,
        "n_moderate": len(moderate),
        "data_drift": bool(critical),
        "psi_scores": {k: round(v, 4) for k, v
                       in sorted(scores.items(), key=lambda kv: -kv[1])[:15]},
    }


def performance_drift(kind: str = "game_clf") -> dict[str, Any]:
    from src.model.registry import current_version, load_model

    bst, meta = load_model(kind)
    cols = meta["feature_cols"]
    df = pd.read_parquet(PROCESSED_DIR / "features_games.parquet"
                         ).sort_values("game_date")
    holdout_end = pd.Timestamp(meta["holdout_span"][1])

    # Baseline: per-game losses on the model's stored holdout span.
    ho = df[(df["game_date"] > pd.Timestamp(meta["holdout_span"][0]))
            & (df["game_date"] <= holdout_end)]
    recent = df[df["game_date"] > holdout_end].tail(MONITOR_WINDOW_GAMES)
    if len(recent) < 30:
        return {"concept_drift": False, "reason": f"only {len(recent)} "
                "post-holdout games — below CLT floor (30)"}

    def _losses(frame: pd.DataFrame) -> np.ndarray:
        dm = xgb.DMatrix(frame[cols].astype("float64"), feature_names=cols)
        p = bst.predict(dm)
        if kind == "game_clf":
            return (p - frame["home_win"].to_numpy(np.float64)) ** 2
        return (p - frame["margin_home"].to_numpy(np.float64)) ** 2

    b_ho, b_re = _losses(ho), _losses(recent)
    z = float((b_re.mean() - b_ho.mean())
              / (b_ho.std(ddof=1) / np.sqrt(len(b_re))))
    return {
        "baseline_loss": round(float(b_ho.mean()), 5),
        "recent_loss": round(float(b_re.mean()), 5),
        "n_recent": int(len(b_re)),
        "z": round(z, 3),
        "concept_drift": bool(z > DRIFT_Z_THRESHOLD),
        "model": f"{kind}/{current_version(kind)}",
    }


def full_report(kind: str = "game_clf") -> dict[str, Any]:
    rep = {"kind": kind,
           "feature_drift": feature_drift(kind),
           "performance_drift": performance_drift(kind)}
    rep["retrain_signal"] = (rep["feature_drift"]["data_drift"]
                             or rep["performance_drift"].get("concept_drift",
                                                             False))
    return rep


if __name__ == "__main__":
    print(json.dumps(full_report(), indent=2, default=str))
