"""
explain.py — Phase 3c: SHAP attribution for any registered model.

Shapley values are computed by XGBoost's NATIVE exact TreeSHAP
(`predict(..., pred_contribs=True)`) rather than shap's model re-parser:
identical algorithm (Lundberg et al. 2020), exact in polynomial time, and
immune to shap<->xgboost serialization drift (shap's loader lags new
XGBoost model formats; the native path has no loader at all). The shap
package is used ONLY for the beeswarm visualization of the raw phi matrix.

Local accuracy holds by construction:  f(x_i) = phi_i1 + ... + phi_id + bias
(margin/log-odds space for classifiers) — still asserted, never assumed.

Outputs (MODEL_DIR/{kind}/{version}/):
    shap_values.parquet     — per-row phi matrix (sampled)
    shap_importance.json    — global mean(|phi|) ranking
    shap_summary.png        — beeswarm (matplotlib, headless-safe)

Usage:
    python -m src.model.explain --kind game_clf [--n 5000]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless: render to file only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import MODEL_DIR, PROCESSED_DIR, XGB_SEED

_MATRIX = {
    "game_clf": ("features_games.parquet", "feature_cols_games.json"),
    "game_margin": ("features_games.parquet", "feature_cols_games.json"),
    "player_pi": ("features_players.parquet", "feature_cols_players.json"),
}


def explain(kind: str, n_sample: int = 5000) -> pd.DataFrame:
    from src.model.registry import current_version, load_model

    bst, meta = load_model(kind)
    mat_file, cols_file = _MATRIX[kind]
    df = pd.read_parquet(PROCESSED_DIR / mat_file)
    cols: list[str] = json.loads((PROCESSED_DIR / cols_file).read_text())
    assert cols == meta["feature_cols"], (
        "FATAL: feature schema drift between matrix and model artifact"
    )

    X = df[cols].astype("float64")
    if len(X) > n_sample:                   # deterministic subsample
        X = X.sample(n=n_sample, random_state=XGB_SEED).sort_index()

    dm = xgb.DMatrix(X, feature_names=cols)
    contribs = bst.predict(dm, pred_contribs=True)   # exact TreeSHAP
    assert contribs.shape == (len(X), len(cols) + 1), "FATAL: phi shape"
    phi, bias = contribs[:, :-1], contribs[:, -1]

    # Local accuracy audit: f(x) = sum_j phi_j + bias, in margin space.
    margin = bst.predict(dm, output_margin=True)
    probe = float(np.abs(phi.sum(axis=1) + bias - margin).max())
    assert probe < 1e-3, f"FATAL: SHAP local accuracy violated ({probe:.2e})"

    version = current_version(kind)
    vdir = MODEL_DIR / kind / version
    pd.DataFrame(phi, columns=cols, index=X.index).to_parquet(
        vdir / "shap_values.parquet")

    imp = (pd.Series(np.abs(phi).mean(axis=0), index=cols)
           .sort_values(ascending=False))
    (vdir / "shap_importance.json").write_text(imp.round(5).to_json(indent=1))

    try:
        import shap
        shap.summary_plot(phi, X, show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(vdir / "shap_summary.png", dpi=150)
        plt.close()
    except ImportError:
        print("[shap] package absent — phi matrix + ranking still written")

    print(f"[shap:{kind}/{version}] local-accuracy err={probe:.2e} | "
          "top 10 by mean |phi|:")
    for name, val in imp.head(10).items():
        print(f"    {name:<28s} {val:.4f}")
    return imp.to_frame("mean_abs_shap")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="game_clf", choices=list(_MATRIX))
    ap.add_argument("--n", type=int, default=5000)
    a = ap.parse_args()
    explain(a.kind, a.n)
