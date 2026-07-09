"""
registry.py — Model version control + metrics ledger.

Layout (MODEL_DIR):
    {kind}/v{NNNN}/model.ubj        — XGBoost native binary (portable, fast)
    {kind}/v{NNNN}/model.joblib     — joblib copy (ecosystem interop)
    {kind}/v{NNNN}/meta.json        — params, spans, holdout + CV metrics
    {kind}/v{NNNN}/oof.parquet      — out-of-fold predictions (calibration audit)
    {kind}/CURRENT                  — text file holding the promoted version id

LOG_DIR/metrics.jsonl — append-only ledger, one line per trained version.

Promotion is explicit: save_model() writes the artifact; promote() moves the
CURRENT pointer. The recalibration loop only promotes when the challenger
beats the champion on the shared holdout (see recalibrate/loop.py), so a bad
retrain can never silently take production traffic.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import LOG_DIR, MODEL_DIR, MODELS_TO_KEEP


def _kind_dir(kind: str) -> Path:
    d = MODEL_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _next_version(kind: str) -> int:
    existing = [int(p.name[1:]) for p in _kind_dir(kind).glob("v*") if p.is_dir()]
    return max(existing, default=0) + 1


def save_model(bst: xgb.Booster, meta: dict, oof: pd.DataFrame | None = None,
               promote_now: bool = True) -> str:
    kind = meta["kind"]
    v = _next_version(kind)
    vdir = _kind_dir(kind) / f"v{v:04d}"
    vdir.mkdir()

    bst.save_model(str(vdir / "model.ubj"))
    joblib.dump(bst, vdir / "model.joblib", compress=3)
    meta = dict(meta)
    meta["version"] = f"v{v:04d}"
    meta["saved_at"] = datetime.now().isoformat(timespec="seconds")
    (vdir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    if oof is not None:
        oof.to_parquet(vdir / "oof.parquet", index=False)

    with open(LOG_DIR / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": meta["saved_at"], "kind": kind, "version": meta["version"],
            "holdout_metrics": meta.get("holdout_metrics"),
            "n_train": meta.get("n_train"),
        }, default=str) + "\n")

    if promote_now:
        promote(kind, f"v{v:04d}")
    _prune(kind)
    print(f"[registry] saved {kind}/{meta['version']}"
          f"{' (promoted)' if promote_now else ''}")
    return meta["version"]


def promote(kind: str, version: str) -> None:
    vdir = _kind_dir(kind) / version
    assert (vdir / "model.ubj").exists(), f"FATAL: cannot promote {kind}/{version}"
    (_kind_dir(kind) / "CURRENT").write_text(version)


def current_version(kind: str) -> str | None:
    p = _kind_dir(kind) / "CURRENT"
    return p.read_text().strip() if p.exists() else None


def load_model(kind: str, version: str | None = None
               ) -> tuple[xgb.Booster, dict]:
    version = version or current_version(kind)
    assert version is not None, f"FATAL: no promoted model for '{kind}'"
    vdir = _kind_dir(kind) / version
    bst = xgb.Booster()
    bst.load_model(str(vdir / "model.ubj"))
    meta = json.loads((vdir / "meta.json").read_text())
    return bst, meta


def _prune(kind: str) -> None:
    """Keep the newest MODELS_TO_KEEP versions (never the promoted one)."""
    cur = current_version(kind)
    vers = sorted((p for p in _kind_dir(kind).glob("v*") if p.is_dir()),
                  key=lambda p: p.name, reverse=True)
    for p in vers[MODELS_TO_KEEP:]:
        if p.name != cur:
            shutil.rmtree(p)


def metrics_history(kind: str | None = None) -> pd.DataFrame:
    p = LOG_DIR / "metrics.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in p.read_text().splitlines() if line]
    df = pd.json_normalize(rows)
    return df if kind is None else df[df["kind"] == kind].reset_index(drop=True)
