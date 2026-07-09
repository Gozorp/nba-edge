"""
archetypes.py — Phase 2b: unsupervised player role discovery (K-Means).

Positions are labels; roles are distributions. We cluster player-SEASONS in a
standardized per-36 + shot-profile + usage space and let k be selected by the
data, not by convention.

Selection rule (deterministic):
    k* = argmax_{k in [4,12]} silhouette(k),  fixed seed, n_init=20.
    FATAL if max silhouette < 0.05 — a value that low means the feature space
    has no cluster structure and any "archetypes" would be fiction.

Eligibility: season MP >= 500 (SE of a per-36 rate scales ~ 1/sqrt(MP);
below ~500 MP the noise dominates the signal being clustered).

Labels are generated deterministically from centroid z-signatures
(top-2 positive deviations), e.g. 'usg36+ ast36+' — no hand-assigned names.

Output: player_archetypes.parquet
    (player_id, season, cluster, label, dist_to_centroid, silhouette_k)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    ARCHETYPE_K_RANGE,
    ARCHETYPE_MIN_MINUTES,
    KMEANS_SEED,
    PROCESSED_DIR,
    RAW_DIR,
    SEASON_START_MONTH,
)

# Clustering space: per-36 volume, shot profile, usage, efficiency, defense.
CLUSTER_COLS: tuple[str, ...] = (
    "pts36", "ast36", "oreb36", "dreb36", "stl36", "blk36", "tov36",
    "fg3a36", "fta36", "fga36",
    "fg3a_per_fga", "usg_pct", "ts_pct", "ast_pct", "trb_pct",
)


def build_player_seasons() -> pd.DataFrame:
    pb_path = RAW_DIR / "player_box.parquet"
    sc_path = RAW_DIR / "schedule.parquet"
    assert pb_path.exists(), "FATAL: player_box.parquet missing — run scraper"
    pb = pd.read_parquet(pb_path)
    sched = pd.read_parquet(sc_path, columns=["bref_game_id", "game_date"])
    pb = pb.merge(sched, on="bref_game_id", how="left")
    assert pb["game_date"].notna().all(), "FATAL: player rows without schedule"
    pb["game_date"] = pd.to_datetime(pb["game_date"])
    y = pb["game_date"].dt.year
    pb["season"] = (y + (pb["game_date"].dt.month >= SEASON_START_MONTH)
                    ).astype("Int32")

    # NULL ROUTING at the processed boundary: counting stats absent on a
    # played-minutes row are true zeros (B-Ref omits zero cells).
    counting = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
                "ast", "stl", "blk", "tov", "pf", "pts"]
    pb[counting] = pb[counting].fillna(0.0)

    g = pb.groupby(["player_id", "player", "season"], observed=True)
    agg = g.agg(
        mp=("mp", "sum"), games=("mp", "size"),
        pts=("pts", "sum"), ast=("ast", "sum"), oreb=("oreb", "sum"),
        dreb=("dreb", "sum"), stl=("stl", "sum"), blk=("blk", "sum"),
        tov=("tov", "sum"), fga=("fga", "sum"), fg3a=("fg3a", "sum"),
        fta=("fta", "sum"), fgm=("fgm", "sum"), fg3m=("fg3m", "sum"),
        ftm=("ftm", "sum"),
    ).reset_index()

    # Minutes-weighted means for rate stats (weighting by MP is exact for
    # possession-denominated rates under constant team pace approximation).
    w = pb["mp"].clip(lower=1e-9)
    for rate in ("usg_pct", "ast_pct", "trb_pct"):
        pb[f"_w{rate}"] = pb[rate].fillna(0.0) * w
    wsum = pb.groupby(["player_id", "season"], observed=True)[
        [f"_w{r}" for r in ("usg_pct", "ast_pct", "trb_pct")] ].sum()
    mpsum = pb.groupby(["player_id", "season"], observed=True)["mp"].sum()
    for rate in ("usg_pct", "ast_pct", "trb_pct"):
        agg[rate] = (wsum[f"_w{rate}"] / mpsum).reindex(
            agg.set_index(["player_id", "season"]).index).values

    agg = agg[agg["mp"] >= ARCHETYPE_MIN_MINUTES].reset_index(drop=True)
    assert len(agg) > 0, "FATAL: no player-seasons above minutes floor"

    for c in ("pts", "ast", "oreb", "dreb", "stl", "blk", "tov",
              "fga", "fg3a", "fta"):
        agg[f"{c}36"] = agg[c] * 36.0 / agg["mp"]
    agg["fg3a_per_fga"] = np.where(agg["fga"] > 0, agg["fg3a"] / agg["fga"], 0.0)
    # True shooting: TS% = PTS / (2 * (FGA + 0.44 * FTA)); 0-attempt guard.
    tsa = agg["fga"] + 0.44 * agg["fta"]
    agg["ts_pct"] = np.where(tsa > 0, agg["pts"] / (2.0 * tsa), 0.0)

    out = agg[["player_id", "player", "season", "mp", "games"]
              + list(CLUSTER_COLS)].copy()
    assert not out[list(CLUSTER_COLS)].isna().any().any(), (
        "FATAL: NaN in clustering space after explicit routing"
    )
    return out


def _centroid_label(centroid_z: np.ndarray, cols: tuple[str, ...]) -> str:
    top2 = np.argsort(centroid_z)[::-1][:2]
    return " ".join(f"{cols[i]}+" for i in top2)


def fit_archetypes(ps: pd.DataFrame) -> pd.DataFrame:
    X = ps[list(CLUSTER_COLS)].to_numpy(dtype=np.float64)
    assert X.ndim == 2 and X.shape[1] == len(CLUSTER_COLS), "FATAL: X shape"
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    k_lo, k_hi = ARCHETYPE_K_RANGE
    assert len(ps) > k_hi, "FATAL: fewer samples than max k"
    results: dict[int, tuple[float, KMeans]] = {}
    for k in range(k_lo, k_hi + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=KMEANS_SEED)
        lab = km.fit_predict(Xz)
        results[k] = (float(silhouette_score(Xz, lab)), km)

    k_star = max(results, key=lambda k: results[k][0])
    sil, km = results[k_star]
    assert sil >= 0.05, (
        f"FATAL: max silhouette {sil:.3f} < 0.05 — no cluster structure"
    )
    print(f"[archetypes] k*={k_star} silhouette={sil:.3f} "
          f"(scan {dict((k, round(v[0], 3)) for k, v in results.items())})")

    lab = km.predict(Xz)
    dists = np.linalg.norm(Xz - km.cluster_centers_[lab], axis=1)
    labels = {c: _centroid_label(km.cluster_centers_[c], CLUSTER_COLS)
              for c in range(k_star)}

    out = ps[["player_id", "player", "season", "mp", "games"]].copy()
    out["cluster"] = lab.astype("int16")
    out["label"] = out["cluster"].map(labels).astype("string")
    out["dist_to_centroid"] = dists
    out["silhouette_k"] = sil
    out.to_parquet(PROCESSED_DIR / "player_archetypes.parquet", index=False)
    print(f"[archetypes] {len(out)} player-seasons -> "
          f"{k_star} archetypes: {sorted(set(labels.values()))}")
    return out


if __name__ == "__main__":
    fit_archetypes(build_player_seasons())
