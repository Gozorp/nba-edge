"""
config.py — Single source of truth for paths, constants, and thresholds.

Every threshold in this file carries its mathematical or documented
justification. No magic numbers exist elsewhere in the pipeline.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# PATHS  (repo root = .../nba_pipeline ; archive DB lives one level up)
# ----------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parent
WORKSPACE_ROOT: Path = REPO_ROOT.parent                      # D:\NBA edge
ARCHIVE_SQLITE: Path = WORKSPACE_ROOT / "archive" / "nba.sqlite"

DATA_DIR: Path = REPO_ROOT / "nba_data"
RAW_DIR: Path = DATA_DIR / "raw"            # scraped HTML-derived tables (parquet)
PROCESSED_DIR: Path = DATA_DIR / "processed"  # feature matrices (parquet)
MODEL_DIR: Path = DATA_DIR / "models"       # versioned joblib artifacts
LOG_DIR: Path = DATA_DIR / "logs"           # scraper + metrics logs (JSONL)

for _d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# ARCHIVE FACTS (asserted at ingestion; fail fast if the archive changes)
# ----------------------------------------------------------------------------
ARCHIVE_GAME_ROWS_MIN: int = 65_000          # observed: 65,698 rows
ARCHIVE_MAX_DATE: str = "2023-06-12"         # observed coverage boundary
ARCHIVE_GAME_COLS: int = 55                  # observed game-table width

# ----------------------------------------------------------------------------
# SCRAPER — basketball-reference.com
# Sports Reference publicly documents a hard cap of 20 requests/minute;
# exceeding it triggers a >= 1 hour IP jail. We operate at 75% of the cap:
#   rate = 0.75 * 20/min = 15/min  ->  min interval = 60/15 = 4.0 s.
# Jitter U(0, 1.5) s decorrelates request timing (avoids fixed-frequency
# signatures). Exponential backoff: delay_k = BASE * FACTOR**k, capped.
# ----------------------------------------------------------------------------
BREF_BASE_URL: str = "https://www.basketball-reference.com"
MIN_REQUEST_INTERVAL_S: float = 4.0
JITTER_MAX_S: float = 1.5
BACKOFF_BASE_S: float = 4.0
BACKOFF_FACTOR: float = 2.0
BACKOFF_MAX_RETRIES: int = 5                 # 4+8+16+32+64 = 124 s worst case
BACKOFF_CAP_S: float = 300.0
HTTP_TIMEOUT_S: float = 30.0
USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 nba-edge-research/1.0"
)
RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# ----------------------------------------------------------------------------
# FEATURES
# Rolling windows: 3 (short-term form), 7 (≈ 2 weeks of schedule, the span
# within which lineup/rotation context is quasi-stationary), and expanding
# season-to-date. All rolling stats are shifted by 1 game: feature at game t
# uses games [t-w, t-1] only — the leakage-free construction.
# ----------------------------------------------------------------------------
ROLLING_WINDOWS: tuple[int, ...] = (3, 7)
MIN_GAMES_FOR_FEATURES: int = 3              # rows with < 3 priors are dropped
SEASON_START_MONTH: int = 8                  # Aug+ -> season = year+1 label

# Archetype clustering: k selected by silhouette scan over this range.
# Lower bound 4: fewer clusters than traditional positions carries no
# information gain. Upper bound 12: beyond this, mean cluster occupancy for a
# ~450-player league drops below ~38, degrading centroid stability.
ARCHETYPE_K_RANGE: tuple[int, int] = (4, 12)
ARCHETYPE_MIN_MINUTES: float = 500.0         # season MP floor: SE of per-36
                                             # rates scales ~1/sqrt(MP);
                                             # 500 MP halves SE vs 125 MP.
KMEANS_SEED: int = 42                        # determinism requirement

# ----------------------------------------------------------------------------
# MODEL
# scale_pos_weight is computed from data (N_neg / N_pos), never hardcoded.
# n_splits=5 on ~2 seasons of gap data yields validation folds of >= ~400
# games each: SE(logloss) ≈ sigma/sqrt(400) — acceptable fold variance.
# ----------------------------------------------------------------------------
TSCV_SPLITS: int = 5
EARLY_STOPPING_ROUNDS: int = 50
XGB_SEED: int = 42

# Game models train on the modern era only. Empirical, not aesthetic: PSI of
# season-to-date 3P-share / ORtg features between the 1985+ pool and current
# games is 7.8 (39x the 0.25 action threshold) — the eras are different
# populations. 2015-16 is the canonical inflection (three-point share regime
# break); post-cut PSI sits inside stable bounds, so the drift monitor
# measures real drift instead of permanent era mismatch.
TRAIN_START_SEASON: int = 2016

# ----------------------------------------------------------------------------
# RECALIBRATION
# PSI thresholds are the documented industry convention (Siddiqi 2006):
#   PSI < 0.10 stable | 0.10–0.25 moderate shift | > 0.25 action required.
# Performance gate: retrain when rolling Brier over the monitor window
# exceeds baseline + 2*sigma_baseline (two-sided z at alpha ≈ 0.05).
# MIN_NEW_GAMES_RETRAIN = 150: SE of a Brier estimate at sigma ≈ 0.25 is
# 0.25/sqrt(150) ≈ 0.02 — small enough to distinguish real drift from noise.
# ----------------------------------------------------------------------------
PSI_MODERATE: float = 0.10
PSI_CRITICAL: float = 0.25
# PSI reference = trailing N seasons of the train span, not the full span.
# The league's scoring environment trends ~+1 pt/season (secular, not
# drift); a pooled decade reference makes trend-induced PSI permanent.
# Two seasons bound trend PSI < 0.25 while abrupt regime breaks
# (rule/ball changes) still trip the gate.
PSI_REF_SEASONS: int = 2
DRIFT_Z_THRESHOLD: float = 2.0
MONITOR_WINDOW_GAMES: int = 100
MIN_NEW_GAMES_RETRAIN: int = 150
MODELS_TO_KEEP: int = 10                     # artifact retention depth
