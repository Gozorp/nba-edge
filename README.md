# NBA Edge — Automated Prediction Pipeline

Deterministic, self-recalibrating pipeline: local archive + basketball-reference
scraper → leakage-free temporal features → XGBoost (moneyline / spread / player
PI) → SHAP attribution → drift-gated daily retraining.

## Data topology

| Layer | Source | Coverage | Content |
|---|---|---|---|
| Base | `../archive/nba.sqlite` (2.35 GB, 16 tables) | 1946-11 → 2023-06-12 | team box scores, PBP, officials, inactives |
| Gap + daily | basketball-reference scraper | 2023-10 → present | team box + **player game logs incl. advanced** (TS%, USG%, ORtg/DRtg, BPM) |

The archive has **no player game logs**; all player-level data enters via the
scraper. The two id spaces are fused in `unify.py` (collision asserted
impossible) with a deterministic team crosswalk (`BRK→BKN, CHO→CHA, PHO→PHX`)
and an exact-match-or-escalate player crosswalk (no fuzzy matching, ever).

## Repo layout

```
nba_pipeline/
├── config.py                    # every path + threshold, each justified
├── run_daily.py                 # scheduler entrypoint (Task Scheduler / cron)
├── nba_data/{raw,processed,models,logs}/
└── src/
    ├── ingest/    local_db.py bref_scraper.py crosswalk.py unify.py
    ├── features/  temporal.py archetypes.py performance_index.py
    ├── model/     train.py registry.py explain.py predict.py
    └── recalibrate/ drift.py loop.py
```

## First-time setup

```bash
pip install -r requirements.txt
python -m src.ingest.local_db                                  # ~1 min
python -m src.ingest.crosswalk
python -m src.ingest.bref_scraper --start 2023-10-01 --end 2026-07-08
#   ≈ 4,000 pages at 15 req/min ≈ 5–6 h. Resumable: kill/rerun anytime.
python -m src.ingest.unify
python -m src.features.temporal
python -m src.features.archetypes                              # player roles
python -m src.features.performance_index                       # PI target
python -m src.model.train --kinds game_clf game_margin player_pi
python -m src.model.explain --kind game_clf
```

Then schedule `run_daily.py` (see its docstring). Each run: scrape yesterday →
rebuild features → drift report → gated retrain + champion/challenger
promotion → predict tonight's slate → `nba_data/processed/predictions/`.

## Design invariants

1. **Leakage**: every feature at game *t* is `shift(1)`-windowed over `[.., t-1]`
   within (team, season); CV is walk-forward `TimeSeriesSplit` with fold
   geometry asserted (`train.max < val.min`). Random K-fold is prohibited.
2. **Determinism**: fixed seeds (KMeans `n_init=20, seed=42`, XGB `seed=42`);
   typed ingestion; `assert`ed matrix dimensions before/after every join;
   NaN is fatal unless explicitly routed (each routing rule documented inline).
3. **Rate safety**: 15 req/min (75 % of Sports Reference's documented 20/min
   cap) + jitter + exponential backoff (base 4 s, factor 2, ≤5 retries,
   Retry-After honored). Every request logged to `logs/scraper.jsonl`.
4. **Recalibration**: retrain gate = PSI > 0.25 on any feature OR Brier
   z > 2.0 OR ≥150 new games; challenger promoted only if it beats the
   champion on the shared temporal holdout. Artifacts versioned
   (`models/{kind}/vNNNN/` + `CURRENT` pointer + `logs/metrics.jsonl`).

## Model targets

| kind | objective | target | cost function |
|---|---|---|---|
| `game_clf` | `binary:logistic` | home_win ∈ {0,1} | J = −(1/N) Σ [y ln p + (1−y) ln(1−p)], `scale_pos_weight = N_neg/N_pos` |
| `game_margin` | `reg:squarederror` | margin ∈ ℤ∖{0} | J = (1/N) Σ (y−ŷ)² (spread) |
| `player_pi` | `reg:squarederror` | PI (ridge-learned event weights, archive-era frozen) | J = (1/N) Σ (y−ŷ)² |

Player archetypes: K-Means on standardized per-36 + shot-profile + usage
space, k* = argmax silhouette over k ∈ [4,12]; FATAL if silhouette < 0.05.

## Fatal boundary conditions (pipeline halts rather than degrades)

- archive missing / row count < 65,000 / width ≠ 55 / max date moved
- duplicate `game_id` > 0.1 %, null core stats > 0.5 %, tied final score
- feature-warmup drops ≥ 15 % of rows; any NaN surviving explicit routing
- opponent join changes row count; home/away resolution ≠ exactly 1+1
- CV fold overlap (`train.max ≥ val.min`); non-finite feature matrix
- silhouette < 0.05; PI ridge R² < 0.45; SHAP local-accuracy error ≥ 1e-3
- scraper: non-retryable HTTP or 5 exhausted retries (halt, never hammer)

## Known limitations (v1)

- Player inference for upcoming games requires a minutes/roster projection;
  the player model trains and evaluates on completed games (trailing-form →
  current-game PI). Slate-level player projections are the natural v2.
- Archetype/PI features are not yet injected into the game-level matrix
  (requires minutes-weighted roster aggregation — v2).
- Betting-market lines are not scraped; `fair_ml_*` are model-fair odds, not
  edges vs. a book. Comparing them to live lines is your edge calculation.

## Publishing the terminal

After a training/predict cycle: `python -m src.site.export_site_data` regenerates
`docs/data/` (per-date slates, manifest, health). `PUSH_NBA_SITE.bat` wraps
export + commit + push — same publish pattern as the MLB terminal.
