"""
run_daily.py — single automation entrypoint.

Schedule at ~06:00 local (all previous night's games final):

Windows Task Scheduler:
    schtasks /Create /SC DAILY /ST 06:00 /TN "NBA Pipeline" ^
      /TR "\"C:\\Path\\to\\python.exe\" \"D:\\NBA edge\\nba_pipeline\\run_daily.py\""

cron:
    0 6 * * *  cd "/path/to/nba_pipeline" && python run_daily.py

First-time setup order (one-off, before scheduling):
    python -m src.ingest.local_db
    python -m src.ingest.crosswalk
    python -m src.ingest.bref_scraper --start 2023-10-01 --end <today>
    python -m src.ingest.unify
    python -m src.features.temporal
    python -m src.model.train
    python -m src.model.explain --kind game_clf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from src.recalibrate.loop import daily_update

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NBA pipeline daily loop")
    ap.add_argument("--no-scrape", action="store_true")
    ap.add_argument("--no-predict", action="store_true")
    ap.add_argument("--force-retrain", action="store_true")
    a = ap.parse_args()
    daily_update(scrape=not a.no_scrape, predict=not a.no_predict,
                 force_retrain=a.force_retrain)
