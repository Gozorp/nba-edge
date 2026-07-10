"""
backup.py — Automated daily backup of everything that cannot be regenerated.

WHAT IS (and isn't) BACKED UP
-----------------------------
IN  : nba_data/models   (every trained model version + metrics ledger)
      nba_data/logs     (scraper audit, loop runs, drift history)
      docs/data         (published site artifacts incl. season archives)
      config.py, pi_weights.json, feature-column contracts
OUT : archive/nba.sqlite (source data, 2.3 GB — belongs on its own drive)
      nba_data/raw|processed parquet (fully regenerable: scraper + pipeline)

Cadence: called at the end of every daily loop run (run_daily.py), so
"automated daily" holds by construction — no extra scheduler entry needed.
Rotation: last 14 dated zips kept in <workspace>/backups/. Idempotent per
day. Zip integrity is verified (testzip) before old backups are pruned.

Standalone:  python -m src.ops.backup
Override dir: BACKUP_DIR=/path python -m src.ops.backup
"""
from __future__ import annotations

import os
import sys
import zipfile
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import DATA_DIR, PROCESSED_DIR, REPO_ROOT, WORKSPACE_ROOT

KEEP = 14
BACKUP_DIR = Path(os.environ.get("BACKUP_DIR", WORKSPACE_ROOT / "backups"))

_TARGETS: list[tuple[Path, str]] = [
    (DATA_DIR / "models", "models"),
    (DATA_DIR / "logs", "logs"),
    (REPO_ROOT / "docs" / "data", "site_data"),
    (REPO_ROOT / "config.py", "config.py"),
    (PROCESSED_DIR / "pi_weights.json", "pi_weights.json"),
    (PROCESSED_DIR / "feature_cols_games.json", "feature_cols_games.json"),
    (PROCESSED_DIR / "feature_cols_players.json", "feature_cols_players.json"),
]


def run(force: bool = False) -> Path | None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    out = BACKUP_DIR / f"nbaedge_backup_{date.today().isoformat()}.zip"
    if out.exists() and not force:
        print(f"[backup] {out.name} already exists — skipping (idempotent)")
        return out

    tmp = out.with_suffix(".zip.part")
    n_files = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arcname in _TARGETS:
            if not src.exists():
                print(f"[backup] WARN missing target: {src}")
                continue
            if src.is_file():
                z.write(src, arcname); n_files += 1
            else:
                for f in sorted(src.rglob("*")):
                    if f.is_file():
                        z.write(f, f"{arcname}/{f.relative_to(src)}")
                        n_files += 1
    # integrity gate BEFORE this backup replaces anything
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
        assert bad is None, f"FATAL: corrupt member in backup: {bad}"
    tmp.replace(out)

    # rotate: newest KEEP dated backups survive
    zips = sorted(BACKUP_DIR.glob("nbaedge_backup_*.zip"), reverse=True)
    for stale in zips[KEEP:]:
        try:
            stale.unlink()
            print(f"[backup] rotated out {stale.name}")
        except OSError as e:
            # housekeeping must never kill a completed backup
            print(f"[backup] WARN could not rotate {stale.name}: {e}")

    mb = out.stat().st_size / 1048576
    print(f"[backup] {out.name}: {n_files} files, {mb:.1f} MB "
          f"(verified; keeping last {KEEP})")
    return out


if __name__ == "__main__":
    run(force="--force" in sys.argv)
