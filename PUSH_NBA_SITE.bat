@echo off
REM Publish latest slates + health to the NBA edge terminal (GitHub Pages).
REM Run from a clone of github.com/Gozorp/nba-edge after the daily pipeline.
python -m src.site.export_site_data
git add docs/data
git commit -m "site: daily slate + health refresh"
git push origin main
