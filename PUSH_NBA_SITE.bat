@echo off
REM Publish latest slates + health to the NBA edge terminal (GitHub Pages).
REM Run from a clone of github.com/Gozorp/nba-edge after the daily pipeline.
python -m src.site.export_site_data
python -m src.model.props
python -m src.site.summer_league

REM ---- data-contract gate: structural violations BLOCK the publish --------
python -m src.site.contracts
if errorlevel 1 (
  echo CONTRACT VIOLATIONS - publish blocked. See docs/data/contract_report.json
  exit /b 1
)

git add docs/data
git commit -m "site: daily slate + props + SL refresh (contracts green)"
git push origin main
