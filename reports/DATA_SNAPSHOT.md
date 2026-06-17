# Data snapshot — backtest reproducibility

The backtest metrics and figures in `reports/` were produced from a fixed
snapshot of the raw match dataset. Recording the snapshot's checksum here makes
the published numbers time-stable: anyone with a file whose SHA256 matches the
value below will reproduce the same 192-match backtest exactly.

## Source dataset (`data/raw/results.csv`)

International football results, downloaded from Kaggle
(`martj42/international-football-results-1872-to-2017`) via `src/fetch_data.py`.

| Field | Value |
| --- | --- |
| File | `data/raw/results.csv` |
| **SHA256** | `50f17eb331a3d8367184f3314cf41616782f842ebb39e42191260b414b56bc78` |
| Size | 3,724,399 bytes |
| Rows | 49,477 match results |
| Columns | `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral` |
| Date range | 1872-11-30 → 2026-06-27 |
| **Access date** (downloaded) | 2026-06-11 (UTC) |
| Snapshot recorded | 2026-06-17 (UTC) |

The date range extends to 2026-06-27 because the feed already lists the
scheduled WC 2026 fixtures; the leakage-free backtest only ever evaluates WC
2014 / 2018 / 2022 and trains each fold strictly on matches *before* that
tournament, so future fixtures never enter any training or evaluation set.

## Verify the checksum

```powershell
# Windows / PowerShell
Get-FileHash data\raw\results.csv -Algorithm SHA256
```

```bash
# Unix
sha256sum data/raw/results.csv
```

Expected: `50f17eb331a3d8367184f3314cf41616782f842ebb39e42191260b414b56bc78`

## Reproduce the reports

```bash
python src/export_backtest.py   # reports/backtest_predictions.csv + metrics summary
python src/make_figures.py      # reports/figures/figure4_*.png, figure5_*.png
python src/paired_tests.py      # appends McNemar + bootstrap rows to the metrics summary
```

`data/raw/` and `data/processed/` are gitignored; only this checksum and the
derived `reports/` outputs are tracked.
