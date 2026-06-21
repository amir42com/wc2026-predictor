# Data snapshot — backtest reproducibility

The backtest metrics and figures in `reports/` were produced from a fixed
snapshot of the raw match dataset. Recording the snapshot's checksum here makes
the published numbers time-stable: anyone with a file whose SHA256 matches the
value below will reproduce the same 192-match backtest exactly.

## Source dataset (`data/raw/results.csv`)

International football results, downloaded from GitHub
(`martj42/international_results`, `master` branch — fetched directly from
`raw.githubusercontent.com`) via `src/fetch_data.py`.

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
python src/export_backtest.py     # reports/backtest_predictions.csv + metrics summary
python src/make_figures.py        # figures 3 (table), 4 (accuracy), 6 (reliability)
python src/make_by_tournament.py  # backtest_by_tournament.csv + figure 5 (by tournament)
python src/paired_tests.py        # appends McNemar + bootstrap rows to the metrics summary
```

`data/raw/` and `data/processed/` are gitignored; only this checksum and the
derived `reports/` outputs are tracked.

## Live tracker target (football-data.org)

The Prediction Tracker scores the model on its **training target**: the result
*after extra time, excluding penalty shootouts*. football-data.org's v4
`score.fullTime` **includes** extra-time goals (verified against Euro 2024:
England 2-1 Slovakia = `regularTime` 1-1 + `extraTime` 1-0, `duration`
`EXTRA_TIME`), so extra-time knockouts already match the target. For a
**penalty shootout**, `fullTime` folds in the shootout tally (verified:
Portugal-Slovenia, `regularTime`/`extraTime` 0-0, `penalties` 3-0, `fullTime`
3-0, `duration` `PENALTY_SHOOTOUT`); `fetch_results.score_matches` strips the
shootout and records the after-ET **draw**, so the recorded outcome matches the
target. Residual presentation note: such a match shows its after-ET score (e.g.
0-0) in the tracker, with the API `winner`/`duration` retained in the cached
payload for audit — the W/D/L the model is graded on is the draw.
