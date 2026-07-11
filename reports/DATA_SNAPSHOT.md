# Data snapshot — backtest reproducibility

The backtest metrics and figures in `reports/` were produced from a frozen view
of the raw match dataset. **The model cutoff is 2026-06-11**: training features
(`data/processed/features.csv`) end at 2026-06-10, and no match on or after the
cutoff enters any training or evaluation set. That cutoff — not the raw file's
byte-level state — is what the published numbers depend on.

The upstream feed (`martj42/international_results`, `master` branch) pre-loaded
the full WC 2026 fixture list in April 2026 with `NA` scores, so in-tournament
updates fill scores into *existing* rows rather than appending new ones. The raw
file's bytes therefore drift with every upstream update while its row count,
columns, and date range stay constant. By construction that drift is confined to
rows dated on/after the cutoff and cannot affect the model. Reproducibility is
pinned accordingly: on the pre-cutoff subset first, the whole file second.

## Primary checksum — frozen training subset (the invariant)

| Field | Value |
| --- | --- |
| Subset | rows of `data/raw/results.csv` with `date <= 2026-06-10` |
| Rows | 49,405 |
| **Subset SHA256** | `a9f564e6e38a316f2797ace3e1c438a6177694d688b08c6c93b7e01195f4f3c0` |

Anyone can reproduce the model from **any** upstream snapshot: filter to the
cutoff, hash as described under "Verify the checksums" below, and compare to
this value. If it matches, the training window is byte-for-byte identical to
the one the published results were built from.

## Secondary checksum — raw file as shipped

| Field | Value |
| --- | --- |
| File | `data/raw/results.csv` |
| **SHA256** | `59f49de6055179ebe8c7f4d9f31b579c938f9eafd25e925abdffd91885f08e43` |
| Upstream commit | `b062f3478c84ccf455acbb05a150a391862daf43` (2026-06-22 UTC) |
| Size | 3,724,319 bytes |
| Rows | 49,477 match results |
| Columns | `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral` |
| Date range | 1872-11-30 → 2026-06-27 |

This whole-file hash **will drift** whenever the raw file is re-fetched and
upstream has filled in more results. Such drift is post-cutoff by construction
(score fill-ins on pre-loaded fixture rows) and does not affect the model; the
primary subset checksum above is the value that must not change.

## Provenance history

| Recorded | Raw-file SHA256 | Upstream commit | Notes |
| --- | --- | --- | --- |
| 2026-06-17 | `50f17eb331a3d8367184f3314cf41616782f842ebb39e42191260b414b56bc78` | `c636851f6e388d7aabd1feabbd4dad94e7e6e266` (2026-06-11) | Original documented snapshot; 3,724,399 bytes. |
| 2026-07-11 | `59f49de6055179ebe8c7f4d9f31b579c938f9eafd25e925abdffd91885f08e43` | `b062f3478c84ccf455acbb05a150a391862daf43` (2026-06-22) | Re-fetch to capture 40 WC group-stage results (repo commit `4a8fbd8`). Supersedes the entry above. |

The 2026-06-11 → 2026-06-22 delta was verified row-by-row against both upstream
commits: **exactly 40 rows differ, all dated 2026-06-11 to 2026-06-21, zero rows
added or removed.** Every differing row is a WC 2026 fixture whose `NA,NA` score
was filled in with the real result. Zero rows on or before 2026-06-10 differ,
and the pre-cutoff subset hashes identically (`a9f564e6…`) in both files. Both
raw-file hashes are genuine byte-exact upstream states, recovered from the
upstream repository's history.

## Verify the checksums

Whole file (secondary, as-shipped):

```powershell
# Windows / PowerShell
Get-FileHash data\raw\results.csv -Algorithm SHA256
```

```bash
# Unix
sha256sum data/raw/results.csv
```

Expected: `59f49de6055179ebe8c7f4d9f31b579c938f9eafd25e925abdffd91885f08e43`
(only valid for the exact 2026-06-22 upstream state; a mismatch here alone is
not an error — check the subset hash below).

Training subset (primary, the invariant) — filter to the cutoff, stable-sort,
hash. Deterministic recipe: parse the CSV (UTF-8), keep rows with
`date <= 2026-06-10`, sort rows lexicographically as field tuples, then SHA256
the header and each row joined with `|` and terminated with `\n`:

```python
import csv, hashlib

with open("data/raw/results.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = [tuple(r) for r in reader if r[0] <= "2026-06-10"]

h = hashlib.sha256()
h.update(("|".join(header) + "\n").encode())
for row in sorted(rows):
    h.update(("|".join(row) + "\n").encode())
print(len(rows), h.hexdigest())
```

Expected: `49405 a9f564e6e38a316f2797ace3e1c438a6177694d688b08c6c93b7e01195f4f3c0`

## Reproduce the reports

```bash
python src/export_backtest.py     # reports/backtest_predictions.csv + metrics summary
python src/make_figures.py        # figures 3 (table), 4 (accuracy), 6 (reliability)
python src/make_by_tournament.py  # backtest_by_tournament.csv + figure 5 (by tournament)
python src/paired_tests.py        # appends McNemar + bootstrap rows to the metrics summary
```

The leakage-free backtest only ever evaluates WC 2014 / 2018 / 2022 and trains
each fold strictly on matches *before* that tournament, so the pre-loaded
future fixtures never enter any training or evaluation set.

`data/raw/` and `data/processed/` are gitignored; only these checksums and the
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
