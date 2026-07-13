# Remediation Phase 1 — canonical data policy + deterministic feature building

Recorded: 2026-07-13, branch `fix/scientific-audit`.
Baseline for all comparisons: `reports/remediation_baseline.md` (commit `096ad4c`).

## 1A. Duplicate policy (approved at Checkpoint 1A)

Full scan of `data/raw/results.csv` (frozen snapshot `59f49de6…`, 49,477 data
rows): zero exact byte-duplicates; exactly two same-`(date, home_team,
away_team)` groups (4 rows), matching the "4-row dedupe" narrative in
`notebooks/04_model_improvement.md` (which traced to `experiments.py`'s
key-only `drop_duplicates` — a rule that would also have silently deleted a
real match).

Adopted policy (implemented in `features.load_canonical_results`, the ONE
canonical load path):

1. **Drop redundant duplicates** — rows identical in all model-relevant
   columns `(date, home_team, away_team, home_score, away_score, tournament,
   neutral)`; keep the first occurrence after the deterministic stable sort.
   Removes exactly one row: the 2026-06-06 Gibraltar v Cayman Islands friendly
   entered twice with venue strings "Gibraltar" vs "Europa Point" (Europa
   Point is Gibraltar's stadium; `city` is consumed by nothing downstream).
   49,477 file rows -> 49,476.
2. **Keep** the 1974-02-17 Tahiti v New Caledonia pair (2-1 and 1-2): genuine
   same-day double-header within a multi-match series (a further friendly
   followed on 1974-02-20); upstream martj42 retains both. Processed within
   the date batch in canonical sort order.
3. **Keep** all 15 reversed-fixture same-date pairs (e.g. the 1969-02-02
   Uganda/Cameroon qualification legs dated identically; the twin Copa
   Newton / Copa Lipton trophy days): real matches with upstream date
   imprecision, not duplicates. This includes the odd 1925-05-20 China v
   Japan pair in Manila (Friendly + Far Eastern Championship Games, both
   2-0 China), which upstream also retains.

**Protocol-comparison note (for the eventual documentation): the published
pipeline (`features.py` as of baseline `2b1e5d8`) applied NO dedupe at all —
the published headline numbers (56.25 / 57.81 / 0.972 / p=0.58) were computed
on data containing both duplicate groups, including the double-counted
Gibraltar match inside the pre-cutoff training window.** `experiments.py`
applied a different, key-only dedupe, so the two consumers did not even agree
on the data.

## 1B. Deterministic feature building

* Canonical stable sort key: `(date, home_team, away_team, tournament,
  home_score, away_score, neutral)`, `kind="stable"`. The 4-column key
  specified at approval is not unique for the kept Tahiti double-header;
  scores + neutral as tiebreakers make the key provably unique after dedupe
  (any residual tie would be a model-relevant duplicate, already removed).
* Date-batched state semantics: every match on date D takes features from the
  identical start-of-date state; all of date D's updates apply after the
  batch, sequentially in canonical sort order (documented policy for a team
  appearing twice on one date). New-team registration happens at the start of
  each batch (order-independent set operation).
* The training cutoff (date < 2026-06-11) is now EXPLICIT in the loader. The
  published pipeline honoured it only by accident: post-cutoff fixtures had NA
  scores at fetch time and fell to `dropna`; the re-fetched raw file has 40 of
  those scores filled in, so a naive rebuild would have leaked in-tournament
  results into `features.csv`. This change converts an accidental exclusion
  of WC 2026 results into a structural one (accepted at Checkpoint 1B).
* `experiments.py` now sources scores through the canonical loader (cumcount
  alignment on the non-unique key), eliminating its divergent dedupe.

Row-count accounting on the frozen snapshot: 49,477 file rows -> 49,476 after
dedupe -> 49,445 scored -> 49,444 loaded (no cutoff) -> **49,404 canonical
training rows** (vs 49,405 published).

## Canonical hashes (recorded here; DATA_SNAPSHOT.md is updated in a later phase)

Serialization recipe identical to DATA_SNAPSHOT.md's primary checksum
(header + `|`-joined field tuples, lexicographically sorted, `\n`-terminated,
SHA-256), applied to the deduped file rows:

| Artifact | Rows | SHA-256 |
| --- | --- | --- |
| Cleaned full file (deduped) | 49,476 | `ebc1b65b2290e10a64677b1f45e0c7db634d6867b947ffe80a405de74b33cb77` |
| Cleaned pre-cutoff subset (date <= 2026-06-10) | 49,404 | `ee74e387f4a06c50db6c53264412eb1f1845d7afaaa517b5eada2d4dd9bf7fe8` |
| `data/processed/features.csv` (rebuilt) | 49,404 | `cdbfc5a6bcc83a6b71f57db1fcf674ec4acad1d5e0e9e1c9ca44b029bc56136d` |
| `data/processed/team_state.csv` (rebuilt) | 336 teams | `999a6f821daaf520e8fbb16d7bd2e10dc284359597dc581f65b808752dc47840` |
| `data/processed/elo_ratings.csv` (rebuilt) | 336 teams | `1857ba3b65fa176f808c025f404135ec58f775280b853e233e68dc40589b9058` |

(Supersedes the published pre-cutoff invariant `a9f564e6…` / 49,405 rows,
which was computed before dedupe.)

## Drift vs Phase 0 baseline

Aligned on `(date, home_team, away_team, tournament, outcome)` + within-key
rank; 49,404 of 49,405 old rows matched (the unmatched row is the dropped
Gibraltar duplicate).

* Feature cells changed: **171,855 of 1,482,120** (30 feature columns), on
  **45,686 rows (92.47%)** — consistent with the audit's ~94% estimate (the
  audit shuffled; this is one specific reordering).
* Changes concentrate in the state-path columns: `away_conf_elo` 37,702,
  `home_conf_elo` 37,658, `home_elo` 33,037, `away_elo` 32,901, `elo_diff`
  29,636; all other columns < 100 cells each.
* Per-feature-row |Elo delta| (home+away cells): 65,938 of 98,808 nonzero;
  **max 35.34** (audit estimated ~36), mean 0.082, median 0.020, p90 0.09,
  p99 0.75 (of nonzero cells).
* Final `elo_ratings.csv`: 39 of 336 teams changed; max |delta| 14.70, mean
  0.103, median 0.00. Same 336 teams in both.
* Backtest-fold rows (192 published WC 2014/2018/2022 rows): **177 of 192
  have >=1 changed feature cell**, but the per-row Elo movement there is tiny
  (max 0.09, mean 0.007) — the churn is dominated by conf_elo tie-order
  effects. Fold metrics are NOT re-run this phase; they move in Phase 4.

## Tests (tests/test_determinism.py) — all pass

Full-history run (`DETERMINISM_FULL=1`, 107 s) and bounded CI run (dates <
1980, whole suite 36 s):

1. `test_cleaned_row_counts` — policy counts pinned to the frozen snapshot.
2. `test_shuffle_within_date_invariance` — within-date shuffles (2 seeds) of
   the canonical frame rebuild byte-identical features/team_state/elo CSVs.
3. `test_shuffled_raw_file_roundtrip` — a within-date-shuffled RAW FILE
   through the full loader+build path is byte-identical end-to-end.
4. `test_idempotence` — building twice from the same input is byte-identical.

Pre-existing suite unaffected: 25/25 tests pass.

## Smoke test

`python src/smoke_test.py`: **all checks pass**, including the serving-state
and leakage-cutoff checks against the frozen (untouched) model bundle — the
rebuilt `team_state.csv` drift was small and every check is semantic
(dates/consistency), not hash-based. No serving mismatch to carry forward.
