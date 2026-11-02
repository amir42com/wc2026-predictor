# Remediation Phase 7: final results, completeness fixes, live evaluation

Recorded: 2026-11-02 (Sydney), branch `fix/scientific-audit`.
Task: ENT-305 (verify the live evaluation code, ingest the official final
results, run the final 104-match evaluation) under the ENT-6 publication work.

## Preflight and Gate

- Preflight Record: `docs/project-preflight/ent-6-publication.json`,
  committed alone in 22ee5b1 on top of base commit 38229c3.
- Baseline at 38229c3: clean tree, HEAD equal to `origin/fix/scientific-audit`.
  Full suite 57 passed (the Phase 6 report recorded 55; commit c84c008 added
  two fixture-ID join tests after that report). Smoke test: all checks pass.
- Gate for ENT-305: **PASS**. The Preflight commit exists, the baseline tests
  pass, and the planned work stays inside the recorded scope (completeness and
  per-tier fixes in `src/live_evaluation.py`, results refresh, final
  evaluation). `src/` was not touched before this Gate entry.

## 7a. Completeness and per-tier fixes (src/live_evaluation.py, commit 1d5dc65)

Two defects found on review of the Phase 6 evaluator:

1. `final_mode_check` counted only valid results. A complete count could
   still hide a fixture that failed to join its prediction, or a dropped
   record. Final mode now requires all of: exactly 104 valid finished
   fixtures, exactly 104 joined rows covering those fixtures once each, zero
   validation problems and zero join problems. A refusal names the fixture ids
   that are not valid/finished, unjoined, or joined more than once.
2. Metrics were pooled only, so the 4 prospective rows were never reported
   apart from the 100 retrospective ones. `evaluate()` now also returns the
   same per-system metrics for each tier that has rows, and
   `write_final_report()` writes a `scope` column with one block per tier
   after the pooled block. McNemar and the paired bootstrap run only where
   n >= 30 (`MIN_PAIRED_N`); below that the block reports correct/n,
   log-loss and Brier and marks the paired tests "not applicable".

Model, features, blend weight and evaluation protocol are unchanged.

Tests (`tests/test_live_evaluation.py`, synthetic only): a join problem
refuses final mode and names the fixture; a missing prediction refuses and
names the fixture; a validation problem refuses and names the fixture; the
per-tier numbers are hand-checked (4 prospective rows: Blend 3/4, LL
mean(-log[.6,.5,.7,.2]), Brier 0.43625; Elo 1/4, Brier 0.785; 30
retrospective rows with LL difference log 0.8); an absent tier is omitted;
the CSV holds pooled, prospective and retrospective blocks with the
not-applicable marker. The existing pooled 3-row test now asserts the paired
tests are not applicable at that size.

- `uv run python -m pytest -q`: 63 passed (baseline 57; 6 new tests).
- `uv run python src/smoke_test.py`: all checks passed.
- Rerun after the results refresh and final evaluation: 63 passed, smoke
  test all checks passed.

## 7b. Results refresh and official-source checks (commit 57e4816)

- `sha256(models/xgb_wc2026.joblib)` =
  `52752a62a0ac53e927e65ac37faeef6bb24c68591891ebd78d8721eb6ee9eb4d`, equal to
  `bundle_sha256` in `models/model-manifest.json`. No retraining.
- `uv run python src/fetch_results.py`: cache refreshed
  (fetched_at 2026-11-02T06:23:55+00:00), 104 FINISHED matches (was 102),
  2026-06-11 to 2026-07-19. Durations: 95 REGULAR, 5 EXTRA_TIME,
  4 PENALTY_SHOOTOUT; every shootout record carries a level after-ET score.
- Third place, 2026-07-18: cache `France 4-6 England`, REGULAR, winner
  AWAY_TEAM (England). FIFA match centre "France v England 4-6" and FIFA
  match report (Miami Stadium; England winner by Bellingham at 90+8, so no
  extra time). Matches the expected England 6-4 France.
- Final, 2026-07-19: cache `Spain 1-0 Argentina`, EXTRA_TIME, fullTime 1-0,
  winner HOME_TEAM (Spain). FIFA match centre "Spain v Argentina 1-0" and
  FIFA final report (0-0 after 90 minutes, Ferran Torres 106'). Matches the
  expected Spain 1-0 Argentina after extra time.

## 7c. Final live evaluation (commit 598ee84)

`uv run python src/live_evaluation.py` (interim) and
`uv run python src/live_evaluation.py --final` printed identical metrics;
final mode passed the completeness gate with no PROBLEM lines and wrote
`reports/live_evaluation.csv`.

    Evaluated 104 fixtures (4 prospective / 100 retrospective)
    [pooled] n=104
      Blend         68/104 correct  acc 65.38%  ll 0.8334  brier 0.4873
      Elo baseline  70/104 correct  acc 67.31%  ll 0.8666  brier 0.5073
      McNemar p=0.5000 (b=0, c=2); bootstrap LL diff -0.0333 [-0.0590, -0.0054]
    [prospective] n=4
      Blend         3/4 correct  acc 75.00%  ll 0.8922  brier 0.5246
      Elo baseline  3/4 correct  acc 75.00%  ll 0.8695  brier 0.5129
      paired tests not applicable (n=4 < 30)
    [retrospective] n=100
      Blend         65/100 correct  acc 65.00%  ll 0.8310  brier 0.4858
      Elo baseline  67/100 correct  acc 67.00%  ll 0.8665  brier 0.5071
      McNemar p=0.5000 (b=0, c=2); bootstrap LL diff -0.0355 [-0.0626, -0.0065]

Reading: on accuracy the Elo baseline is 2 picks ahead (the only two
discordant fixtures both went to Elo; McNemar p=0.50, no evidence of a
difference). On probability quality the Blend is better pooled and
retrospective: lower log-loss (95% bootstrap CI of the difference excludes
zero) and lower Brier. The 4-row prospective tier is too small for inference.

Prospective ledger rows against the results (pick = argmax of the ledgered
vector; probabilities team_a / draw / team_b):

| Fixture | Ledger pairing | Blend pick | Elo pick | Result | Actual |
|---|---|---|---|---|---|
| M101 | France v Spain | Spain (28.1/24.5/47.4) | Spain (31.0/22.7/46.3) | France 0-2 Spain | Spain |
| M102 | Argentina v England | Argentina (46.5/26.4/27.1) | Argentina (50.2/22.7/27.1) | England 1-2 Argentina | Argentina |
| M103 | England v France | France (33.8/25.4/40.8) | France (33.5/22.7/43.8) | France 4-6 England | England |
| M104 | Argentina v Spain | Spain (35.7/26.4/37.9) | Spain (37.6/22.7/39.7) | Spain 1-0 Argentina (aet) | Spain |

All four were ledgered before kickoff (generated_before_kickoff=true; M101
and M102 on 2026-07-13, M103 and M104 on 2026-07-16).

## Evidence

- `reports/prediction_ledger.csv`: `git diff 38229c3 --stat` empty after
  every step; sha256 `c45e28c5...0999` identical to the 38229c3 blob.
- No file created outside the repository; the tests' temporary directories
  are created and removed by `tempfile.TemporaryDirectory`.
- Commits on top of 38229c3: 22ee5b1 (Preflight), 1d5dc65 (fixes),
  57e4816 (results), 598ee84 (evaluation), then this report.

## Residuals

1. `reports/live_evaluation.csv` writes the `correct` column as a float
   (`68.0`) because the paired-test rows leave it empty. Cosmetic; fix with an
   integer cast when the reporting code is next touched (ENT-307 pipeline
   rerun).
2. Preflight EXC-1 says AGENTS.md and CLAUDE.md are "gitignored"; they are
   excluded locally through `.git/info/exclude`, not `.gitignore`. Same
   effect (untracked); correct the wording when EXC-1 is revisited at the
   ENT-306 merge.
3. The ledger has no `kickoff_utc` values (date-level before-kickoff bound,
   as recorded in Phase 6). Unchanged by design.
