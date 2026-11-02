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

# ENT-306: reproducibility, serving state, MatchContext, figures, merge

Recorded: 2026-11-02 (Sydney), branch `fix/scientific-audit` at 25e0e56.

## Gate

- HEAD 25e0e56 descends from base commit 38229c3: yes.
- Suite at 25e0e56: 63 passed (rerun recorded in 7a/7b); smoke test passes.
- Planned work is inside Preflight scope items 4 to 6 (pipeline rerun,
  tests, website-style figures; MatchContext serving migration; merge into
  master and push).
- Gate for ENT-306: **PASS**, recorded before the first mutation.

## A. Reproducibility check (served model never replaced)

Frozen copies of `models/`, `data/raw/` and `data/processed/` were taken to
the scratch folder first (sha256 list recorded), because none of the
pipeline scripts can write elsewhere.

- `src/fetch_data.py` (1 s): all three raw files already present, skipped,
  so the rerun used the frozen snapshot (`results.csv` sha256 59f49de6...,
  equal to `raw_results_sha256`).
- `src/features.py`: 49,404 rows; `features.csv` sha256 cdbfc5a6... equal to
  `features_csv_sha256`; `elo_ratings.csv` and `team_state.csv`
  byte-identical to the frozen copies.
- `src/train.py` (8 s): best iteration 175; rebuilt bundle sha256
  52752a62...eb4d, equal to `bundle_sha256`.
- Verdict: **frozen bundle reproducible from source** (on the frozen raw
  snapshot). `train.py` rewrote `models/model-manifest.json` (only
  `created_utc` and `code_git_sha` differed); it was restored from the
  frozen copy and all eight frozen files re-verified against the recorded
  sha256 list.
- `backtest.py` (2m36s), `export_backtest.py` (2m36s), `paired_tests.py`
  (2 s), `protocol_comparison.py` (34 s), `make_by_tournament.py` (3 s):
  every report CSV byte-identical. The only change was
  `reports/figures/figure5_by_tournament.png`, re-rendered as a side effect;
  the committed PNG dated from 2026-07-11, before the Phase 4 remediation, so
  the committed figures 3 to 6 carried pre-remediation numbers. Restored, then
  replaced in D.
- Suite after restoring: 63 passed (65 with the two serving-state tests of
  B); smoke test all checks pass.

**Finding: the deployed cold start did not serve the frozen bundle.**
`app/streamlit_app.py` bootstrapped on Streamlit Community Cloud (where
`data/` and `models/` were untracked) by downloading the current upstream
`results.csv`, building features with its own loader (no canonical dedupe,
unstable sort) and retraining. Replicating that path on the frozen raw
snapshot in scratch gave 49,405 rows (the duplicated Gibraltar friendly
kept), `features.csv` 1efda71e... and a bundle 1bed5d1c... with best
iteration 150. On Cloud the upstream data also differs. So although training
is reproducible from source, the deployed app served a different model from
the one behind the ledger and the article, which is why B was done.

## B. Serving-state integrity (commit ad1f109)

- Tracked byte for byte: `models/xgb_wc2026.joblib` (1,577,761 B),
  `data/processed/features.csv` (8,810,518 B), `elo_ratings.csv` (8,012 B),
  `team_state.csv` (20,660 B); all far below 100 MB. `data/raw/` stays
  ignored; `.gitignore` re-includes exactly these files.
- The CSVs were written with CRLF and the manifest hashes those bytes, so
  `.gitattributes` unsets `text` for them; the staged blob sha256 values were
  checked equal to the manifest before commit (a Linux checkout would
  otherwise get LF bytes and fail the hash check).
- The cold-start retraining is removed. The app verifies the bundle and
  `features.csv` against the manifest at start-up and stops with an
  explanation if anything is missing or different; it never retrains.
- Tests: frozen serving files match the manifest; no line-ending conversion.
  Headless app run (Streamlit AppTest): loads in 14.6 s, no exception.

## C. MatchContext serving migration (commit 5bb8051)

- `Predictor.feature_X(home, away, neutral=1)` is the contract's
  three-argument `build_x`; `Predictor.predict` calls `predict_match_proba`
  with a neutral `MatchContext`; the new `Predictor.predict_ctx(ctx)` serves
  both paths (neutral returns `predict`; host is one orientation with the
  HFA-shifted prior).
- Match Predictor page: a Venue choice (neutral, or the home team at home)
  builds a `MatchContext`. Behaviour changes on the host path only: the old
  hand-rolled path set `is_world_cup` to 0 and used an unshifted Elo prior;
  it now follows the contract (`is_world_cup` 1, prior shifted by HFA 125.58).
- The deprecated `predict_neutral_proba` shim had no remaining user and is
  removed; its byte-equivalence test is replaced by tests on the real
  serving state.
- Byte identity: full-precision probabilities for all 104 cached fixtures
  (the tracker path) and the four ledger rows recomputed with
  `prediction_ledger`'s functions are identical before and after, and all
  four ledger rows equal the ledger strings exactly. New tests pin two
  neutral vectors, the host path and the four ledger rows.
- App check: neutral Argentina v Spain 35.7 / 26.4 / 37.9 (equal to ledger
  M104); Argentina at home 46.9 / 26.2 / 26.9; no exception.
- Suite 67 passed; smoke test passes.

## D. Website figures (commit e3d6cee) and residual fixes (commit 756377c)

- Figures 3, 4, 5 and 6 regenerated from the remediated CSVs and figure 10
  added (final live evaluation: all matches, 4 logged before kickoff, 100
  recomputed; Blend and Elo accuracy and log-loss; Blend minus Elo log-loss
  with the paired bootstrap interval). Each is written as
  `<name>-light|dark.png|svg`, 1600 px wide, transparent, with the website
  inks and the series colours of the website's `THEME_INK` (amir42-web
  `scripts/editorial-art/figures.ts`) mapped to Raw XGBoost / Blend / Elo
  baseline; figure 6 uses the same hues for outcomes with marker shapes and a
  legend. Labels 48 px, notes 40 px, figures 44 px. The old dark-navy PNGs
  are deleted. Titles and findings are left to the article captions.
- Figure 5 is now a dot plot: the per-tournament differences are a few
  thousandths, which bars could only show with a truncated baseline. Its old
  subtitle ("a 2014 effect that reverses in 2018") no longer matched the
  remediated numbers (the Blend also has the lower log-loss in 2022), and the
  script's printed verdict is now derived from the data.
- Rendering is deterministic (a second render left git clean). SVG text stays
  text (`svg.fonttype none`).
- `reports/live_evaluation.csv`: the `correct` column is an integer;
  regenerated with `--final`, every value identical (only `68.0` to `68`).
  EXC-1 in the Preflight JSON now says the files are excluded through
  `.git/info/exclude`.

## E. README (commit 669bde7) and merge

The README carries the remediated backtest (Blend 53.13%, Elo 56.25%,
log-loss 0.9818 vs 0.9859, McNemar p 0.21, bootstrap -0.004 [-0.025,
+0.018]), the per-tournament table, the final live evaluation (Blend 68/104,
Elo 70/104, log-loss 0.8334 vs 0.8666, bootstrap -0.033 [-0.059, -0.005]),
the frozen-serving note, and the blend weight 0.75 as a kept convention: the
tune-fold search found its minimum at 1.0 and put 0.75 within about one
bootstrap standard error of it (0.9118 vs 0.9071, standard error of the
difference 0.0048).

Merge into master: **held for explicit owner authorization.** Pushing master
redeploys the public Streamlit app, which is a deployment step; DEC-166,
ENT-6 and Update 40cb7da3... do not state it. The branch is pushed; nothing
is merged.

## ENT-306 residuals

1. The website series palette fails the generic palette validator: light
   green #4f8f6a below the chroma floor; the dark set outside the lightness
   band, with dark blue and green below the normal-vision floor (14.9 < 15);
   orange against green in the 6 to 8 colour-vision band. Kept unchanged for
   consistency with the site's other figures and mitigated with hatches,
   marker shapes, legends and direct labels. Revisit in the website figure
   contract (amir42-web), not here.
2. Inter is not installed locally, so the figures use DejaVu Sans (the agreed
   fallback) and the SVGs name DejaVu Sans. Re-render if Inter is installed.
3. `article/wc2026-predictor-article.md` (the old in-repo draft) still links
   the deleted figure files and carries pre-remediation numbers. The
   publication text lives in the CMS (ENT-307 onwards); decide there whether
   to delete it or mark it superseded.
4. Two pre-existing em dashes remain in untouched helper docstrings of
   `src/make_figures.py` and one in a comment of `src/make_by_tournament.py`.
