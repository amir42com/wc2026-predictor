# Remediation Phase 3 — blend-weight search + production bundle retrain

Recorded: 2026-07-13, branch `fix/scientific-audit`.

## Search protocol (the Phase 4 target protocol, on tune folds only)

`src/blend_weight_search.py`, tune folds WC 2006 + WC 2010 ONLY (the
2014/2018/2022 benchmark remains an untouched Phase 4 holdout; the
PROTOCOL-INCOMPLETE gate stayed down throughout). Per fold: fresh model
trained on canonical matches strictly before the tournament's first fixture
(production `train_model`, chronological 10% early-stopping tail); ONE frozen
pre-tournament team-state/H2H snapshot rebuilt via `features.build_features`
on truncated canonical data (serving semantics — recorded in-tournament rows
are never used, so no within-tournament state updates); Phase 2 MatchContext
from recorded venue flags (host rows: Germany 2006 ×7, South Africa 2010 ×3,
single-orientation with HFA_ELO = 125.58); production prior (draw rate 0.227,
HFA on host rows). Sweep w = 0.00…1.00 step 0.01; selection metric pooled
log-loss (the original selection story); accuracy/Brier recorded as secondary.

Artifact: `reports/blend_weight_search.csv` (101 rows; header metadata: code
SHA, canonical features hash `cdbfc5a6…`, HFA, draw rate, date, argmin,
bootstrap SEs). This is the artifact the article's selection narrative refers
to; it did not previously exist.

## Result and pre-registered decision

* **Argmin: w = 1.00** (pure symmetry-averaged XGB), pooled LL 0.907081.
* **w = 0.75: pooled LL 0.911846 — delta 0.004765.**
* Curve shape: no sharp minimum. Pooled LL decreases monotonically from
  w = 0 (0.9426) to w = 1 (0.9071); the entire 101-point curve spans 0.035,
  i.e. less than ONE bootstrap SE of the minimum (0.0425, 10,000 paired
  resamples, seed 12345). Under the much tighter paired-difference SE
  (0.00478) the band is [0.75, 1.00] with 0.75 exactly at its edge. The two
  folds disagree: 2006 prefers w=1 monotonically; 2010 has a shallow interior
  minimum near w≈0.5 — the pooled slope is 2006's.
* **Decision per the pre-registered rule: KEEP ELO_BLEND_W = 0.75.** The
  argmin was NOT 0.75 — stated plainly — but 0.75 sits well within one
  bootstrap SE of the minimum (0.0048 ≪ 0.0425), which is exactly the rule's
  keep condition. The corrected search corroborates the shipped weight; it
  does not vindicate it as optimal. The tune data alone cannot distinguish
  any weight in [0, 1] at one-SE resolution on 128 matches.

## Picks-flip (audit C6 re-verification, TUNE folds only)

At w = 0.75 vs raw XGB (w = 1.0): **4 of 128 hard picks differ** (2 in 2006,
2 in 2010). The C6 finding — the blend barely changes hard predictions —
replicates under the corrected protocol. Benchmark-fold picks-flip is Phase 4.

## Production bundle retrain (once)

Old bundle backed up and verified against the Phase 0 baseline hash
(`a81bca98…`). Retrained via `python src/train.py --production-only` (new
flag: trains and ships ONLY the production bundle — the legacy default path's
pre-2018 eval model and its 2018+ read-out were skipped, keeping benchmark-era
evaluation out of this phase).

* Training input: all 49,404 canonical pre-cutoff rows (production_mask
  excludes 0 rows — the canonical loader already ends at 2026-06-10).
* Production settings unchanged; best_iteration 175 (old bundle: 130, read
  from the backed-up baseline bundle — recorded for Phase 4's metric
  attribution).
* Article §4 rewording flagged for Phase 8 (checkpoint note): 0.75 must be
  presented as a frozen convention that the corrected search corroborates as
  indistinguishable from optimal (weight unidentifiable at n=128), not as a
  discovered optimum.
* New bundle `models/xgb_wc2026.joblib` SHA-256 `52752a62a0ac53e927e65ac37fae
  ef6bb24c68591891ebd78d8721eb6ee9eb4d`.
* Serving artifacts: `features.py` re-run — features.csv / team_state.csv /
  elo_ratings.csv byte-identical to the Phase 1 hashes (`cdbfc5a6…` /
  `999a6f82…` / `1857ba3b…`), as expected (they are bundle-independent;
  idempotence held).
* `models/model-manifest.json` (force-tracked; models/ is gitignored) records:
  bundle hash, training rows, features.csv + raw results.csv hashes,
  ELO_BLEND_W / ELO_DRAW_RATE / HFA_ELO, best_iteration, n_features, xgboost
  version, and the search artifact path. `tests/test_model_manifest.py`
  asserts the manifest matches the on-disk bundle, data, and constants.
  Note: `code_git_sha` records HEAD at train time (`018c6c3`); the commit
  containing this manifest is the true code state (inherent chicken-and-egg).
* NOT deployed; no push; live app untouched. Serving cutover happens at merge,
  atomically.

## Verification

Full suite 34/34 (new manifest test included). `src/smoke_test.py`: all
checks pass against the NEW bundle (serving-state, leakage cutoff, scorelines,
SHAP reconcile, Annexe C, Monte Carlo). Serving outputs shifted as expected
with the retrained bundle (e.g. the smoke fixture's probabilities moved by a
few points); per the quotability rule these are not performance claims.

## Quotability

Tune-fold metrics above are selection machinery, not article numbers. The only
quotable outputs of this phase: **ELO_BLEND_W stays 0.75**, and the artifact
exists at `reports/blend_weight_search.csv`. Benchmark numbers arrive only
when Phase 4 lifts the gate.
