# 04 — Trying to beat the Elo baseline on World Cup matches

**Goal:** the naive Elo baseline ("higher-Elo team wins", probabilities from
the Elo logistic formula) beat our XGBoost model on WC 2014/2018/2022:
**57.8% vs 54.2% combined accuracy**. Close that gap without ever letting a
model, ensemble weight, or calibrator see data from after the tournament
being evaluated.

**Harness:** `src/experiments.py` — walk-forward WC 2014/2018/2022 exactly
like `src/backtest.py` (training cutoff = each tournament's first match
date). All ensemble weights, draw-boost factors, and selection decisions are
tuned on **WC 2006 + 2010 only** (strictly pre-2014). XGB probabilities per
(variant, year) are cached in `data/processed/exp_cache/`.

**Reference (combined, 192 matches):**

| Variant | Acc | Log-loss | Brier |
|---|---|---|---|
| Base XGB | 54.2% | 0.9897 | 0.5859 |
| Elo baseline | **57.8%** | **0.9769** | **0.5770** |

---

## E1 — Specialized training set

Hypothesis: WC matches are neutral-venue; a model trained on home-heavy data
learns a home-advantage prior that misfires at tournaments.

| Variant | Train rows (2014) | Acc | Log-loss |
|---|---|---|---|
| E1a: neutral OR non-friendly | 24,554 | 54.2% | 0.9879 |
| E1b: **neutral-venue only** | 9,527 | **57.8%** | 0.9926 |

**E1b ties the baseline accuracy (+3.6 pp over base XGB)** — the
home-advantage prior was genuinely hurting. But with only ~10k training rows
the model is overconfident: log-loss is worse than both base XGB and the
baseline. **Kept as a component for blending.**

## E2 — Ensemble XGB + Elo baseline

Weight tuned on WC 2006+2010: **w = 0.75 XGB / 0.25 Elo**.

| Variant | Acc | Log-loss | Brier |
|---|---|---|---|
| 0.75·XGB + 0.25·Elo | 55.2% | 0.9809 | 0.5804 |

Strictly better than raw XGB on all three metrics (+1.0 pp, −0.009 LL,
−0.006 Brier). Does **not** overtake the baseline's log-loss on the clean
rerun (see "noise" below). **Kept — adopted into production.**

## E3 — Extended features  ❌

Added to `features.py` (all strictly pre-match): rest days
(`*_days_since_last`), competitive-only form (`*_comp_wr_10`),
qualification record over 730 days (`*_qual_wr`), tournament match number as
a stage proxy (`*_tourn_match_n`).

Result: 53.6% / 0.9835 — **no improvement; discarded from the model** (the
columns remain available in `features.csv` for future work, but are not in
`NUMERIC_COLS`).

Set-piece goals: **not available** — `goalscorers.csv` only carries
`own_goal` and `penalty` flags, no corner/free-kick provenance. Skipped.

## E4 — Poisson goal model  ⭐

Independent-Poisson scoreline model: one `PoissonRegressor` on
team-perspective rows (`goals_for ~ elo_diff/400 + home_advantage`), W/D/L
from the scoreline grid.

| Variant | Acc | Log-loss |
|---|---|---|
| Poisson alone | 57.3% | 0.9813 |
| 0.30·XGB + 0.70·Poisson | 55.7% | 0.9771 |

A 2-feature Poisson model nearly ties the baseline — strong signal that
Elo difference is essentially the whole story at World Cups.

## E5 — Isotonic calibration  ❌

Per-class isotonic regression fitted on the last 4,000 pre-tournament
matches: 54.7% / 0.9982 — **worse; discarded.** 4,000 matches is too few to
fit three reliable isotonic maps, and the calibration window (mostly
qualifiers/friendlies) doesn't match the WC distribution.

## E6–E7 — Multi-way blends

Simplex grids over {neutral-XGB, base XGB, Poisson, Elo} tuned on 2006/2010
consistently selected ~0.5·neutralXGB + 0.5·Poisson (Elo weight → 0, since
the Poisson model is itself Elo-driven): 57.3% / 0.9820. A draw-boost
multiplier tuned for accuracy selected k=1.0 (no boost) — inflating draw
probabilities never paid off even on the tune set.

## E8 — Dixon-Coles Poisson  ⭐

Time-decayed fit (8-year half-life) + DC low-score `rho` correction fitted
by maximum likelihood (rho ≈ −0.06 every year — mild positive draw
dependence, stable across decades).

| Variant | Acc | Log-loss |
|---|---|---|
| **DC-Poisson alone** | **57.8%** | 0.9865 |
| 0.45·neutralXGB + 0.55·DC-Poisson | 57.3% | 0.9844 |

**DC-Poisson ties the baseline accuracy in every tournament** (60.9 / 57.8 /
54.7) — its picks agree with the Elo ranking essentially everywhere.

## E9–E10 — Accuracy-targeted selection

Selecting blends by tune-set *accuracy* instead of log-loss picked
0.9·neutralXGB + 0.1·Elo → 57.3% on eval. Tying configs never generalized
into beating configs.

---

## A finding about noise

The E2 blend scored 0.9732 log-loss with one training realization and 0.9809
with another (a 4-row dedupe shifted XGBoost's early-stopping point).
**Training stochasticity moves combined log-loss by ±0.01 — larger than the
gap between every top method and the baseline.** Any claimed sub-0.01 edge
on 192 matches is unverifiable.

## Verdict

| | Acc | Log-loss | Brier |
|---|---|---|---|
| Elo baseline | 57.8% | 0.9769 | 0.5770 |
| Old production (raw XGB) | 54.2% | 0.9897 | 0.5859 |
| **New production (0.75·XGB + 0.25·Elo)** | **55.2%** | **0.9809** | **0.5804** |
| Best accuracy configs (E1b, DC-Poisson) | 57.8% (tie) | ≥0.9865 | ≥0.5825 |

1. **The baseline was not beaten.** Two configurations tie its accuracy
   (neutral-only XGB; DC-Poisson); none beats it on any metric robustly.
   At World Cups, "pick the higher-Elo team" appears to sit at the edge of
   what's achievable from public match-history data — every strong model's
   picks converge to the Elo ranking, and deviations don't pay out-of-sample.
2. **Production still improved.** The 0.75/0.25 Elo-prior blend (E2) is
   adopted in `simulate.Predictor` and `backtest.py`: it beats the old raw
   model on accuracy, log-loss, and Brier in both training realizations, and
   closes most of the calibration gap to the baseline. The Monte Carlo
   simulator consumes probabilities, not picks, so this directly improves
   the app.
3. **Kept:** Elo-prior blend (production), `src/experiments.py` (harness),
   extended feature columns (data only). **Discarded:** extended features in
   the model, isotonic calibration, draw boost, specialized-only and
   Poisson-only production candidates.

Reproduce: `python src/experiments.py all` (or any of `e1`–`e10`).
