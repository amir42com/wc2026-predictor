# FIFA World Cup 2026 Prediction App

Machine-learning match-outcome predictor for the 2026 FIFA World Cup (USA / Canada / Mexico), built in Python with a Streamlit dashboard. It predicts win / draw / loss probabilities for any fixture, simulates the full 48-team tournament, explains predictions with SHAP, and grades its own pre-tournament forecasts against live results.

The project is deliberately framed around an honest finding: on historical World Cup matches, an XGBoost-plus-Elo blend **does not significantly beat a plain Elo baseline**. Over the 104 matches of WC 2026, Elo picked two more winners, while the blend's probabilities scored better on log-loss. See [Results](#results).

> Version is tracked in [`VERSION`](VERSION) (currently `2.0`).

## What it does

- Downloads and processes historical international results (1872–present) from [martj42/international_results](https://github.com/martj42/international_results)
- Engineers features: Elo ratings, recent form (5/10 games), head-to-head records, home/neutral venue, confederation strength, and a World Cup flag
- Trains an XGBoost classifier, then **blends it with an Elo logistic prior** (75% XGBoost / 25% Elo, with a fixed 0.227 draw share). The 0.75 weight is a kept convention, not a tuned optimum: a search on the 2006 and 2010 tune folds found its minimum at 1.0 (pure XGBoost) and put 0.75 within about one bootstrap standard error of it (pooled log-loss 0.9118 against 0.9071), so the search does not contradict the convention
- Backtests leakage-free against WC 2014 / 2018 / 2022 and runs exact paired significance tests vs the Elo baseline
- Simulates the full WC 2026 bracket (Monte Carlo) for champion and advancement probabilities
- Layers an exact-scoreline model on top of the W/D/L probabilities
- Explains predictions with plain-language SHAP groupings
- Grades pre-tournament predictions against live WC 2026 results, leakage-free
- Serves everything via an interactive Streamlit app

## App pages

1. **Match Predictor**: predict any fixture at a neutral venue or with home advantage for the home team, compare two teams, and see a SHAP explanation of the drivers
2. **Tournament Simulator**: Monte Carlo champion probabilities, group qualification rates, and an advancement-probability table
3. **Team Rankings**: Elo ratings for every rated national team
4. **Prediction Tracker**: WC 2026 results (all 104 matches) against the model's pre-tournament predictions, fetched from football-data.org

The deployed app serves the frozen model bundle and serving snapshot tracked in this repository (`models/`, `data/processed/`). It checks their SHA-256 against [`models/model-manifest.json`](models/model-manifest.json) at start-up and refuses to serve predictions if they differ; it never retrains.

## Results

Leakage-free backtest over the three most recent World Cups (192 matches; result *after extra time, excluding penalty shootouts*):

| System | Accuracy | Log-loss | Brier |
|--------|----------|----------|-------|
| Raw XGBoost | 53.65% | 0.9862 | 0.5841 |
| Blend (0.75 XGB / 0.25 Elo) | 53.13% | 0.9818 | 0.5816 |
| Elo baseline | 56.25% | 0.9859 | 0.5836 |

Elo is ahead on accuracy; the blend has the lowest log-loss and Brier. Neither difference is **statistically significant**: McNemar exact `p = 0.21` (5 blend-only-correct against 11 Elo-only-correct of 16 discordant matches), and the paired bootstrap 95% CI for the blend minus Elo log-loss difference includes zero (mean -0.004, CI [-0.025, +0.018], 10,000 resamples).

The blend's probabilistic edge is **uneven across tournaments**:

| Tournament | Blend acc | Elo acc | Blend log-loss | Elo log-loss |
|------------|-----------|---------|----------------|--------------|
| WC 2014 | 56.25% | 59.38% | 0.9559 | 0.9651 |
| WC 2018 | 53.13% | 56.25% | 0.9575 | 0.9510 |
| WC 2022 | 50.00% | 53.13% | 1.0320 | 1.0416 |

The blend has the lower log-loss in 2014 and 2022 and Elo in 2018; Elo is more accurate in all three. These numbers come from the remediated evaluation protocol (one frozen pre-tournament snapshot per World Cup, venue-aware inference); the pre-remediation figures are kept in [`reports/remediation_baseline.md`](reports/remediation_baseline.md). Full artifacts live in [`reports/`](reports/) (`backtest_metrics_summary.csv`, `backtest_by_tournament.csv`, `backtest_predictions.csv`) with figures in [`reports/figures/`](reports/figures/).

### WC 2026: final live evaluation

All 104 matches, scored on the result after extra time with penalty shootouts counted as draws, by the frozen pre-tournament model ([`reports/live_evaluation.csv`](reports/live_evaluation.csv)):

| Scope | Blend correct | Elo correct | Blend log-loss | Elo log-loss |
|-------|---------------|-------------|----------------|--------------|
| All 104 matches | 68/104 | 70/104 | 0.8334 | 0.8666 |
| 4 logged before kickoff | 3/4 | 3/4 | 0.8922 | 0.8695 |
| 100 recomputed after | 65/100 | 67/100 | 0.8310 | 0.8665 |

Elo picked two more winners (McNemar exact `p = 0.50`: the only two discordant matches both went to Elo). The blend's probabilities scored better: blend minus Elo log-loss -0.033, paired bootstrap 95% CI [-0.059, -0.005] over all 104 matches. The four semifinal, third-place and final predictions were written to the append-only [`reports/prediction_ledger.csv`](reports/prediction_ledger.csv) before kickoff; the other 100 were recomputed afterwards from the same frozen model and are labelled as such. Four matches are too few for a paired test.

## Project structure

```
wc2026_predictor/
├── app/
│   └── streamlit_app.py        # Interactive Streamlit dashboard (4 pages)
├── src/
│   ├── fetch_data.py           # Download raw international results (martj42)
│   ├── features.py             # Feature engineering pipeline
│   ├── train.py                # XGBoost training + Elo blend & prior
│   ├── simulate.py             # WC 2026 Monte Carlo simulation; PRE_TOURNAMENT_CUTOFF
│   ├── backtest.py             # Leakage-free WC 2014/18/22 backtest
│   ├── paired_tests.py         # Exact paired Blend-vs-Elo tests (McNemar + bootstrap)
│   ├── export_backtest.py      # Row-level export of the backtest
│   ├── make_figures.py         # Article figures 3, 4, 6, 10 (light and dark pairs)
│   ├── make_by_tournament.py   # Per-tournament breakdown and figure 5
│   ├── live_evaluation.py      # Final WC 2026 evaluation (prospective + retrospective)
│   ├── prediction_ledger.py    # Append-only pre-kickoff prediction ledger
│   ├── experiments.py          # Experiments to beat the Elo baseline
│   ├── explain.py              # Plain-language SHAP grouping for Match Predictor
│   ├── scorelines.py           # Exact-scoreline probability layer
│   ├── fetch_results.py        # Auto-fetch live WC results (capped at cutoff)
│   ├── team_names.py           # Team-name alias → canonical mapping
│   ├── third_place_mapping.py  # FIFA Annexe C third-place R32 assignment
│   └── smoke_test.py           # Pre-push smoke test
├── tests/                      # pytest suite (serving state, team names, third place)
├── reports/                    # Backtest CSVs, DATA_SNAPSHOT.md, figures/
├── notebooks/
│   └── 04_model_improvement.md # Experiment log
├── data/
│   ├── raw/                    # Downloaded CSVs (gitignored)
│   ├── processed/              # Frozen serving snapshot (tracked, hashed in the manifest)
│   └── wc2026_results.json     # Cached live-results feed
├── scripts/
│   ├── pre-push                # Git hook: runs smoke test before push
│   ├── pre-commit              # Git hook
│   └── sync_version.py         # Keeps VERSION in sync
├── .streamlit/
│   └── config.toml             # (secrets.toml is gitignored)
├── pyproject.toml / uv.lock / requirements.txt
├── CITATION.cff / LICENSE / VERSION / .python-version
└── README.md
```

## Quick start

### 1. Install dependencies

This project uses [uv](https://docs.astral.sh/uv/). One command creates the virtual environment and installs the locked dependencies:

```bash
uv sync
```

Prefix subsequent commands with `uv run` (no manual activation needed), or activate once: `.venv\Scripts\activate` (Windows PowerShell) / `source .venv/bin/activate`.

> **Cross-volume note:** if uv's cache and the repo live on different drives (e.g. cache on `C:`, project on `D:`), uv prints a harmless `Failed to hardlink files; falling back to full copy` warning. To silence it, set `UV_LINK_MODE=copy` for the session — PowerShell: `$env:UV_LINK_MODE = "copy"`. Local-only guidance; don't commit it.

### 2. Fetch raw data

```bash
uv run python src/fetch_data.py
```

Downloads to `data/raw/`:
- `results.csv` — all international match results since 1872
- `goalscorers.csv` — individual goalscorer records
- `shootouts.csv` — penalty shootout outcomes

### 3. Build features & train the model

```bash
uv run python src/features.py
uv run python src/train.py
```

The served bundle is already in the repository. On the frozen raw snapshot recorded in [`reports/DATA_SNAPSHOT.md`](reports/DATA_SNAPSHOT.md), these two steps rebuild it byte for byte. A newer upstream download gives a different model, so do not commit a retrained bundle over the frozen one.

### 4. (Optional) Reproduce the backtest and figures

```bash
uv run python src/backtest.py
uv run python src/paired_tests.py
uv run python src/export_backtest.py
uv run python src/make_figures.py
uv run python src/make_by_tournament.py
```

### 5. Run the Streamlit app

```bash
uv run streamlit run app/streamlit_app.py
```

### 6. (For contributors) Enable the git hooks

One-time setup — runs `src/smoke_test.py` automatically before every push:

```bash
git config core.hooksPath scripts
```

Bypass with `git push --no-verify` when needed. Run the smoke test manually any time with `uv run python src/smoke_test.py`.

## Live result tracker

The Prediction Tracker grades the model's pre-tournament predictions against live WC 2026 results from [football-data.org](https://www.football-data.org). It is scored on the **same target the model was trained on**: the result *after extra time, excluding penalty shootouts* (the martj42 convention).

football-data.org's v4 `score.fullTime` already **includes** extra-time goals, so extra-time knockouts score directly (verified: England 2–1 Slovakia = regular-time 1–1 + extra-time 1–0). For **penalty-shootout** knockouts, `fullTime` folds in the shootout tally (verified: Portugal 0–0 Slovenia after extra time, won 3–0 on penalties, reported by the API as `fullTime` 3–0). The tracker strips the shootout and records such a match as the **after-extra-time draw** it was on the pitch, matching the training target — the model is never credited or penalised for predicting a shootout winner. Group-stage games are always 90 minutes, so this only affects knockout fixtures.

The model's team state is frozen strictly before the tournament (`simulate.PRE_TOURNAMENT_CUTOFF`, `2026-06-11`) in code, so predictions stay leakage-free even if the data is rebuilt mid-tournament from the live feed.

## Dataset

[International football results (1872–present)](https://github.com/martj42/international_results) — Mart Jürisoo, GitHub. See [`CITATION.cff`](CITATION.cff) for citation details and [`reports/DATA_SNAPSHOT.md`](reports/DATA_SNAPSHOT.md) for the snapshot used.

## Tech stack

| Layer | Library |
|-------|---------|
| Data | pandas, numpy, requests |
| Modelling | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualisation | Plotly, Matplotlib, Seaborn |
| App | Streamlit |
| Stats | SciPy |

## Tournament details

- **Edition:** 23rd FIFA World Cup
- **Dates:** June 11 – July 19, 2026
- **Hosts:** USA, Canada, Mexico
- **Teams:** 48 (expanded from 32)
- **Format:** 12 groups of 4; top 2 from each group plus the 8 best third-placed teams advance to a 32-team Round of 32

## License

See [`LICENSE`](LICENSE).
