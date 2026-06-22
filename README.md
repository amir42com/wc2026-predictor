# FIFA World Cup 2026 Prediction App

Machine-learning-powered match outcome predictor for the 2026 FIFA World Cup (USA / Canada / Mexico), built in Python with a Streamlit dashboard.

## What it does

- Downloads and processes historical international football results (1872–present) from GitHub (martj42/international_results)
- Engineers features: Elo ratings, recent form, head-to-head records, home/neutral venue, confederation strength
- Trains an XGBoost classifier to predict win / draw / loss probabilities for any fixture
- Simulates the full tournament bracket and surfaces expected winners
- Explains predictions with SHAP force plots
- Serves everything via an interactive Streamlit app

## Project structure

```
wc2026_predictor/
├── data/
│   ├── raw/            # Downloaded CSVs (gitignored)
│   └── processed/      # Engineered feature tables (gitignored)
├── notebooks/          # EDA and experiment notebooks
├── src/
│   ├── fetch_data.py   # Download raw data from GitHub
│   ├── features.py     # Feature engineering pipeline
│   ├── train.py        # Model training & evaluation
│   ├── simulate.py     # Tournament bracket simulation
│   └── utils.py        # Shared helpers
├── app/
│   └── streamlit_app.py  # Interactive dashboard
├── requirements.txt
└── README.md
```

## Quick start

### 1. Install dependencies

This project uses [uv](https://docs.astral.sh/uv/). One command creates the
virtual environment and installs the locked dependencies:

```bash
uv sync
```

Prefix subsequent commands with `uv run` (no manual activation needed), or
activate `.venv\Scripts\activate` (Windows) / `source .venv/bin/activate` once.

> **Cross-volume note:** if uv's cache and the repo live on different drives
> (e.g. cache on `C:`, project on `D:`), uv prints a harmless
> `Failed to hardlink files; falling back to full copy` warning. To silence it,
> set `UV_LINK_MODE=copy` in your shell for the session, e.g. PowerShell
> `$env:UV_LINK_MODE = "copy"`. This is local-only guidance — don't commit it.

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

### 4. Run the Streamlit app

```bash
uv run streamlit run app/streamlit_app.py
```

### 5. (For contributors) Enable the pre-push smoke test

One-time setup — runs `src/smoke_test.py` automatically before every push:

```bash
git config core.hooksPath scripts
```

Bypass with `git push --no-verify` when needed.

## Dataset

[International football results (1872–present)](https://github.com/martj42/international_results) — Mart Jürisoo, GitHub.

## Tech stack

| Layer | Library |
|-------|---------|
| Data | pandas, numpy, requests |
| Modelling | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualisation | Plotly, Matplotlib, Seaborn |
| App | Streamlit |
| Stats | SciPy |

## Live result tracker

The Prediction Tracker grades the model's pre-tournament predictions against live
WC 2026 results from [football-data.org](https://www.football-data.org). It is
scored on the **same target the model was trained on**: the result *after extra
time, excluding penalty shootouts* (the martj42 convention).

football-data.org's v4 `score.fullTime` already **includes** extra-time goals,
so extra-time knockouts score directly (verified: England 2-1 Slovakia =
regular-time 1-1 + extra-time 1-0). For **penalty-shootout** knockouts,
`fullTime` folds in the shootout tally (verified: Portugal 0-0 Slovenia after
extra time, won 3-0 on penalties, reported by the API as `fullTime` 3-0). The
tracker strips the shootout and records such a match as the **after-extra-time
draw** it was on the pitch, matching the training target — the model is never
credited or penalised for predicting a shootout winner. Group-stage games are
always 90 minutes, so this only affects knockout fixtures.

The model's team state is frozen strictly before the tournament
(`simulate.PRE_TOURNAMENT_CUTOFF`, 2026-06-11) in code, so predictions stay
leakage-free even if the data is rebuilt mid-tournament from the live feed.

## Tournament details

- **Edition:** 23rd FIFA World Cup
- **Dates:** June 11 – July 19, 2026
- **Hosts:** USA, Canada, Mexico
- **Teams:** 48 (expanded from 32)
- **Format:** 12 groups of 4, top 2 + 8 best third-place advance (32 teams in round of 32)
