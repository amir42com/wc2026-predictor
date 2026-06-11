# FIFA World Cup 2026 Prediction App

Machine-learning-powered match outcome predictor for the 2026 FIFA World Cup (USA / Canada / Mexico), built in Python with a Streamlit dashboard.

## What it does

- Downloads and processes historical international football results (1872–present) from Kaggle
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
│   ├── fetch_data.py   # Download raw data from Kaggle
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

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Set up Kaggle credentials

1. Go to https://www.kaggle.com/settings → **API** → **Create New Token**
2. Place the downloaded `kaggle.json` in `%USERPROFILE%\.kaggle\` (Windows) or `~/.kaggle/` (Linux/Mac)
3. The file must contain `{"username":"...","key":"..."}`

### 3. Fetch raw data

```bash
python src/fetch_data.py
```

Downloads to `data/raw/`:
- `results.csv` — all international match results since 1872
- `goalscorers.csv` — individual goalscorer records
- `shootouts.csv` — penalty shootout outcomes

### 4. Build features & train the model

```bash
python src/features.py
python src/train.py
```

### 5. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## Dataset

[International football results from 1872 to 2017 (and beyond)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) — Mart Jürisoo, Kaggle.

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
- **Format:** 12 groups of 4, top 2 + 8 best third-place advance (32 teams in round of 32)
