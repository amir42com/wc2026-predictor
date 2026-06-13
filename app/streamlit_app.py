"""
WC 2026 Prediction Dashboard
Run: streamlit run app/streamlit_app.py
"""

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulate import GROUPS, Predictor, monte_carlo, simulate_group, resolve_r32
from train import ELO_BLEND_W, elo_prior_proba, make_X
import fetch_results

RESULTS_PATH = Path(__file__).resolve().parents[1] / "data" / "wc2026_results.json"

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR    = Path(__file__).resolve().parents[1] / "models"
RAW_DIR       = Path(__file__).resolve().parents[1] / "data" / "raw"

WC_TEAMS = sorted({t for teams in GROUPS.values() for t in teams})

CONF_COLOR = {
    "UEFA":     "#3498db",
    "CONMEBOL": "#2ecc71",
    "CONCACAF": "#e67e22",
    "CAF":      "#e74c3c",
    "AFC":      "#9b59b6",
    "OFC":      "#1abc9c",
    "Other":    "#95a5a6",
}

# ISO 3166-1 alpha-2 codes for flagcdn.com — flag images render everywhere,
# unlike emoji (which fall back to letter pairs on Windows)
ISO2: dict[str, str] = {
    # CONMEBOL
    "Argentina": "ar", "Brazil": "br", "Colombia": "co", "Ecuador": "ec",
    "Uruguay": "uy", "Paraguay": "py", "Chile": "cl", "Venezuela": "ve",
    "Peru": "pe", "Bolivia": "bo",
    # UEFA
    "France": "fr", "Spain": "es", "Germany": "de", "Portugal": "pt",
    "England": "gb-eng", "Netherlands": "nl", "Belgium": "be", "Italy": "it",
    "Croatia": "hr", "Switzerland": "ch", "Denmark": "dk", "Sweden": "se",
    "Norway": "no", "Austria": "at", "Poland": "pl", "Serbia": "rs",
    "Scotland": "gb-sct", "Turkey": "tr", "Czech Republic": "cz",
    "Slovakia": "sk", "Hungary": "hu", "Romania": "ro", "Ukraine": "ua",
    "Greece": "gr", "Bosnia and Herzegovina": "ba", "Albania": "al",
    "Georgia": "ge", "Slovenia": "si",
    # CONCACAF
    "United States": "us", "Mexico": "mx", "Canada": "ca", "Panama": "pa",
    "Costa Rica": "cr", "Jamaica": "jm", "Haiti": "ht", "Honduras": "hn",
    "Guatemala": "gt", "El Salvador": "sv", "Trinidad and Tobago": "tt",
    "Curaçao": "cw",
    # CAF
    "Morocco": "ma", "Senegal": "sn", "Egypt": "eg", "Nigeria": "ng",
    "Ghana": "gh", "Ivory Coast": "ci", "Cameroon": "cm", "Mali": "ml",
    "Algeria": "dz", "Tunisia": "tn", "South Africa": "za", "DR Congo": "cd",
    "Cape Verde": "cv",
    # AFC
    "Japan": "jp", "South Korea": "kr", "Australia": "au", "Iran": "ir",
    "Saudi Arabia": "sa", "Qatar": "qa", "Iraq": "iq", "Jordan": "jo",
    "Uzbekistan": "uz", "China": "cn", "India": "in", "UAE": "ae",
    # OFC
    "New Zealand": "nz",
}


def flag_img(team: str) -> str:
    """16x12 flag <img> for HTML-rendered contexts; empty string if unknown."""
    iso = ISO2.get(team)
    if not iso:
        return ""
    return (f'<img src="https://flagcdn.com/16x12/{iso}.png" width="16" height="12" '
            f'alt="" loading="lazy" style="vertical-align:baseline">')


# Compact display names so narrow table columns don't wrap
SHORT_NAMES: dict[str, str] = {
    "Czech Republic":         "Czechia",
    "Bosnia and Herzegovina": "Bosnia & Herz.",
    "United States":          "USA",
    "South Korea":            "S. Korea",
    "South Africa":           "S. Africa",
}


def short_name(team: str) -> str:
    return SHORT_NAMES.get(team, team)


# Human-readable names for every model feature (used in the SHAP chart)
CONF_FULL = {
    "UEFA": "Europe", "CONMEBOL": "South America",
    "CONCACAF": "North/Central America", "CAF": "Africa",
    "AFC": "Asia", "OFC": "Oceania", "Other": "other region",
}
FEATURE_LABELS: dict[str, str] = {
    "home_elo":          "Home team strength (Elo)",
    "away_elo":          "Away team strength (Elo)",
    "elo_diff":          "Team strength gap (Elo)",
    "home_win_rate_5":   "Home: win rate, last 5 games",
    "away_win_rate_5":   "Away: win rate, last 5 games",
    "home_gd_5":         "Home: goal difference, last 5 games",
    "away_gd_5":         "Away: goal difference, last 5 games",
    "home_win_rate_10":  "Home: win rate, last 10 games",
    "away_win_rate_10":  "Away: win rate, last 10 games",
    "home_gd_10":        "Home: goal difference, last 10 games",
    "away_gd_10":        "Away: goal difference, last 10 games",
    "h2h_n":             "Head-to-head: games played",
    "h2h_home_wr":       "Head-to-head: home win rate",
    "home_conf_elo":     "Home confederation strength",
    "away_conf_elo":     "Away confederation strength",
    "neutral":           "Neutral venue",
    "is_world_cup":      "World Cup match",
    **{f"h_conf_{c}": f"Home team from {c} ({full})" for c, full in CONF_FULL.items()},
    **{f"a_conf_{c}": f"Away team from {c} ({full})" for c, full in CONF_FULL.items()},
}

# ── cached resources ───────────────────────────────────────────────────────

@st.cache_resource
def load_resources():
    bundle    = joblib.load(MODELS_DIR / "xgb_wc2026.joblib")
    predictor = Predictor(
        PROCESSED_DIR / "features.csv",
        PROCESSED_DIR / "elo_ratings.csv",
        bundle,
    )
    explainer = shap.TreeExplainer(bundle["model"])
    elo_df    = pd.read_csv(PROCESSED_DIR / "elo_ratings.csv")
    return bundle, predictor, explainer, elo_df


# ── first-run bootstrap ───────────────────────────────────────────────────

def _bootstrap_if_needed() -> None:
    """
    On Streamlit Cloud (or any cold start) the processed data and model
    files won't exist.  This runs the full pipeline once, showing progress
    in the UI, then calls st.rerun() so the app loads normally.
    """
    missing_raw      = not (RAW_DIR / "results.csv").exists()
    missing_features = not (PROCESSED_DIR / "features.csv").exists()
    missing_model    = not (MODELS_DIR / "xgb_wc2026.joblib").exists()

    if not (missing_raw or missing_features or missing_model):
        return  # everything already present

    import fetch_data as _fd
    import features  as _feat
    import train     as _tr

    st.title("⚽ WC 2026 Predictor")
    with st.status("First-run setup — this takes ~2 minutes…", expanded=True) as _s:

        if missing_raw:
            st.write("Downloading match data from GitHub (1872–2026)…")
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            _fd.main()
            st.write("Data downloaded.")

        if missing_features:
            st.write("Engineering features (Elo, form, H2H, confederation)…")
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            _df = (
                pd.read_csv(RAW_DIR / "results.csv", parse_dates=["date"])
                .dropna(subset=["home_score", "away_score"])
                .sort_values("date")
                .reset_index(drop=True)
            )
            feat_df, elo_df = _feat.build_features(_df)
            feat_df.to_csv(PROCESSED_DIR / "features.csv",    index=False)
            elo_df.to_csv( PROCESSED_DIR / "elo_ratings.csv", index=False)
            st.write(f"Features built: {len(feat_df):,} matches.")

        if missing_model:
            st.write("Training XGBoost model (~30 seconds)…")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            _df2 = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
            mask  = _df2["date"].dt.year < _tr.TEST_YEAR
            model, feature_cols = _tr.train_model(_df2[mask].reset_index(drop=True))
            _bundle = {
                "model":        model,
                "feature_cols": feature_cols,
                "label_map":    _tr.LABEL_MAP,
            }
            joblib.dump(_bundle, MODELS_DIR / "xgb_wc2026.joblib")
            st.write("Model trained and saved.")

        _s.update(label="Setup complete — loading app…", state="complete")

    st.rerun()


# ── page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WC 2026 Predictor · Amir Mohammadi",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/amir42com/wc2026-predictor",
        "Report a bug": "https://github.com/amir42com/wc2026-predictor/issues",
        "About": (
            "**WC 2026 Predictor** — XGBoost + Elo-prior blend trained on "
            "49,000+ international matches (1872–2026). "
            "Match predictions, Monte Carlo tournament simulation, and "
            "SHAP explainability."
        ),
    },
)

_bootstrap_if_needed()

bundle, predictor, explainer, elo_df = load_resources()
ALL_TEAMS = sorted(predictor._state.keys())

# ── design system: "Midnight Pitch" ─────────────────────────────────────────
# Deep navy surfaces + electric sky-blue accent (#4fc3f7), Sora for headings.
# IMPORTANT: Noto Color Emoji must come LAST in every font stack so it only
# activates for emoji codepoints — listing it first breaks digit/letter spacing.
_BODY_FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, "
              "sans-serif, 'Noto Color Emoji'")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Noto+Color+Emoji&display=swap');

/* ── typography ── */
h1, h2, h3 {
    font-family: 'Sora', -apple-system, sans-serif !important;
    letter-spacing: 0.01em;
}
[data-testid="stMetricLabel"],
[data-testid="stSelectbox"] * {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Arial, sans-serif, "Noto Color Emoji" !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Sora', sans-serif, 'Noto Color Emoji' !important;
    font-weight: 700;
}

/* ── metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(160deg, #151d3b 0%, #10162e 100%);
    border: 1px solid #26305a;
    border-radius: 14px;
    padding: 1rem 1.25rem;
    transition: border-color .2s ease, transform .2s ease, box-shadow .2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(79, 195, 247, 0.55);
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(79, 195, 247, 0.08);
}
[data-testid="stMetricLabel"] { color: #93a1c8; }

/* ── probability cards (HTML so flag images render) ── */
.prob-card {
    background: linear-gradient(160deg, #151d3b 0%, #10162e 100%);
    border: 1px solid #26305a;
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
    transition: border-color .2s ease, transform .2s ease, box-shadow .2s ease;
}
.prob-card:hover {
    border-color: rgba(79, 195, 247, 0.55);
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(79, 195, 247, 0.08);
}
.prob-card .prob-label { color: #93a1c8; font-size: 0.9rem; margin-bottom: 0.2rem; }
.prob-card .prob-value {
    font-family: 'Sora', sans-serif;
    font-size: 2rem; font-weight: 700; color: #e8ecf6;
}
.prob-card img { vertical-align: -1px; margin-right: 5px; }

/* ── HTML stacked probability bar ── */
.prob-bar {
    display: flex; height: 46px;
    border-radius: 10px; overflow: hidden;
    margin: 0.4rem 0 1rem 0;
    font-size: 0.9rem; font-weight: 600; color: #fff;
}
.prob-bar .seg {
    display: flex; align-items: center; justify-content: center;
    gap: 6px; white-space: nowrap; overflow: hidden;
}

/* ── HTML tables (flag images need unescaped HTML) ── */
.html-table table {
    width: 100%; border-collapse: collapse;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Arial, sans-serif, "Noto Color Emoji";
    font-size: 0.92rem;
    border: 1px solid #26305a; border-radius: 12px;
}
.html-table th {
    color: #93a1c8; font-weight: 600; text-align: left;
    background: #10162e;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #26305a;
}
.html-table td { padding: 0.45rem 0.75rem; border-bottom: 1px solid #1d2547; }
.html-table tbody tr:hover td { background: rgba(79,195,247,0.05); }

/* fixed equal columns so the 12 group tables align */
.group-table table { table-layout: fixed; }
.group-table th:nth-child(1) { width: 48%; }
.group-table th:nth-child(2) { width: 26%; }
.group-table th:nth-child(3) { width: 26%; }
.group-table th, .group-table td {
    white-space: nowrap;
    font-size: 0.85rem;
    padding: 0.4rem 0.5rem;
    overflow: hidden;
}

/* group blocks in the Elo snapshot table */
.html-table tr.grp-even td { background: rgba(79, 195, 247, 0.045); }
.html-table tr.grp-start td { border-top: 2px solid #2a3560; }

/* ── top navigation pills ── */
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] {
    display: flex; gap: 0.5rem; flex-wrap: wrap;
}
[data-testid="stMain"] [data-testid="stRadio"] label {
    background: #151d3b;
    border: 1px solid #26305a;
    border-radius: 999px;
    padding: 0.4rem 1.1rem;
    cursor: pointer;
    transition: border-color .15s ease, background .15s ease;
}
[data-testid="stMain"] [data-testid="stRadio"] label:hover { border-color: #4fc3f7; }
[data-testid="stMain"] [data-testid="stRadio"] label > div:first-of-type { display: none; }
[data-testid="stMain"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(90deg, #2196f3, #4fc3f7);
    border-color: transparent;
}
[data-testid="stMain"] [data-testid="stRadio"] label:has(input:checked) p {
    color: #fff; font-weight: 600;
}

/* ── tables ── */
[data-testid="stTable"] {
    border: 1px solid #26305a;
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stTable"] table {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Arial, sans-serif, "Noto Color Emoji" !important;
    font-size: 0.92rem;
}
[data-testid="stTable"] th {
    color: #93a1c8 !important;
    font-weight: 600;
    background: #10162e;
    border-bottom: 1px solid #26305a !important;
}
[data-testid="stTable"] td { border-color: #1d2547 !important; }
[data-testid="stTable"] tbody tr:hover td { background: rgba(79,195,247,0.05); }

/* hide element toolbars (fullscreen/download/search) — they break mobile touch */
[data-testid="stElementToolbar"] { display: none !important; }

/* ── sidebar credits ── */
.credits { font-size: 0.85rem; line-height: 1.6; }
.credits p { margin: 0 0 0.3rem 0; }
.credits-name { color: #8a93ad; font-weight: 600; }
.credits-links a {
    color: #8a93ad !important;
    text-decoration: none;
    transition: color .15s ease;
    white-space: nowrap;
}
.credits-links a:hover { color: #4fc3f7 !important; }
.credits-links svg {
    width: 16px; height: 16px;
    vertical-align: -3px;
    margin-right: 3px;
}
.credits-version { color: #5b6379; font-size: 0.78rem; }

/* ── buttons ── */
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, #2196f3, #4fc3f7);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: filter .15s ease, transform .15s ease;
}
[data-testid="stBaseButton-primary"]:hover {
    filter: brightness(1.12);
    transform: translateY(-1px);
}
[data-testid="stBaseButton-secondary"] {
    border: 1px solid #26305a;
    border-radius: 10px;
    transition: border-color .15s ease;
}
[data-testid="stBaseButton-secondary"]:hover { border-color: #4fc3f7; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0e27;
    border-right: 1px solid #1d2547;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { font-size: 0.85rem; }

/* main-page captions: keep body text readable (never below 0.9rem) */
[data-testid="stMain"] [data-testid="stCaptionContainer"] p { font-size: 0.9rem; }

/* ── spacing & dividers ── */
hr { border-color: #1d2547 !important; margin: 1.6rem 0 !important; }
.block-container { padding-top: 2.2rem; }

/* ── hero banner ── */
.hero {
    position: relative; overflow: hidden; text-align: center;
    border-radius: 16px;
    padding: 2.6rem 2rem 2.2rem 2rem;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #0a0e27 0%, #0d1b3e 40%, #0a1628 70%, #000 100%);
    border: 1px solid rgba(79, 195, 247, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background:
        repeating-linear-gradient(90deg, transparent, transparent 49%, rgba(255,255,255,.03) 49%, rgba(255,255,255,.03) 51%),
        repeating-linear-gradient(0deg,  transparent, transparent 49%, rgba(255,255,255,.03) 49%, rgba(255,255,255,.03) 51%);
    background-size: 60px 60px;
}
.hero::after {
    content: ''; position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 220px; height: 220px; border-radius: 50%;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.hero > * { position: relative; z-index: 1; }
.hero .hero-icon { font-size: 2.6rem; line-height: 1; }
.hero h1 {
    margin: 0.35rem 0 0.25rem 0;
    font-size: clamp(1.6rem, 5vw, 2.5rem);
    font-weight: 800;
    letter-spacing: 0.05em;
    background: linear-gradient(90deg, #4fc3f7, #ffffff, #81d4fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero .hero-sub {
    margin: 0;
    font-size: clamp(0.8rem, 2.5vw, 1rem);
    color: rgba(232, 236, 246, 0.6);
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.hero .hero-tag {
    margin: 0.55rem 0 0 0;
    font-size: clamp(0.75rem, 2vw, 0.9rem);
    color: rgba(232, 236, 246, 0.38);
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


def hero(icon: str, title: str, subtitle: str, tagline: str) -> None:
    """Page hero banner — single source of truth for the header treatment."""
    st.markdown(f"""
<div class="hero">
  <div class="hero-icon">{icon}</div>
  <h1>{title}</h1>
  <p class="hero-sub">{subtitle}</p>
  <p class="hero-tag">{tagline}</p>
</div>
""", unsafe_allow_html=True)


# ── shared Plotly template: every chart inherits fonts/colors from here ─────
import plotly.io as pio

_tpl = go.layout.Template()
_tpl.layout = go.Layout(
    font=dict(family=_BODY_FONT, color="#cdd6ee", size=13),
    title_font=dict(family="'Sora', sans-serif", color="#e8ecf6", size=18),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="#1d2547", zerolinecolor="#2a3560",
               linecolor="#26305a", title_font=dict(color="#93a1c8")),
    yaxis=dict(gridcolor="#1d2547", zerolinecolor="#2a3560",
               linecolor="#26305a", title_font=dict(color="#93a1c8")),
    hoverlabel=dict(bgcolor="#151d3b", bordercolor="#4fc3f7",
                    font=dict(family=_BODY_FONT, color="#e8ecf6")),
    colorway=["#4fc3f7", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c"],
)
pio.templates["wc2026"] = _tpl
pio.templates.default = "wc2026"

# staticPlot: charts render as static images so they don't capture touch
# events on mobile (all values are labeled on the bars; hover isn't needed)
_PLOTLY_CONFIG = {"staticPlot": True}

# ── sidebar navigation ─────────────────────────────────────────────────────

st.sidebar.title("⚽ WC 2026 Predictor")

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About the model"):
    st.markdown(
        "Predictions come from a **machine-learning model** (XGBoost) that "
        "learned from **49,000+ international matches played since 1872**.\n\n"
        "For each fixture it weighs team strength, recent form, and "
        "head-to-head history to estimate win / draw / loss chances.\n\n"
        "Those chances are blended with **Elo ratings** — a chess-style "
        "strength score updated after every match — which keeps the "
        "probabilities realistic for tournament play.\n\n"
        "[Technical details on GitHub](https://github.com/amir42com/wc2026-predictor)"
    )

# ── credits ──────────────────────────────────────────────────────────────
_ICON_GLOBE = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
    '<line x1="2" y1="12" x2="22" y2="12"/>'
    '<path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 '
    '15.3 15.3 0 0 1 4-10z"/></svg>'
)
_ICON_LINKEDIN = (
    '<svg viewBox="0 0 24 24" fill="currentColor">'
    '<path d="M20.45 20.45h-3.55v-5.57c0-1.33-.03-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 '
    '2.94v5.67H9.36V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 '
    '5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12zM7.12 '
    '20.45H3.56V9h3.56v11.45zM22.22 0H1.77C.79 0 0 .77 0 1.72v20.55C0 23.22.79 24 '
    '1.77 24h20.45c.98 0 1.78-.78 1.78-1.73V1.72C24 .77 23.2 0 22.22 0z"/></svg>'
)
_ICON_GITHUB = (
    '<svg viewBox="0 0 16 16" fill="currentColor">'
    '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
    '0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-'
    '.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-'
    '1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 '
    '.67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 '
    '1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 '
    '1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-'
    '8-8z"/></svg>'
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div class="credits">
  <p class="credits-name">Built by Amir Mohammadi</p>
  <p class="credits-links">
    <a href="https://amir42.com" target="_blank">{_ICON_GLOBE} Portfolio</a> ·
    <a href="https://www.linkedin.com/in/amir42com/" target="_blank">{_ICON_LINKEDIN} LinkedIn</a> ·
    <a href="https://github.com/amir42com/wc2026-predictor" target="_blank">{_ICON_GITHUB} GitHub</a>
  </p>
  <p class="credits-version">v1.5 · June 2026</p>
</div>
""", unsafe_allow_html=True)


# ── top navigation pills (always visible — sidebar collapses on mobile) ────

page = st.radio(
    "Navigate",
    ["Match Predictor", "Tournament Simulator", "Team Rankings", "📊 Prediction Tracker"],
    horizontal=True,
    label_visibility="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════

if page == "Match Predictor":
    hero("⚽", "FIFA WORLD CUP 2026",
         "United States &nbsp;·&nbsp; Canada &nbsp;·&nbsp; Mexico",
         "June 11 – July 19, 2026")

    # ── team selectors ─────────────────────────────────────────────────────
    show_all = st.session_state.get("show_all_teams", False)
    team_pool = ALL_TEAMS if show_all else WC_TEAMS

    c1, cmid, c2 = st.columns([5, 1, 5])
    with c1:
        home_default = team_pool.index("Argentina") if "Argentina" in team_pool else 0
        home_team = st.selectbox("Home team", team_pool, index=home_default)
    with cmid:
        st.markdown("<div style='text-align:center;padding-top:2rem;font-size:1.5rem'>vs</div>",
                    unsafe_allow_html=True)
    with c2:
        away_default = team_pool.index("France") if "France" in team_pool else 1
        away_team = st.selectbox("Away team", team_pool, index=away_default)

    st.checkbox(f"Show all {len(ALL_TEAMS)} teams", key="show_all_teams",
                help="Off: only the 48 WC 2026 participants")
    neutral = st.checkbox("Neutral venue (World Cup)", value=True)

    if home_team == away_team:
        st.warning("Select two different teams.")
        st.stop()

    # ── feature row ────────────────────────────────────────────────────────
    hs  = predictor._state.get(home_team, predictor._default_state())
    as_ = predictor._state.get(away_team, predictor._default_state())
    n_h2h, hwr = predictor._h2h_stats(home_team, away_team)

    feat_row = {
        "home_elo":           hs["elo"],
        "away_elo":           as_["elo"],
        "elo_diff":           hs["elo"] - as_["elo"],
        "home_win_rate_5":    hs["win_rate_5"],
        "away_win_rate_5":    as_["win_rate_5"],
        "home_gd_5":          hs["gd_5"],
        "away_gd_5":          as_["gd_5"],
        "home_win_rate_10":   hs["win_rate_10"],
        "away_win_rate_10":   as_["win_rate_10"],
        "home_gd_10":         hs["gd_10"],
        "away_gd_10":         as_["gd_10"],
        "h2h_n":              n_h2h,
        "h2h_home_wr":        hwr,
        "home_conf_elo":      hs["conf_elo"],
        "away_conf_elo":      as_["conf_elo"],
        "neutral":            int(neutral),
        "is_world_cup":       int(neutral),
        "home_confederation": hs["confederation"],
        "away_confederation": as_["confederation"],
    }
    X, _ = make_X(pd.DataFrame([feat_row]), bundle["feature_cols"])
    proba = bundle["model"].predict_proba(X)[0]   # [p_home_win, p_draw, p_away_win]
    # Production blend: shrink toward the Elo-logistic prior (same as simulator)
    proba = ELO_BLEND_W * proba + (1 - ELO_BLEND_W) * elo_prior_proba(hs["elo"], as_["elo"])
    predicted_class = int(proba.argmax())
    outcome_labels  = [f"{home_team} Win", "Draw", f"{away_team} Win"]

    # ── probability display ────────────────────────────────────────────────
    # Identity colors (home blue / draw gray / away amber), not good/bad
    # semantics. HTML cards + bar so flag images render on every platform.
    OUTCOME_COLORS = ["#3b82f6", "#6b7280", "#f59e0b"]

    st.markdown("---")
    card_labels = [
        f"{flag_img(home_team)} {home_team} Win",
        "Draw",
        f"{flag_img(away_team)} {away_team} Win",
    ]
    mcols = st.columns(3)
    for col, label, p, color in zip(mcols, card_labels, proba, OUTCOME_COLORS):
        col.markdown(
            f'<div class="prob-card" style="border-left:4px solid {color}">'
            f'<div class="prob-label">{label}</div>'
            f'<div class="prob-value">{p*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )

    # Stacked probability bar — pure HTML so flags render and touch scrolling works
    seg_texts = [
        f"{flag_img(home_team)} {proba[0]*100:.1f}%",
        f"{proba[1]*100:.1f}%",
        f"{flag_img(away_team)} {proba[2]*100:.1f}%",
    ]
    segs = "".join(
        f'<div class="seg" style="width:{p*100:.2f}%;background:{c}">{t}</div>'
        for p, c, t in zip(proba, OUTCOME_COLORS, seg_texts)
    )
    st.markdown(f'<div class="prob-bar">{segs}</div>', unsafe_allow_html=True)

    # ── team comparison ────────────────────────────────────────────────────
    st.subheader("Team comparison")
    comp_df = pd.DataFrame({
        "": ["Elo rating", "Form — win rate (last 5)",
             "Form — avg GD (last 5)", "Form — win rate (last 10)",
             "Form — avg GD (last 10)", "Confederation"],
        home_team: [
            f"{hs['elo']:.0f}",
            f"{hs['win_rate_5']*100:.0f}%",
            f"{hs['gd_5']:+.2f}",
            f"{hs['win_rate_10']*100:.0f}%",
            f"{hs['gd_10']:+.2f}",
            hs["confederation"],
        ],
        away_team: [
            f"{as_['elo']:.0f}",
            f"{as_['win_rate_5']*100:.0f}%",
            f"{as_['gd_5']:+.2f}",
            f"{as_['win_rate_10']*100:.0f}%",
            f"{as_['gd_10']:+.2f}",
            as_["confederation"],
        ],
    }).set_index("")
    st.table(comp_df)
    st.caption(
        "**Elo rating** = team strength score built from 150 years of results "
        "(top teams ≈ 2000+). **Form** = results in the team's most recent "
        "matches. **GD** = average goal difference (+2.0 means winning by "
        "2 goals on average)."
    )

    if n_h2h > 0:
        hw_n = round(hwr * n_h2h)
        d_n  = round(sum(
            1 for _, o in predictor._h2h.get(frozenset({home_team, away_team}), [])
            if o == 1
        ))
        aw_n = n_h2h - hw_n - d_n
        st.caption(
            f"Last {n_h2h} meetings:  "
            f"{home_team} {hw_n}W – {d_n}D – {aw_n}W {away_team}"
        )

    # ── SHAP explanation ───────────────────────────────────────────────────
    # Explain relative to the FAVOURED team's win class so positive bars
    # always mean "pushes toward the favourite".
    fav_is_home = proba[0] >= proba[2]
    fav_team  = home_team if fav_is_home else away_team
    opp_team  = away_team if fav_is_home else home_team
    fav_color = OUTCOME_COLORS[0] if fav_is_home else OUTCOME_COLORS[2]
    opp_color = OUTCOME_COLORS[2] if fav_is_home else OUTCOME_COLORS[0]
    fav_class = 0 if fav_is_home else 2

    st.subheader(f"Why is {fav_team} favoured?")

    sv = explainer.shap_values(X)  # shape (n_samples, n_features, n_classes)
    sv_series = pd.Series(sv[0, :, fav_class], index=bundle["feature_cols"])
    xrow = X.iloc[0]

    def _side(f: str) -> str:
        """Team a feature belongs to (home/away prefix), else the home team."""
        return away_team if f.startswith(("away_", "a_conf_")) else home_team

    def _value_label(f: str) -> str:
        """Axis label: human name + the actual value, with team names."""
        v = float(xrow[f])
        if f == "elo_diff":
            stronger = home_team if v > 0 else away_team
            return f"Team strength gap: {stronger} +{abs(v):.0f} Elo"
        if f in ("home_elo", "away_elo"):
            return f"{_side(f)} strength: {v:.0f} Elo"
        if "_win_rate_" in f:
            return f"{_side(f)} win rate, last {f.rsplit('_', 1)[1]}: {v*100:.0f}%"
        if "_gd_" in f:
            return f"{_side(f)} goal diff, last {f.rsplit('_', 1)[1]}: {v:+.1f}"
        if f == "h2h_n":
            return f"Head-to-head games played: {v:.0f}"
        if f == "h2h_home_wr":
            return f"Head-to-head: {home_team} wins {v*100:.0f}%"
        if f in ("home_conf_elo", "away_conf_elo"):
            return f"{_side(f)} confederation strength: {v:.0f}"
        if f == "neutral":
            return f"Neutral venue: {'yes' if v else 'no'}"
        if f == "is_world_cup":
            return f"World Cup match: {'yes' if v else 'no'}"
        if f.startswith(("h_conf_", "a_conf_")):
            conf = f.split("_", 2)[2]
            return f"{_side(f)}: {conf} team" + ("" if v else " (not)")
        return FEATURE_LABELS.get(f, f)

    def _fragment(f: str) -> str:
        """Short phrase for the auto-generated summary sentence."""
        v = float(xrow[f])
        if f == "elo_diff":
            stronger = home_team if v > 0 else away_team
            return f"{stronger}'s +{abs(v):.0f} Elo advantage"
        if f in ("home_elo", "away_elo"):
            return f"{_side(f)}'s overall strength ({v:.0f} Elo)"
        if "_win_rate_" in f:
            return (f"{_side(f)}'s recent form "
                    f"({v*100:.0f}% wins in the last {f.rsplit('_', 1)[1]})")
        if "_gd_" in f:
            return f"{_side(f)}'s recent goal difference ({v:+.1f} per game)"
        if f in ("h2h_n", "h2h_home_wr"):
            return "the head-to-head record"
        if f in ("home_conf_elo", "away_conf_elo"):
            owner = _side(f)
            favours_fav = sv_series[f] > 0
            adj = "strong" if (favours_fav == (owner == fav_team)) else "weaker"
            return f"{owner}'s {adj} confederation"
        if f == "neutral":
            return "the neutral venue"
        if f == "is_world_cup":
            return "it being a World Cup match"
        if f.startswith(("h_conf_", "a_conf_")):
            conf = f.split("_", 2)[2]
            return f"{_side(f)} being a {CONF_FULL.get(conf, conf)} side"
        return FEATURE_LABELS.get(f, f).lower()

    # Auto-generated summary from the top 3 contributors
    top3 = sv_series.abs().nlargest(3).index
    pos  = [f for f in top3 if sv_series[f] > 0]
    neg  = [f for f in top3 if sv_series[f] < 0]
    if pos:
        summary = (f"{fav_team} is favoured mainly because of "
                   + " and ".join(_fragment(f) for f in pos[:2]))
        if neg:
            summary += f"; {_fragment(neg[0])} narrows the gap"
    else:
        summary = f"{fav_team} is narrowly favoured despite {_fragment(neg[0])}"
    st.markdown(f"**{summary}.**")

    st.markdown(
        f"<span style='color:{fav_color}'>&#9632;</span> favours {fav_team} "
        f"&nbsp;·&nbsp; "
        f"<span style='color:{opp_color}'>&#9632;</span> favours {opp_team}",
        unsafe_allow_html=True,
    )

    # Top 8 factors + everything else aggregated into one bar
    top8 = sv_series.abs().nlargest(8).index
    rest_net = float(sv_series.drop(top8).sum())
    shown = sv_series[top8].sort_values()  # ascending → biggest at top of chart

    y_labels = ["All other factors (net)"] + [_value_label(f) for f in shown.index]
    x_vals   = [rest_net] + list(shown.values)
    colors   = [fav_color if v > 0 else opp_color for v in x_vals]

    fig_shap = go.Figure(go.Bar(
        x=x_vals,
        y=y_labels,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig_shap.update_layout(
        xaxis_title=f"← favours {opp_team}      favours {fav_team} →",
        height=400,
        margin=dict(l=10, r=20, t=20, b=40),
        yaxis=dict(automargin=True),
    )
    st.plotly_chart(fig_shap, use_container_width=True, config=_PLOTLY_CONFIG)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — TOURNAMENT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════

elif page == "Tournament Simulator":
    hero("🏆", "TOURNAMENT SIMULATOR",
         "Monte Carlo Simulation &nbsp;·&nbsp; 48 Teams &nbsp;·&nbsp; Full Bracket",
         "June 11 – July 19, 2026")

    st.markdown(
        "This page plays out the **entire World Cup thousands of times** — "
        "every group game, every knockout round — using the match predictor "
        "for each game. The more often a team lifts the trophy across those "
        "simulated tournaments, the higher its championship probability."
    )

    ctrl_l, ctrl_r = st.columns([3, 2])
    with ctrl_l:
        n_sims = st.slider("Number of simulations", 500, 10_000, 2_000, step=500,
                           help="More simulations = more stable percentages, but slower")
    with ctrl_r:
        seed = int(st.number_input("Random seed", value=42, step=1,
                                   help="Same seed = same results; change it to reshuffle the randomness"))

    run_btn = st.button("▶ Run simulations", type="primary")

    if run_btn:
        with st.spinner(f"Simulating {n_sims:,} tournaments …"):
            wins, rounds = monte_carlo(n_sims, predictor, seed=seed, track_rounds=True)
        st.session_state["sim_wins"]   = dict(wins)
        st.session_state["sim_n"]      = n_sims
        st.session_state["sim_rounds"] = {k: dict(v) for k, v in rounds.items()}

    if "sim_wins" not in st.session_state:
        st.info("Set parameters above and click **Run simulations**.")
        st.stop()

    wins  = st.session_state["sim_wins"]
    total = sum(wins.values())

    # ── championship probability chart ────────────────────────────────────
    st.subheader(f"Championship win probability  ({total:,} simulations)")
    st.caption(
        f"How often each team won the whole tournament across {total:,} "
        f"simulated World Cups. Bar colors show the team's confederation."
    )

    df_wins = (
        pd.DataFrame(wins.items(), columns=["team", "wins"])
        .sort_values("wins", ascending=False)
        .assign(pct=lambda d: d["wins"] / total * 100)
        .head(20)
        .reset_index(drop=True)
    )
    df_wins["confederation"] = df_wins["team"].map(
        lambda t: predictor._state.get(t, {}).get("confederation", "Other")
    )
    df_wins["color"] = df_wins["confederation"].map(CONF_COLOR).fillna(CONF_COLOR["Other"])

    # reverse so highest probability appears at top of horizontal chart
    dw = df_wins.iloc[::-1].reset_index(drop=True)

    fig_wins = go.Figure(go.Bar(
        x=dw["pct"],
        y=dw["team"],
        orientation="h",
        marker_color=dw["color"].tolist(),
        text=dw["pct"].map("{:.1f}%".format),
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
    ))
    fig_wins.update_layout(
        xaxis_title="Win probability (%)",
        height=max(420, len(dw) * 34),
        margin=dict(t=20, b=30, l=175, r=55),
        xaxis=dict(range=[0, dw["pct"].max() * 1.35], automargin=True),
        yaxis=dict(automargin=True),
    )
    st.plotly_chart(fig_wins, use_container_width=True, config=_PLOTLY_CONFIG)

    # confederation legend
    legend_md = "  ".join(
        f"<span style='color:{c}'>&#9632;</span> {conf}"
        for conf, c in CONF_COLOR.items() if conf != "Other"
    )
    st.markdown(legend_md, unsafe_allow_html=True)

    # ── knockout advancement table ─────────────────────────────────────────
    sim_rounds = st.session_state.get("sim_rounds")
    if sim_rounds:
        st.markdown("---")
        st.subheader("How far does each team go?")
        st.caption(
            "Share of simulations in which each team reached every stage — "
            "from escaping the group (R32) to lifting the trophy. "
            "Darker cell = happens more often."
        )

        STAGES = ["R32", "R16", "QF", "SF", "Final", "Champion"]
        adv = pd.DataFrame([
            {"team": t, **{s: sim_rounds.get(s, {}).get(t, 0) / total for s in STAGES}}
            for t in WC_TEAMS
        ]).sort_values(["Champion", "Final", "SF"], ascending=False)

        head = "<tr><th>Team</th>" + "".join(f"<th>{s} %</th>" for s in STAGES) + "</tr>"
        body = []
        for _, r in adv.iterrows():
            cells = [f'<td>{flag_img(r["team"])} {short_name(r["team"])}</td>']
            for s in STAGES:
                v = float(r[s])
                cells.append(
                    f'<td style="background:rgba(33,150,243,{v*0.65:.3f})">'
                    f'{v*100:.0f}%</td>'
                )
            body.append("<tr>" + "".join(cells) + "</tr>")
        st.markdown(
            f'<div class="html-table"><table><thead>{head}</thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>',
            unsafe_allow_html=True,
        )

    # ── group-stage qualification rates ───────────────────────────────────
    st.markdown("---")
    st.subheader("Group-stage qualification rates")

    gs_btn = st.button("Compute qualification rates (1 000 sims)")
    if gs_btn:
        with st.spinner("Simulating group stage 1 000 times …"):
            rng_gs       = np.random.default_rng(seed)
            qual_counts: Counter = Counter()
            third_counts: Counter = Counter()
            N_GS = 1_000
            for _ in range(N_GS):
                tables = {g: simulate_group(ts, predictor, rng_gs) for g, ts in GROUPS.items()}
                for table in tables.values():
                    qual_counts[table[0]["team"]] += 1
                    qual_counts[table[1]["team"]] += 1
                    third_counts[table[2]["team"]] += 1
        st.session_state["qual_counts"]  = dict(qual_counts)
        st.session_state["third_counts"] = dict(third_counts)
        st.session_state["gs_n"]         = N_GS

    if "qual_counts" in st.session_state:
        qc  = st.session_state["qual_counts"]
        tc  = st.session_state["third_counts"]
        n   = st.session_state["gs_n"]

        st.caption(
            "How often each team escaped its group: **Top-2 qual %** = finished "
            "1st or 2nd (guaranteed through). **3rd place %** = finished 3rd — "
            "at WC 2026 the 8 best third-placed teams also advance."
        )

        cols = st.columns(3)
        for idx, grp in enumerate(sorted(GROUPS.keys())):
            with cols[idx % 3]:
                st.markdown(f"**Group {grp}**")
                ranked = sorted(GROUPS[grp], key=lambda t: -qc.get(t, 0))
                body = "".join(
                    f"<tr><td>{flag_img(t)} {short_name(t)}</td>"
                    f"<td>{qc.get(t, 0) / n * 100:.0f}%</td>"
                    f"<td>{tc.get(t, 0) / n * 100:.0f}%</td></tr>"
                    for t in ranked
                )
                st.markdown(
                    '<div class="html-table group-table"><table>'
                    "<thead><tr><th>Team</th><th>Top-2 %</th><th>3rd %</th></tr></thead>"
                    f"<tbody>{body}</tbody></table></div>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — TEAM RANKINGS
# ══════════════════════════════════════════════════════════════════════════

elif page == "Team Rankings":
    hero("📊", "TEAM RANKINGS",
         "Elo Ratings &nbsp;·&nbsp; 49,000+ Matches &nbsp;·&nbsp; 1872–2026",
         "336 national teams ranked by predictive strength")

    st.markdown(
        "**Elo** is a strength score (the same idea used in chess): every team "
        "starts at 1500 and gains or loses points after each match — beating a "
        "strong team earns more than beating a weak one. Top national teams "
        "score around 2000+."
    )

    confs = sorted(elo_df["confederation"].dropna().unique())
    sel   = st.multiselect("Filter by confederation", confs, default=confs)

    show_all_rank = st.checkbox(f"Show all {len(elo_df)} teams", value=False,
                                help="Off: only the 48 WC 2026 participants")

    filtered = elo_df[elo_df["confederation"].isin(sel)].reset_index(drop=True)
    if not show_all_rank:
        filtered = filtered[filtered["team"].isin(WC_TEAMS)].reset_index(drop=True)
    slider_max = max(11, min(100, len(filtered)))
    top_n    = st.slider("Show top N teams", 10, slider_max, min(30, slider_max), step=5)
    top_df   = filtered.head(top_n).copy()
    top_df["color"] = top_df["confederation"].map(CONF_COLOR).fillna(CONF_COLOR["Other"])
    top_df["wc2026"] = top_df["team"].isin(WC_TEAMS)

    # reverse so highest Elo appears at top
    td = top_df.iloc[::-1].reset_index(drop=True)

    fig_elo = go.Figure(go.Bar(
        x=td["elo"],
        y=td["team"],
        orientation="h",
        marker_color=td["color"].tolist(),
        marker_line_width=td["wc2026"].map(lambda b: 2.5 if b else 0).tolist(),
        marker_line_color="white",
        text=td["elo"].map("{:.0f}".format),
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>Elo: %{x:.1f}<extra></extra>",
    ))
    fig_elo.update_layout(
        xaxis_title="Elo rating",
        height=max(420, len(td) * 28),
        margin=dict(t=20, b=30, l=175, r=55),
        xaxis=dict(range=[td["elo"].min() * 0.97, td["elo"].max() * 1.1], automargin=True),
        yaxis=dict(automargin=True),
    )
    st.plotly_chart(fig_elo, use_container_width=True, config=_PLOTLY_CONFIG)

    legend_md = "  ".join(
        f"<span style='color:{c}'>&#9632;</span> {conf}"
        for conf, c in CONF_COLOR.items() if conf != "Other"
    ) + "  &nbsp; (white border = WC 2026 participant)"
    st.markdown(legend_md, unsafe_allow_html=True)

    # ── WC 2026 teams table ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("WC 2026 teams — Elo snapshot")
    st.caption(
        "All 48 qualified teams, organised by their group at the tournament. "
        "Higher Elo = stronger team going in."
    )

    wc_df = (
        elo_df[elo_df["team"].isin(WC_TEAMS)]
        .reset_index(drop=True)
        .rename(columns={"team": "Team", "elo": "Elo", "confederation": "Confederation"})
    )

    # Add group and flag image (HTML table so the <img> tags render)
    team_to_group = {t: g for g, ts in GROUPS.items() for t in ts}
    wc_df.insert(0, "Group", wc_df["Team"].map(team_to_group))
    wc_df.insert(1, "Flag", wc_df["Team"].map(flag_img))
    wc_df = wc_df.sort_values("Group")
    wc_df["Elo"] = wc_df["Elo"].map("{:.1f}".format)

    # Manual rows: alternate shading per group + border at each group boundary
    table_cols = ["Group", "Flag", "Team", "Elo", "Confederation"]
    head = "<tr>" + "".join(f"<th>{c}</th>" for c in table_cols) + "</tr>"
    body, prev_grp = [], None
    for _, r in wc_df.iterrows():
        cls = "grp-even" if (ord(r["Group"]) - ord("A")) % 2 == 0 else ""
        if prev_grp is not None and r["Group"] != prev_grp:
            cls = (cls + " grp-start").strip()
        prev_grp = r["Group"]
        cells = "".join(f"<td>{r[c]}</td>" for c in table_cols)
        body.append(f'<tr class="{cls}">{cells}</tr>')
    st.markdown(
        f'<div class="html-table"><table><thead>{head}</thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 4 — PREDICTION TRACKER
# ══════════════════════════════════════════════════════════════════════════

else:
    hero("📊", "PREDICTION TRACKER",
         "Model vs Reality &nbsp;·&nbsp; Live WC 2026 Results",
         "How the model's pre-tournament predictions are holding up")

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_tracked_results() -> tuple[dict | None, str | None]:
        """
        Fetch live results (hourly cache); on any failure fall back to the
        last saved JSON. Returns (payload, warning_or_error).
        """
        try:
            return fetch_results.fetch_and_save(), None
        except Exception as exc:  # API key missing, network down, rate-limited…
            if RESULTS_PATH.exists():
                payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
                return payload, f"Couldn't reach the live API ({exc}); showing last saved results."
            return None, (
                "Live results are unavailable and no saved results were found. "
                "This usually means the football-data.org API key isn't configured."
            )

    def _ago(iso: str) -> str:
        try:
            then = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            mins = (datetime.now(timezone.utc) - then).total_seconds() / 60
            if mins < 1:
                return "just now"
            if mins < 60:
                return f"{int(mins)} minute{'s' if int(mins) != 1 else ''} ago"
            hours = mins / 60
            if hours < 24:
                return f"{int(hours)} hour{'s' if int(hours) != 1 else ''} ago"
            return f"{int(hours / 24)} day{'s' if int(hours / 24) != 1 else ''} ago"
        except Exception:
            return "unknown"

    top_l, top_r = st.columns([4, 1])
    with top_l:
        st.markdown(
            "Every match below was predicted **before the tournament began** — "
            "the model has never seen a single WC 2026 result. Here's how its "
            "calls compare to what actually happened."
        )
    with top_r:
        if st.button("🔄 Refresh results"):
            load_tracked_results.clear()
            st.rerun()

    payload, warning = load_tracked_results()

    if payload is None:
        st.error(warning)
        st.stop()
    if warning:
        st.warning(warning)

    matches = payload.get("matches", [])
    scored  = [m for m in matches if m.get("correct") is not None]

    st.caption(f"Last updated: {_ago(payload.get('fetched_at', ''))} · "
               f"Match data: [football-data.org](https://www.football-data.org)")

    if not scored:
        st.info(
            "No finished matches with predictions yet. Check back once WC 2026 "
            "games have been played."
        )
        st.stop()

    # ── headline stats ─────────────────────────────────────────────────────
    n          = len(scored)
    n_correct  = sum(m["correct"] for m in scored)
    acc        = n_correct / n

    # Model vs naive Elo-baseline log-loss (both leakage-free, pre-tournament)
    eps = 1e-15
    model_ll, base_ll = [], []
    for m in scored:
        a = m["actual_outcome"]
        pm = np.clip([m["p_home"], m["p_draw"], m["p_away"]], eps, 1)
        model_ll.append(-np.log(pm[a]))
        eh = predictor._state.get(m["home_team"], predictor._default_state())["elo"]
        ea = predictor._state.get(m["away_team"], predictor._default_state())["elo"]
        pb = np.clip(elo_prior_proba(eh, ea), eps, 1)
        base_ll.append(-np.log(pb[a]))
    model_ll = float(np.mean(model_ll))
    base_ll  = float(np.mean(base_ll))
    ll_delta = base_ll - model_ll  # positive = model beats baseline

    s1, s2, s3 = st.columns(3)
    cards = [
        (s1, "#3b82f6", "Matches tracked", f"{n}", f"{payload.get('n_matches', n)} finished"),
        (s2, "#2ecc71", "Correct predictions", f"{acc*100:.0f}%", f"{n_correct} of {n}"),
        (s3, "#f59e0b", "Model log-loss",
         f"{model_ll:.3f}",
         (f"{'beats' if ll_delta >= 0 else 'trails'} Elo baseline "
          f"({base_ll:.3f})")),
    ]
    for col, color, label, value, sub in cards:
        col.markdown(
            f'<div class="prob-card" style="border-left:4px solid {color}">'
            f'<div class="prob-label">{label}</div>'
            f'<div class="prob-value">{value}</div>'
            f'<div style="color:#5b6379;font-size:0.8rem;margin-top:0.25rem">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "**Log-loss** rewards confident correct calls and punishes confident "
        "wrong ones (lower is better). The **Elo baseline** is the simple "
        "\"stronger team wins\" rule — beating it means the model adds real value."
    )

    # ── results table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Match-by-match")

    OUTCOME_COLORS = ["#3b82f6", "#6b7280", "#f59e0b"]

    def _prob_mini_bar(m: dict) -> str:
        probs = [m["p_home"], m["p_draw"], m["p_away"]]
        pred  = m["predicted_outcome"]
        segs = "".join(
            f'<div style="width:{p*100:.0f}%;background:{c};'
            f'{"opacity:1" if i == pred else "opacity:0.4"}"></div>'
            for i, (p, c) in enumerate(zip(probs, OUTCOME_COLORS))
        )
        return (f'<div style="display:flex;height:7px;border-radius:4px;'
                f'overflow:hidden;width:100%;min-width:96px">{segs}</div>')

    def _wdl_cell(m: dict) -> str:
        """Percentages with a thin inline probability bar folded underneath."""
        pp = (f'{m["p_home"]*100:.0f}% / {m["p_draw"]*100:.0f}% / '
              f'{m["p_away"]*100:.0f}%')
        return (f'<div style="font-size:0.82rem;color:#93a1c8;margin-bottom:4px">{pp}</div>'
                f'{_prob_mini_bar(m)}')

    # Heat scale for the probability the model gave the ACTUAL outcome.
    # Anchored at ~33% (random across 3 outcomes): below trends amber→red,
    # above trends green, deep green only past ~60%.
    _HEAT_ANCHORS = [
        (0.00, (176,  42,  42)),   # deep red
        (0.20, (231,  76,  60)),   # red
        (0.33, (245, 158,  11)),   # amber (≈ random baseline)
        (0.50, (150, 196,  70)),   # yellow-green
        (0.60, ( 46, 204, 113)),   # green
        (1.00, ( 22, 128,  70)),   # deep green
    ]

    def _heat_rgb(p: float) -> tuple[int, int, int]:
        if p <= _HEAT_ANCHORS[0][0]:
            return _HEAT_ANCHORS[0][1]
        if p >= _HEAT_ANCHORS[-1][0]:
            return _HEAT_ANCHORS[-1][1]
        for (p0, c0), (p1, c1) in zip(_HEAT_ANCHORS, _HEAT_ANCHORS[1:]):
            if p0 <= p <= p1:
                t = (p - p0) / (p1 - p0)
                return tuple(int(round(c0[i] + t * (c1[i] - c0[i]))) for i in range(3))
        return _HEAT_ANCHORS[-1][1]

    def _lighten(rgb: tuple[int, int, int], t: float = 0.35) -> tuple[int, int, int]:
        return tuple(int(round(c + t * (255 - c))) for c in rgb)

    def _call_cell(m: dict) -> str:
        """Gradient badge = P(actual outcome); subtle ✓/✗ = was top pick right."""
        probs   = [m["p_home"], m["p_draw"], m["p_away"]]
        p_act   = probs[m["actual_outcome"]]
        r, g, b = _heat_rgb(p_act)
        lr, lg, lb = _lighten((r, g, b))
        mark     = "✓" if m["correct"] else "✗"
        mark_col = "#7fd99a" if m["correct"] else "#e57373"
        return (
            f'<span style="display:inline-block;padding:2px 9px;border-radius:7px;'
            f'background:rgba({r},{g},{b},0.18);border:1px solid rgba({r},{g},{b},0.5);'
            f'color:rgb({lr},{lg},{lb});font-weight:700">{p_act*100:.0f}%</span> '
            f'<span style="color:{mark_col};opacity:.7;font-size:.85rem" '
            f'title="top pick {"correct" if m["correct"] else "wrong"}">{mark}</span>'
        )

    rows = []
    for m in matches:
        if m.get("correct") is None:
            continue
        h, a = m["home_team"], m["away_team"]
        rows.append(
            f'<tr>'
            f'<td>{m["date"]}</td>'
            f'<td>{flag_img(h)} {short_name(h)} '
            f'<b>{m["home_score"]}–{m["away_score"]}</b> '
            f'{short_name(a)} {flag_img(a)}</td>'
            f'<td>{_wdl_cell(m)}</td>'
            f'<td>{_call_cell(m)}</td>'
            f'</tr>'
        )

    st.markdown(
        '<div class="html-table"><table><thead><tr>'
        '<th>Date</th><th>Result</th><th>Win / Draw / Loss</th><th>Call</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "**Win / Draw / Loss** lists the model's three probabilities; the thin "
        "bar beneath shows them visually (home blue · draw gray · away amber), "
        "solid segment = the model's pick. **Call** shows the probability the "
        "model gave the *actual* result, colour-coded around a 33% midpoint "
        "(one-in-three is random): green = the model saw it coming, amber ≈ a "
        "coin-toss, red = caught out. The small ✓ / ✗ marks whether the top "
        "pick was right."
    )
