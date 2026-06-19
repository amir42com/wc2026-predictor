"""
WC 2026 Prediction Dashboard
Run: streamlit run app/streamlit_app.py
"""

import json
import sys
import urllib.parse
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
from simulate import (GROUPS, Predictor, monte_carlo, simulate_group, resolve_r32,
                      PRE_TOURNAMENT_CUTOFF)
from train import ELO_BLEND_W, elo_prior_proba, make_X
from scorelines import top_scorelines
from explain import build_reasons
import fetch_results

RESULTS_PATH = Path(__file__).resolve().parents[1] / "data" / "wc2026_results.json"

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR    = Path(__file__).resolve().parents[1] / "models"
RAW_DIR       = Path(__file__).resolve().parents[1] / "data" / "raw"

WC_TEAMS = sorted({t for teams in GROUPS.values() for t in teams})

# ── Color / side invariant (single source of truth) ────────────────────────
# Home / first-listed team is ALWAYS shown on the LEFT and in blue; the away /
# second team is ALWAYS on the RIGHT and in amber; a draw is the grey middle.
# These never flip with the matchup. Used by the result strip, the scorelines,
# the team comparison and the reasons panel so the convention can't regress.
HOME_COLOR = "#3b82f6"   # home / first-listed — left + blue
DRAW_COLOR = "#6b7280"   # draw — grey middle
AWAY_COLOR = "#f59e0b"   # away / second — right + amber
OUTCOME_COLORS = [HOME_COLOR, DRAW_COLOR, AWAY_COLOR]   # [home, draw, away]

CONF_COLOR = {
    "UEFA":     "#3498db",
    "CONMEBOL": "#2ecc71",
    "CONCACAF": "#e67e22",
    "CAF":      "#e74c3c",
    "AFC":      "#9b59b6",
    "OFC":      "#1abc9c",
    "Other":    "#95a5a6",
}


def conf_legend_html(extra: str = "") -> str:
    """Confederation legend as a flex-wrap row so items never clip on mobile."""
    items = "".join(
        '<span style="display:inline-flex;align-items:center;gap:5px;white-space:nowrap">'
        f'<span style="width:11px;height:11px;border-radius:2px;background:{c};'
        f'display:inline-block;flex:0 0 auto"></span>{conf}</span>'
        for conf, c in CONF_COLOR.items() if conf != "Other"
    )
    tail = (f'<span style="color:#93a1c8;white-space:nowrap">{extra}</span>'
            if extra else "")
    return (f'<div style="display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;'
            f'font-size:0.85rem;color:#cdd6ee;margin-top:4px">{items}{tail}</div>')

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
            # Leakage cutoff: the live martj42 feed gains 2026 WC results during
            # the tournament. Truncate raw data strictly before the cutoff BEFORE
            # building features/Elo so a cold-start rebuild can never fold a
            # tournament result into the deployed model state (features.csv,
            # elo_ratings.csv and the trained model are all pre-tournament).
            _df = _df[_df["date"] < PRE_TOURNAMENT_CUTOFF].reset_index(drop=True)
            feat_df, elo_df, state_df = _feat.build_features(
                _df, state_cutoff=PRE_TOURNAMENT_CUTOFF)
            feat_df.to_csv( PROCESSED_DIR / "features.csv",    index=False)
            elo_df.to_csv(  PROCESSED_DIR / "elo_ratings.csv", index=False)
            state_df.to_csv(PROCESSED_DIR / "team_state.csv",  index=False)
            st.write(f"Features built: {len(feat_df):,} matches.")

        if missing_model:
            st.write("Training XGBoost model (~30 seconds)…")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            _df2 = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
            # Ship a model trained on ALL pre-tournament data (1872–2026), not
            # the pre-2018 eval slice — same logic as src/train.py main().
            model, feature_cols = _tr.train_model(
                _df2[_tr.production_mask(_df2)].reset_index(drop=True))
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
    page_icon="⚽",   # browser-tab favicon (football)
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

# ── page icons (Tabler outline set; monochrome, accent-coloured) ────────────
# Each page uses the same icon in its nav item and (on desktop) its hero.
# Path data copied verbatim from @tabler/icons outline SVGs (all four verified
# to exist: crystal-ball, trophy, list-numbers, radar).
_ACCENT = "#4fc3f7"
_TABLER_PATHS: dict[str, list[str]] = {
    "crystal-ball": [
        "M6.73 17.018a8 8 0 1 1 10.54 0",
        "M5 19a2 2 0 0 0 2 2h10a2 2 0 1 0 0 -4h-10a2 2 0 0 0 -2 2",
        "M11 7a3 3 0 0 0 -3 3",
    ],
    "trophy": [
        "M8 21l8 0", "M12 17l0 4", "M7 4l10 0",
        "M17 4v8a5 5 0 0 1 -10 0v-8",
        "M3 9a2 2 0 1 0 4 0a2 2 0 1 0 -4 0",
        "M17 9a2 2 0 1 0 4 0a2 2 0 1 0 -4 0",
    ],
    "list-numbers": [
        "M11 6h9", "M11 12h9", "M12 18h8",
        "M4 16a2 2 0 1 1 4 0c0 .591 -.5 1 -1 1.5l-3 2.5h4",
        "M6 10v-6l-2 2",
    ],
    "radar": [
        "M21 12h-8a1 1 0 1 0 -1 1v8a9 9 0 0 0 9 -9",
        "M16 9a5 5 0 1 0 -7 7",
        "M20.486 9a9 9 0 1 0 -11.482 11.495",
    ],
}


def tabler_svg(name: str, stroke: str = "currentColor", size: int = 24) -> str:
    """Inline Tabler outline SVG; monochrome (colour via `stroke`/currentColor)."""
    body = "".join(f'<path d="{d}"/>' for d in _TABLER_PATHS[name])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


def _tabler_data_uri(name: str, stroke: str, size: int = 22) -> str:
    return "data:image/svg+xml," + urllib.parse.quote(tabler_svg(name, stroke=stroke, size=size))


# Nav order must match the st.radio options: Predict, Tracker, Simulate, Rankings.
# Hero of each page reuses the same icon (HERO_* constants below).
NAV_ICON_ORDER = ["crystal-ball", "radar", "trophy", "list-numbers"]
HERO_PREDICT  = tabler_svg("crystal-ball")
HERO_SIMULATE = tabler_svg("trophy")
HERO_RANKINGS = tabler_svg("list-numbers")
HERO_TRACKER  = tabler_svg("radar")


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

/* ── result unit: three cards + a thin proportional base strip ── */
.prob-unit { margin: 0.2rem 0 1.1rem 0; }
.prob-cards { display: flex; gap: 0.6rem; align-items: stretch; }
.prob-cards > .prob-card { flex: 1; margin-bottom: 0; }
.prob-strip {
    display: flex; height: 10px; margin-top: 8px;
    border-radius: 5px; overflow: hidden;   /* symmetric pill — not tucked/clipped */
}
.prob-strip > div { height: 100%; }
@media (max-width: 768px) { .prob-cards { flex-direction: column; } }

/* ── Most Likely Scorelines panel ── */
.scl-panel { display: flex; flex-direction: column; gap: 6px; margin: 0.2rem 0 0.2rem; }
.scl-row {
    display: flex; align-items: center; gap: 12px;
    background: linear-gradient(160deg, #151d3b 0%, #10162e 100%);
    border: 1px solid #26305a; border-radius: 10px; padding: 8px 14px;
}
.scl-rank {
    flex: 0 0 auto; width: 22px; height: 22px; border-radius: 50%;
    border: 1px solid #4fc3f7; color: #4fc3f7;
    font-size: 0.72rem; font-weight: 700;
    display: inline-flex; align-items: center; justify-content: center;
}
.scl-score {
    font-family: 'Sora', sans-serif; font-size: 1.05rem; font-weight: 700;
    letter-spacing: 0.03em; white-space: nowrap;
}
.scl-prob {
    margin-left: auto; color: #cdd6ee; font-weight: 600;
    font-variant-numeric: tabular-nums;
}

/* ── HTML tables (flag images need unescaped HTML) ── */
/* Only the table scrolls sideways on narrow screens — never the page.
   max-width:100% keeps the wrapper inside its column; a thin styled
   scrollbar makes the horizontal scroll discoverable (Task 5). */
.html-table {
    overflow-x: auto;
    max-width: 100%;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: thin;
    scrollbar-color: #2a3560 transparent;
}
.html-table::-webkit-scrollbar { height: 6px; }
.html-table::-webkit-scrollbar-thumb { background: #2a3560; border-radius: 3px; }
.html-table::-webkit-scrollbar-track { background: transparent; }
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
    justify-content: center;   /* center tabs within the full-width bar */
    width: 100%;               /* fill the bar so centering actually centers */
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
    box-shadow: 0 2px 14px rgba(79, 195, 247, 0.40);
}
[data-testid="stMain"] [data-testid="stRadio"] label:has(input:checked) p {
    color: #fff; font-weight: 600;
}

/* sticky nav: pin the radio's container below Streamlit's ~60px top toolbar.
   FRAGILE: relies on Streamlit DOM testids + header height. */
[data-testid="stHeader"] { background: #0c1124; }
[data-testid="stMain"] [data-testid="stElementContainer"]:has(> [data-testid="stRadio"]) {
    position: sticky;
    top: 56px;
    z-index: 99;
    background: #0c1124;
    overflow: visible;
    padding: 0.55rem 0.7rem;
    /* full-bleed: cancel the block-container's horizontal padding so the bar
       spans the full main-content width. Base -1rem is <= the minimum 1rem
       padding at any width (overflow-safe); desktop gets -5rem only where the
       padding is confirmed 5rem. FRAGILE: tied to Streamlit's padding values
       and ~864px breakpoint. */
    margin-inline: -1rem;
    /* stretch to the full main-content width (Streamlit caps containers at
       max-width:100%, which clamps the bar to the content column and leaves
       the tabs + underline left-aligned). width = content + cancelled padding. */
    width: calc(100% + 2rem);
    max-width: none;
    box-sizing: border-box;
    border-bottom: 1px solid #1d2547;
}
@media (min-width: 864px) {
    [data-testid="stMain"] [data-testid="stElementContainer"]:has(> [data-testid="stRadio"]) {
        margin-inline: -5rem;
        width: calc(100% + 10rem);
    }
}
[data-testid="stMain"] [data-testid="stRadio"],
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] {
    overflow: visible;
}

/* Task 2 — mobile: all 4 destinations on one row, each a small icon ABOVE a
   short label, evenly distributed with NO horizontal scroll. The real (long)
   label text is collapsed and the icon + short label are injected per position.
   ORDER-DEPENDENT: the nth-of-type list must match the st.radio options order
   (Match Predictor, Tournament Simulator, Team Rankings, Prediction Tracker). */
@media (max-width: 768px) {
    [data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] {
        flex-wrap: nowrap;
        justify-content: space-between;
        gap: 4px;
        width: 100%;
        overflow: visible;
        padding: 2px 0 4px;
    }
    [data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] > label {
        flex: 1 1 0;
        min-width: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 3px;
        background: transparent;
        border: none;
        box-shadow: none;
        border-radius: 12px;
        padding: 6px 2px;
        white-space: nowrap;
    }
    /* collapse the real (long) label text */
    [data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] > label p {
        font-size: 0; line-height: 0; height: 0; margin: 0;
    }
    /* icon (above) + short label (below). The Tabler SVG per position is
       injected as a content:url() image in a separate generated <style>
       (see _nav_icon_css) — these rules set size + the short labels. */
    [data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] > label::before {
        display: inline-block; width: 1.45rem; height: 1.45rem; line-height: 0;
    }
    [data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] > label::after {
        font-size: 0.7rem; line-height: 1; color: #93a1c8; font-weight: 600;
    }
    [data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-of-type(1)::after  { content: "Predict"; }
    [data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-of-type(2)::after  { content: "Tracker"; }
    [data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-of-type(3)::after  { content: "Simulate"; }
    [data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-of-type(4)::after  { content: "Rankings"; }
    /* active highlight (no pill; tint + accent label) */
    [data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
        background: rgba(79, 195, 247, 0.12);
        box-shadow: none;
    }
    [data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked)::after {
        color: #4fc3f7; font-weight: 700;
    }
}

/* ── Match Predictor "vs" divider ──
   Desktop: padded down to sit beside the dropdowns (unchanged look). */
.vs-divider { text-align: center; padding-top: 2rem; font-size: 1.5rem; color: #cdd6ee; }
/* Mobile: columns stack, so the big top padding becomes dead space — compact
   it into a slim inline divider and tighten the gap between the stacked
   dropdowns. Scoped to the selector row via :has(.vs-divider). */
@media (max-width: 768px) {
    .vs-divider {
        padding-top: 0; margin: 0; font-size: 0.9rem;
        font-weight: 600; color: #93a1c8; letter-spacing: 0.05em;
    }
    [data-testid="stHorizontalBlock"]:has(.vs-divider) { gap: 0.2rem; }
    [data-testid="stHorizontalBlock"]:has(.vs-divider) [data-testid="stWidgetLabel"] {
        margin-bottom: 0.1rem;
    }
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
.hero .hero-icon { line-height: 0; color: #4fc3f7; margin-bottom: 0.2rem; }
.hero .hero-icon svg { width: 3rem; height: 3rem; }
/* mobile: drop the hero icon entirely to keep the banner compact */
@media (max-width: 768px) { .hero .hero-icon { display: none; } }
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

# Mobile nav icons: inject each Tabler SVG (accent-coloured) as a content:url()
# image, generated in Python so the data URIs are URL-encoded correctly. Mobile
# only — desktop nav stays text pills. Order matches NAV_ICON_ORDER / st.radio.
_nav_icon_css = "\n".join(
    f'[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] > '
    f'label:nth-of-type({i})::before {{ content: url("{_tabler_data_uri(name, _ACCENT)}"); }}'
    for i, name in enumerate(NAV_ICON_ORDER, start=1)
)
st.markdown(f"<style>@media (max-width: 768px) {{\n{_nav_icon_css}\n}}</style>",
            unsafe_allow_html=True)


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
  <p class="credits-version">v2.0 · June 2026</p>
</div>
""", unsafe_allow_html=True)


# ── top navigation pills (always visible — sidebar collapses on mobile) ────

page = st.radio(
    "Navigate",
    ["Match Predictor", "Prediction Tracker", "Tournament Simulator", "Team Rankings"],
    horizontal=True,
    label_visibility="collapsed",
)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_next_fixture() -> dict:
    """
    Hourly-cached next WC fixture. Raises on no-fixture/API-failure on purpose:
    st.cache_data does NOT cache exceptions, so a transient failure isn't pinned
    for the whole hour — the next load retries instead of serving a stale fallback.
    """
    fx = fetch_results.get_next_fixture()
    if not fx:
        raise RuntimeError("no upcoming fixture")
    return fx


def next_fixture_defaults() -> tuple[str, str]:
    """
    Default (home, away) for the Match Predictor = the next upcoming WC 2026
    fixture. Each team must be a valid WC participant; anything unmatched or any
    API failure falls back to a sensible default. Never raises.
    """
    fallback = ("Argentina", "France")
    try:
        fx = _cached_next_fixture()
        # Auto-advance: if the hourly-cached fixture has already kicked off (and
        # isn't the live match), the cache is stale — clear it once and refetch
        # so the default moves on to the next genuinely-upcoming match.
        ko = fetch_results._parse_utc(fx.get("utc"))
        if (ko is not None and ko <= datetime.now(timezone.utc)
                and fx.get("status") not in ("IN_PLAY", "PAUSED")):
            _cached_next_fixture.clear()
            fx = _cached_next_fixture()
    except Exception:
        return fallback
    home = fx.get("home") if fx.get("home") in WC_TEAMS else fallback[0]
    away = fx.get("away") if fx.get("away") in WC_TEAMS else fallback[1]
    return fallback if home == away else (home, away)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════

if page == "Match Predictor":
    hero(HERO_PREDICT, "FIFA WORLD CUP 2026",
         "United States &nbsp;·&nbsp; Canada &nbsp;·&nbsp; Mexico",
         "June 11 – July 19, 2026")

    # ── team selectors ─────────────────────────────────────────────────────
    show_all = st.session_state.get("show_all_teams", False)
    team_pool = ALL_TEAMS if show_all else WC_TEAMS

    # Default to the next real WC 2026 fixture. `index` only seeds the FIRST
    # render — Streamlit persists the user's choice afterwards, so a manual
    # change is never overridden on rerun.
    def_home, def_away = next_fixture_defaults()

    def _idx(pool: list[str], pref: str, fb: str, fb_i: int) -> int:
        if pref in pool:
            return pool.index(pref)
        return pool.index(fb) if fb in pool else fb_i

    c1, cmid, c2 = st.columns([5, 1, 5])
    with c1:
        home_team = st.selectbox("Home team", team_pool,
                                 index=_idx(team_pool, def_home, "Argentina", 0))
    with cmid:
        st.markdown("<div class='vs-divider'>vs</div>", unsafe_allow_html=True)
    with c2:
        away_team = st.selectbox("Away team", team_pool,
                                 index=_idx(team_pool, def_away, "France", 1))

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
    X, _ = make_X(pd.DataFrame([feat_row]), bundle["feature_cols"])   # for SHAP (home orientation)
    if neutral:
        # Neutral World Cup match: use the EXACT shared deployed inference
        # (symmetry-averaged + Elo-blended) so this page agrees with the
        # simulator and tracker to the digit. Oriented home_team -> away_team.
        proba = predictor.predict(home_team, away_team)
    else:
        # Home-advantage mode (user turned neutral off): single orientation,
        # then the same Elo blend — averaging would cancel the home edge.
        proba = bundle["model"].predict_proba(X)[0]   # [p_home_win, p_draw, p_away_win]
        proba = ELO_BLEND_W * proba + (1 - ELO_BLEND_W) * elo_prior_proba(hs["elo"], as_["elo"])
    predicted_class = int(proba.argmax())
    outcome_labels  = [f"{home_team} Win", "Draw", f"{away_team} Win"]

    # ── probability display ────────────────────────────────────────────────
    # Identity colours (home blue / draw grey / away amber) from the page-wide
    # invariant (OUTCOME_COLORS), not good/bad semantics. HTML cards + strip so
    # flag images render on every platform.
    st.markdown("---")
    card_labels = [
        f"{flag_img(home_team)} {home_team} Win",
        "Draw",
        f"{flag_img(away_team)} {away_team} Win",
    ]
    # Outcome labels follow the home-blue / away-amber invariant (draw stays a
    # neutral grey); the big % numbers keep their high-contrast near-white.
    label_colors = [HOME_COLOR, "#93a1c8", AWAY_COLOR]
    cards = "".join(
        f'<div class="prob-card" style="border-left:4px solid {color}">'
        f'<div class="prob-label" style="color:{lc}">{label}</div>'
        f'<div class="prob-value">{p*100:.1f}%</div></div>'
        for label, p, color, lc in zip(card_labels, proba, OUTCOME_COLORS, label_colors)
    )
    # A single thin proportional strip (no text) attached to the bottom of the
    # three-card row: home blue · draw grey · away amber. The percentages are
    # shown only once (in the cards); the strip just visualises the split, so
    # the cards and strip read as one unit.
    strip = "".join(
        f'<div style="width:{p*100:.2f}%;background:{c}"></div>'
        for p, c in zip(proba, OUTCOME_COLORS)
    )
    st.markdown(
        f'<div class="prob-unit"><div class="prob-cards">{cards}</div>'
        f'<div class="prob-strip">{strip}</div></div>',
        unsafe_allow_html=True,
    )

    # ── most likely scorelines (additive layer; rolls up to the W/D/L above) ─
    st.subheader("Most Likely Scorelines")
    scl = top_scorelines(proba[0], proba[1], proba[2], top_n=5)
    scl_rows = "".join(
        f'<div class="scl-row">'
        f'<span class="scl-rank">{i}</span>'
        f'<span class="scl-score">'
        f'<span style="color:{HOME_COLOR}">{hg}</span>'
        f'<span style="color:#5b6379;margin:0 7px">–</span>'
        f'<span style="color:{AWAY_COLOR}">{ag}</span></span>'
        f'<span class="scl-prob">{pr*100:.1f}%</span>'
        f'</div>'
        for i, ((hg, ag), pr) in enumerate(scl, start=1)
    )
    st.markdown(
        f'<div style="font-size:0.82rem;color:#93a1c8;margin-bottom:6px">'
        f'{flag_img(home_team)} <span style="color:{HOME_COLOR}">{short_name(home_team)}</span> – '
        f'<span style="color:{AWAY_COLOR}">{short_name(away_team)}</span> {flag_img(away_team)}</div>'
        f'<div class="scl-panel">{scl_rows}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "For entertainment only. Match outcome probabilities are generally more "
        "reliable than exact score predictions."
    )

    # ── team comparison ────────────────────────────────────────────────────
    st.subheader("Team comparison")
    # Home (left) header tinted blue, away (right) header amber — same side /
    # colour invariant as the result strip, scorelines and reasons panel.
    comp_metrics = [
        ("Elo rating",               f"{hs['elo']:.0f}",            f"{as_['elo']:.0f}"),
        ("Form — win rate (last 5)", f"{hs['win_rate_5']*100:.0f}%", f"{as_['win_rate_5']*100:.0f}%"),
        ("Form — avg GD (last 5)",   f"{hs['gd_5']:+.2f}",          f"{as_['gd_5']:+.2f}"),
        ("Form — win rate (last 10)", f"{hs['win_rate_10']*100:.0f}%", f"{as_['win_rate_10']*100:.0f}%"),
        ("Form — avg GD (last 10)",  f"{hs['gd_10']:+.2f}",         f"{as_['gd_10']:+.2f}"),
        ("Confederation",            hs["confederation"],           as_["confederation"]),
    ]
    cmp_head = (
        f'<tr><th></th>'
        f'<th style="color:{HOME_COLOR}">{flag_img(home_team)} {home_team}</th>'
        f'<th style="color:{AWAY_COLOR}">{flag_img(away_team)} {away_team}</th></tr>'
    )
    cmp_body = "".join(
        f'<tr><td style="color:#93a1c8">{m}</td><td>{h}</td><td>{a}</td></tr>'
        for m, h, a in comp_metrics
    )
    st.markdown(
        f'<div class="html-table"><table><thead>{cmp_head}</thead>'
        f'<tbody>{cmp_body}</tbody></table></div>',
        unsafe_allow_html=True,
    )
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

    # ── Why is X favoured?  (plain-language reasons) ───────────────────────
    # The model's SHAP contributions are grouped into a fixed set of reasons
    # (see src/explain.py) and summed per group — SHAP is additive, so this
    # loses nothing and still reconciles to the W/D/L shown above. SHAP is
    # oriented to the HOME class, so a positive group sum favours the home
    # team (blue) and a negative sum favours the away team (amber). The
    # favourite (heading) is whoever has the higher win probability.
    fav_is_home = proba[0] >= proba[2]
    fav_team = home_team if fav_is_home else away_team
    opp_team = away_team if fav_is_home else home_team

    st.subheader(f"Why is {fav_team} favoured?")

    sv = explainer.shap_values(X)  # (n_samples, n_features, n_classes)
    shap_home = dict(zip(bundle["feature_cols"], sv[0, :, 0]))   # HOME-oriented
    feat_vals = dict(zip(bundle["feature_cols"], X.iloc[0].values))
    ctx = {
        "home_team": home_team, "away_team": away_team,
        "home_conf": hs["confederation"], "away_conf": as_["confederation"],
    }
    reasons = build_reasons(shap_home, feat_vals, ctx, max_reasons=5)

    # One-line lead: top reasons for the favourite, plus the biggest against.
    favouring = [r for r in reasons if r["home_favoured"] == fav_is_home]
    against   = [r for r in reasons if r["home_favoured"] != fav_is_home]
    if favouring:
        lead = (f"{fav_team} is favoured mainly on "
                + " and ".join(r["headline"].lower() for r in favouring[:2]))
        if against:
            lead += f", though {against[0]['headline'].lower()} narrows the gap"
    else:
        lead = f"{fav_team} is only narrowly favoured"
    st.markdown(f"**{lead}.**")

    st.markdown(
        f"<span style='color:{HOME_COLOR}'>&#9632;</span> favours {home_team}"
        f" &nbsp;·&nbsp; "
        f"<span style='color:{AWAY_COLOR}'>&#9632;</span> favours {away_team}"
        f" &nbsp;·&nbsp; <span style='opacity:.6'>click / tap a row to expand"
        f" its detailed breakdown</span>",
        unsafe_allow_html=True,
    )

    def _esc(s: str) -> str:
        return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))

    # Each reason is a native <details> accordion: click / tap / keyboard
    # (Enter or Space) toggles it, and the browser exposes the open/closed
    # state to assistive tech. No JavaScript (Streamlit strips <script>). The
    # dot, headline, description and magnitude bar live in <summary> so they
    # stay visible at all times; only the raw-factor breakdown sits behind the
    # toggle. The top (largest) reason opens by default as an expand hint.
    _CHEVRON = (
        "<svg class='wcr-chev' viewBox='0 0 24 24' fill='none' "
        "stroke='currentColor' stroke-width='2' stroke-linecap='round' "
        "stroke-linejoin='round'><path d='M6 9l6 6l6 -6'/></svg>"
    )
    rows_html = []
    for i, r in enumerate(reasons):
        home_fav = r["home_favoured"]
        color = HOME_COLOR if home_fav else AWAY_COLOR   # dot: blue home / amber away
        pct   = max(4.0, r["magnitude"] * 100.0)       # bar fill, min visible
        # Side invariant: home (blue) extends LEFT, away (amber) extends RIGHT,
        # matching the result strip / scorelines / comparison table.
        fill_l = f"{pct:.0f}" if home_fav else "0"
        fill_r = f"{pct:.0f}" if not home_fav else "0"

        detail_rows = []
        for label, val in r["detail"]:
            dcolor = HOME_COLOR if val > 0 else AWAY_COLOR
            detail_rows.append(
                f"<div class='wcr-drow'><span>{_esc(label)}</span>"
                f"<span style='color:{dcolor}'>{val:+.3f}</span></div>"
            )
        detail_html = (
            "<div class='wcr-detail'><div class='wcr-dhead'>Behind this group "
            "(raw model factors and their contributions):</div>"
            + "".join(detail_rows) + "</div>"
        ) if detail_rows else ""

        rows_html.append(
            f"<details class='wcr-row'{' open' if i == 0 else ''}>"
            f"<summary class='wcr-summary'>"
            f"<span class='wcr-dot' style='background:{color}'></span>"
            f"<div class='wcr-body'>"
            f"<div class='wcr-head'>{_esc(r['headline'])}</div>"
            f"<div class='wcr-desc'>{_esc(r['description'])}</div>"
            f"<div class='wcr-track'>"
            f"<div class='wcr-half wcr-left'>"
            f"<div class='wcr-fill' style='width:{fill_l}%;background:{HOME_COLOR}'></div></div>"
            f"<div class='wcr-mid'></div>"
            f"<div class='wcr-half wcr-right'>"
            f"<div class='wcr-fill' style='width:{fill_r}%;background:{AWAY_COLOR}'></div></div>"
            f"</div></div>{_CHEVRON}</summary>{detail_html}</details>"
        )

    panel_css = """
    <style>
    .wcr-panel{display:flex;flex-direction:column;gap:10px;margin-top:6px;}
    .wcr-row{border-radius:10px;background:rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.07);overflow:hidden;}
    .wcr-row[open]{background:rgba(255,255,255,0.06);}
    .wcr-summary{display:flex;gap:12px;align-items:flex-start;cursor:pointer;
        padding:12px 14px;list-style:none;outline:none;}
    .wcr-summary::-webkit-details-marker{display:none;}
    .wcr-summary::marker{content:"";}
    .wcr-summary:hover{background:rgba(255,255,255,0.04);}
    .wcr-summary:focus-visible{outline:2px solid #4fc3f7;outline-offset:-2px;}
    .wcr-dot{flex:0 0 10px;width:10px;height:10px;border-radius:50%;margin-top:7px;}
    .wcr-body{flex:1;min-width:0;}
    .wcr-head{font-weight:700;font-size:1.0rem;}
    .wcr-desc{font-size:.9rem;opacity:.85;margin-top:2px;}
    .wcr-track{display:flex;align-items:center;height:14px;margin-top:8px;}
    .wcr-half{flex:1;display:flex;height:7px;}
    .wcr-left{justify-content:flex-end;}
    .wcr-right{justify-content:flex-start;}
    .wcr-fill{height:7px;border-radius:4px;min-width:0;}
    .wcr-mid{width:2px;height:14px;background:#9ca3af;flex:0 0 2px;}
    .wcr-chev{flex:0 0 18px;width:18px;height:18px;margin-top:3px;color:#cdd6ee;
        opacity:.55;transition:transform .2s ease;}
    .wcr-row[open] .wcr-chev{transform:rotate(180deg);}
    .wcr-detail{margin:0 14px 12px 36px;padding:8px 10px;border-radius:8px;
        background:rgba(0,0,0,0.25);font-size:.82rem;}
    .wcr-dhead{opacity:.6;margin-bottom:4px;}
    .wcr-drow{display:flex;justify-content:space-between;gap:12px;
        padding:2px 0;font-variant-numeric:tabular-nums;}
    </style>
    """
    st.markdown(
        panel_css + "<div class='wcr-panel'>" + "".join(rows_html) + "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — TOURNAMENT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════

elif page == "Tournament Simulator":
    hero(HERO_SIMULATE, "TOURNAMENT SIMULATOR",
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
        y=dw["team"].map(short_name),
        orientation="h",
        marker_color=dw["color"].tolist(),
        text=dw["pct"].map("{:.1f}%".format),
        textposition="outside",
        textfont=dict(size=12),
        cliponaxis=False,
        hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
    ))
    fig_wins.update_layout(
        xaxis_title="Win probability (%)",
        # min ~26px/bar so bars stay readable; short y-labels + automargin keep
        # the left margin tight so bars get maximum width on mobile (Task 6)
        height=max(360, len(dw) * 30),
        margin=dict(t=12, b=28, l=8, r=44),
        bargap=0.22,
        xaxis=dict(range=[0, dw["pct"].max() * 1.32], automargin=True,
                   tickfont=dict(size=11)),
        yaxis=dict(automargin=True, tickfont=dict(size=12)),
    )
    st.plotly_chart(fig_wins, use_container_width=True, config=_PLOTLY_CONFIG)

    st.markdown(conf_legend_html(), unsafe_allow_html=True)

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
    hero(HERO_RANKINGS, "TEAM RANKINGS",
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
        y=td["team"].map(short_name),
        orientation="h",
        marker_color=td["color"].tolist(),
        marker_line_width=td["wc2026"].map(lambda b: 2.5 if b else 0).tolist(),
        marker_line_color="white",
        text=td["elo"].map("{:.0f}".format),
        textposition="outside",
        textfont=dict(size=12),
        cliponaxis=False,
        hovertemplate="%{y}<br>Elo: %{x:.1f}<extra></extra>",
    ))
    fig_elo.update_layout(
        xaxis_title="Elo rating",
        height=max(360, len(td) * 28),
        margin=dict(t=12, b=28, l=8, r=44),
        bargap=0.2,
        xaxis=dict(range=[td["elo"].min() * 0.97, td["elo"].max() * 1.1],
                   automargin=True, tickfont=dict(size=11)),
        yaxis=dict(automargin=True, tickfont=dict(size=12)),
    )
    st.plotly_chart(fig_elo, use_container_width=True, config=_PLOTLY_CONFIG)

    st.markdown(conf_legend_html("· white border = WC 2026 participant"),
                unsafe_allow_html=True)

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
    hero(HERO_TRACKER, "PREDICTION TRACKER",
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

    # Model vs naive Elo-baseline metrics (both leakage-free, pre-tournament)
    eps = 1e-15
    model_ll, base_ll, base_correct = [], [], 0
    for m in scored:
        a = m["actual_outcome"]
        pm = np.clip([m["p_home"], m["p_draw"], m["p_away"]], eps, 1)
        model_ll.append(-np.log(pm[a]))
        eh = predictor._state.get(m["home_team"], predictor._default_state())["elo"]
        ea = predictor._state.get(m["away_team"], predictor._default_state())["elo"]
        pb = np.clip(elo_prior_proba(eh, ea), eps, 1)
        base_ll.append(-np.log(pb[a]))
        if int(np.argmax(pb)) == a:
            base_correct += 1
    model_ll  = float(np.mean(model_ll))
    base_ll   = float(np.mean(base_ll))
    base_acc  = base_correct / n
    acc_delta = (acc - base_acc) * 100          # percentage points; +ve = good
    ll_gap    = model_ll - base_ll              # +ve = trailing (worse)

    # Prominent small-sample caveat — these numbers are not final
    st.markdown(
        f'<div style="display:inline-block;background:rgba(245,158,11,0.12);'
        f'border:1px solid rgba(245,158,11,0.4);color:#f0b54a;border-radius:999px;'
        f'padding:3px 13px;font-size:0.8rem;font-weight:600;margin-bottom:0.7rem">'
        f'⚠ Small sample — through {n} match{"es" if n != 1 else ""}, '
        f'expect these to swing</div>',
        unsafe_allow_html=True,
    )

    def _chip(text: str, is_good: bool) -> str:
        col = "#2ecc71" if is_good else "#e57373"
        bg  = "rgba(46,204,113,0.13)" if is_good else "rgba(229,115,115,0.13)"
        return (f'<span style="display:inline-block;background:{bg};'
                f'border:1px solid {col}55;color:{col};border-radius:999px;'
                f'padding:1px 8px;font-size:0.72rem;font-weight:600">{text}</span>')

    # Cohesive cyan/blue family — colour carries NO good/bad meaning
    s1, s2, s3 = st.columns(3)
    cards = [
        (s1, "#4fc3f7", "Matches tracked", f"{n}",
         f'<span style="color:#5b6379;font-size:0.8rem">'
         f'{payload.get("n_matches", n)} finished so far</span>'),
        (s2, "#38bdf8", "Correct predictions", f"{acc*100:.0f}%",
         _chip(f"{acc_delta:+.0f} pp vs Elo", acc_delta >= 0)
         + f' <span style="color:#5b6379;font-size:0.78rem">{n_correct} of {n}</span>'),
        (s3, "#60a5fa", "Model log-loss", f"{model_ll:.3f}",
         _chip(f"{ll_gap:+.3f} vs Elo", ll_gap <= 0)
         + f' <span style="color:#5b6379;font-size:0.78rem">lower is better</span>'),
    ]
    for col, color, label, value, sub in cards:
        col.markdown(
            f'<div class="prob-card" style="border-left:4px solid {color}">'
            f'<div class="prob-label">{label}</div>'
            f'<div class="prob-value">{value}</div>'
            f'<div style="margin-top:0.35rem">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "Card colours are neutral; the green/red **chips** carry the verdict "
        "versus the **Elo baseline** (the simple \"stronger team wins\" rule). "
        "**Log-loss** rewards confident correct calls and punishes confident "
        "wrong ones — lower is better, so a negative gap means the model beats "
        "the baseline."
    )

    # ── results table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Match-by-match")

    # Categorical (not performance) colours: bar segments and number tints.
    # Draw NUMBER uses a light grey (mid-grey reads as "disabled" on dark).
    # Same home/draw/away invariant as the rest of the app (see OUTCOME_COLORS);
    # NUMBER_COLORS only lightens the draw text so it reads on the dark bar.
    NUMBER_COLORS = [HOME_COLOR, "#b4bdd1", AWAY_COLOR]  # text: home / draw(light) / away

    def _wdl_cell(m: dict) -> str:
        """
        Three pre-tournament probabilities (home/draw/away), each tinted to its
        zone (home blue · draw light-grey · away amber). Numbers sit in fixed
        thirds above the bar — home left, draw centre, away right. The outcome
        that ACTUALLY happened is larger + bold at full strength; the other two
        are smaller and dimmed. In the (thicker) bar, the live segment is full
        opacity with a faint inner highlight; the others are dimmed. No
        performance colour-coding.
        """
        probs   = [m["p_home"], m["p_draw"], m["p_away"]]
        hs, as_ = m["home_score"], m["away_score"]
        actual  = 0 if hs > as_ else (1 if hs == as_ else 2)   # home perspective
        aligns  = ["left", "center", "right"]

        labels = "".join(
            f'<div style="flex:1 1 0;text-align:{aligns[i]};white-space:nowrap;'
            f'color:{NUMBER_COLORS[i]};'
            + ("font-size:16px;font-weight:700;opacity:1" if i == actual
               else "font-size:13px;font-weight:400;opacity:0.5")
            + f'">{p*100:.0f}%</div>'
            for i, p in enumerate(probs)
        )
        bar = "".join(
            f'<div style="flex:0 0 {p*100:.4f}%;background:{c};'
            + ("opacity:1;box-shadow:inset 0 1.5px 0 rgba(255,255,255,0.4)"
               if i == actual else "opacity:0.3")
            + '"></div>'
            for i, (p, c) in enumerate(zip(probs, OUTCOME_COLORS))
        )
        return (
            f'<div style="min-width:150px">'
            f'<div style="display:flex;align-items:flex-end;margin-bottom:4px">{labels}</div>'
            f'<div style="display:flex;height:13px;border-radius:5px;overflow:hidden">'
            f'{bar}</div>'
            f'</div>'
        )

    _TRK_CSS = """
    <style>
    .trk-list{display:flex;flex-direction:column;gap:8px;margin-top:6px;}
    .trk-row{border-radius:10px;background:rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.07);overflow:hidden;}
    .trk-row[open]{background:rgba(255,255,255,0.06);}
    .trk-summary{cursor:pointer;padding:10px 14px;list-style:none;outline:none;}
    .trk-summary::-webkit-details-marker{display:none;}
    .trk-summary::marker{content:"";}
    .trk-summary:hover{background:rgba(255,255,255,0.04);}
    .trk-summary:focus-visible{outline:2px solid #4fc3f7;outline-offset:-2px;}
    .trk-head{display:flex;align-items:center;gap:12px;}
    .trk-date{flex:0 0 auto;color:#93a1c8;font-size:0.82rem;
        font-variant-numeric:tabular-nums;}
    .trk-result{flex:1;min-width:0;font-size:0.95rem;}
    .trk-chev{flex:0 0 18px;width:18px;height:18px;color:#cdd6ee;opacity:.55;
        transition:transform .2s ease;}
    .trk-row[open] .trk-chev{transform:rotate(180deg);}
    .trk-wdl{margin-top:8px;}
    .trk-detail{padding:2px 14px 12px;}
    .trk-dhead{opacity:.65;font-size:0.8rem;margin:2px 0 8px;}
    .trk-verdict{margin-top:9px;font-size:0.9rem;}
    .trk-note{margin-top:6px;font-size:0.78rem;opacity:.6;line-height:1.35;}
    .scl-hit{border-color:#4fc3f7;
        background:linear-gradient(160deg,#16263f 0%,#101a2e 100%);}
    .scl-tag{font-size:0.62rem;font-weight:700;letter-spacing:0.05em;
        text-transform:uppercase;color:#4fc3f7;border:1px solid #4fc3f7;
        border-radius:999px;padding:1px 7px;margin-left:8px;}
    </style>
    """

    # Tabler chevron — same accordion pattern as the SHAP reasons panel.
    _TRK_CHEVRON = (
        "<svg class='trk-chev' viewBox='0 0 24 24' fill='none' "
        "stroke='currentColor' stroke-width='2' stroke-linecap='round' "
        "stroke-linejoin='round'><path d='M6 9l6 6l6 -6'/></svg>"
    )

    def _scoreline_detail(m: dict) -> str:
        """Top-5 pre-match scorelines (home-away) with the actual score flagged."""
        hs, as_ = m["home_score"], m["away_score"]
        sl = m.get("scorelines")
        if not sl:   # older cached payloads: derive from the locked pre-match W/D/L
            sl = [[int(hg), int(ag), round(float(pr), 4)] for (hg, ag), pr in
                  top_scorelines(m["p_home"], m["p_draw"], m["p_away"], top_n=5)]
        hit_rank = next((i + 1 for i, (hg, ag, _) in enumerate(sl)
                         if hg == hs and ag == as_), None)

        items = "".join(
            f'<div class="scl-row{" scl-hit" if (hg == hs and ag == as_) else ""}">'
            f'<span class="scl-rank">{i}</span>'
            f'<span class="scl-score"><span style="color:{HOME_COLOR}">{hg}</span>'
            f'<span style="color:#5b6379;margin:0 7px">–</span>'
            f'<span style="color:{AWAY_COLOR}">{ag}</span></span>'
            + ('<span class="scl-tag">actual</span>' if (hg == hs and ag == as_) else "")
            + f'<span class="scl-prob">{pr*100:.1f}%</span></div>'
            for i, (hg, ag, pr) in enumerate(sl, start=1)
        )
        if hit_rank is not None:
            verdict = (f'Actual <b>{hs}–{as_}</b> — the model\'s '
                       f'<b>#{hit_rank}</b> scoreline at {sl[hit_rank-1][2]*100:.1f}%')
        else:
            verdict = f'Actual <b>{hs}–{as_}</b> — outside the model\'s top 5'
        return (
            f'<div class="trk-detail">'
            f'<div class="trk-dhead">Model\'s most likely scorelines (pre-match)'
            f'</div><div class="scl-panel">{items}</div>'
            f'<div class="trk-verdict">{verdict}</div>'
            f'<div class="trk-note">For entertainment only — outcome probabilities '
            f'are far more reliable than exact scores, so a result outside the top 5 '
            f'(expected roughly half the time) is not the model failing.</div></div>'
        )

    cards_html, first = [], True
    for m in matches:
        if m.get("p_home") is None:   # no model prediction for this match
            continue
        h, a = m["home_team"], m["away_team"]
        open_attr = " open" if first else ""   # open the newest as an expand hint
        first = False
        cards_html.append(
            f'<details class="trk-row"{open_attr}><summary class="trk-summary">'
            f'<div class="trk-head">'
            f'<span class="trk-date">{m["date"]}</span>'
            f'<span class="trk-result">{flag_img(h)} {short_name(h)} '
            f'<b>{m["home_score"]}–{m["away_score"]}</b> '
            f'{short_name(a)} {flag_img(a)}</span>{_TRK_CHEVRON}</div>'
            f'<div class="trk-wdl">{_wdl_cell(m)}</div></summary>'
            f'{_scoreline_detail(m)}</details>'
        )

    st.markdown(_TRK_CSS + '<div class="trk-list">' + "".join(cards_html)
                + '</div>', unsafe_allow_html=True)
    st.caption(
        "The three numbers are the model's pre-tournament **win / draw / loss** "
        "probabilities, tinted to their zone (home blue · draw grey · away amber). "
        "The larger **bold** number and the highlighted full-strength bar segment "
        "mark the outcome that actually happened; the other two are dimmed. "
        "**Expand a match** to see the model's most likely scorelines."
    )
