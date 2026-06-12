"""
WC 2026 Prediction Dashboard
Run: streamlit run app/streamlit_app.py
"""

import sys
from collections import Counter
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

FLAGS: dict[str, str] = {
    # CONMEBOL
    "Argentina":   "🇦🇷", "Brazil":     "🇧🇷", "Colombia":   "🇨🇴",
    "Ecuador":     "🇪🇨", "Uruguay":    "🇺🇾", "Paraguay":   "🇵🇾",
    "Chile":       "🇨🇱", "Venezuela":  "🇻🇪", "Peru":       "🇵🇪",
    "Bolivia":     "🇧🇴",
    # UEFA
    "France":      "🇫🇷", "Spain":      "🇪🇸", "Germany":    "🇩🇪",
    "Portugal":    "🇵🇹", "England":    "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Netherlands": "🇳🇱",
    "Belgium":     "🇧🇪", "Italy":      "🇮🇹", "Croatia":    "🇭🇷",
    "Switzerland": "🇨🇭", "Denmark":    "🇩🇰", "Sweden":     "🇸🇪",
    "Norway":      "🇳🇴", "Austria":    "🇦🇹", "Poland":     "🇵🇱",
    "Serbia":      "🇷🇸", "Scotland":   "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Turkey":     "🇹🇷",
    "Czech Republic": "🇨🇿", "Slovakia": "🇸🇰", "Hungary":   "🇭🇺",
    "Romania":     "🇷🇴", "Ukraine":    "🇺🇦", "Greece":     "🇬🇷",
    "Bosnia and Herzegovina": "🇧🇦", "Albania": "🇦🇱", "Georgia": "🇬🇪",
    "Slovenia":    "🇸🇮",
    # CONCACAF
    "United States": "🇺🇸", "Mexico":   "🇲🇽", "Canada":    "🇨🇦",
    "Panama":      "🇵🇦", "Costa Rica": "🇨🇷", "Jamaica":   "🇯🇲",
    "Haiti":       "🇭🇹", "Honduras":  "🇭🇳", "Guatemala":  "🇬🇹",
    "El Salvador": "🇸🇻", "Trinidad and Tobago": "🇹🇹",
    "Curaçao":     "🇨🇼",
    # CAF
    "Morocco":     "🇲🇦", "Senegal":   "🇸🇳", "Egypt":      "🇪🇬",
    "Nigeria":     "🇳🇬", "Ghana":     "🇬🇭", "Ivory Coast": "🇨🇮",
    "Cameroon":    "🇨🇲", "Mali":      "🇲🇱", "Algeria":    "🇩🇿",
    "Tunisia":     "🇹🇳", "South Africa": "🇿🇦", "DR Congo":  "🇨🇩",
    "Cape Verde":  "🇨🇻",
    # AFC
    "Japan":       "🇯🇵", "South Korea": "🇰🇷", "Australia": "🇦🇺",
    "Iran":        "🇮🇷", "Saudi Arabia": "🇸🇦", "Qatar":    "🇶🇦",
    "Iraq":        "🇮🇶", "Jordan":    "🇯🇴", "Uzbekistan": "🇺🇿",
    "China":       "🇨🇳", "India":     "🇮🇳", "UAE":        "🇦🇪",
    # OFC
    "New Zealand": "🇳🇿",
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

/* identity-color accents linking metric cards to the probability bar:
   col 1 = home (blue), col 2 = draw (gray), col 3 = away (amber) */
[data-testid="stColumn"]:nth-of-type(1) [data-testid="stMetric"] { border-left: 4px solid #3b82f6; }
[data-testid="stColumn"]:nth-of-type(2) [data-testid="stMetric"] { border-left: 4px solid #6b7280; }
[data-testid="stColumn"]:nth-of-type(3) [data-testid="stMetric"] { border-left: 4px solid #f59e0b; }

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
[data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size: 1rem; }
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
page = st.sidebar.radio(
    "Navigate",
    ["Match Predictor", "Tournament Simulator", "Team Rankings"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: XGBoost + Elo prior blend (0.75/0.25) · "
    f"{len(bundle['feature_cols'])} features · "
    "data up to 2026-06-10"
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
  <p class="credits-version">v1.2 · June 2026</p>
</div>
""", unsafe_allow_html=True)


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
        home_team = st.selectbox("Home team", team_pool, index=home_default,
                                  format_func=lambda t: f"{FLAGS.get(t, '')} {t}".strip())
    with cmid:
        st.markdown("<div style='text-align:center;padding-top:2rem;font-size:1.5rem'>vs</div>",
                    unsafe_allow_html=True)
    with c2:
        away_default = team_pool.index("France") if "France" in team_pool else 1
        away_team = st.selectbox("Away team", team_pool, index=away_default,
                                 format_func=lambda t: f"{FLAGS.get(t, '')} {t}".strip())

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
    outcome_labels  = [
        f"{FLAGS.get(home_team, '')} {home_team} Win".strip(),
        "Draw",
        f"{FLAGS.get(away_team, '')} {away_team} Win".strip(),
    ]

    # ── probability display ────────────────────────────────────────────────
    st.markdown("---")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(outcome_labels[0], f"{proba[0]*100:.1f}%")
    mc2.metric("Draw",             f"{proba[1]*100:.1f}%")
    mc3.metric(outcome_labels[2], f"{proba[2]*100:.1f}%")

    # Stacked probability bar — identity colors (home blue / draw gray /
    # away amber), not good/bad semantics; flags make segments self-explanatory
    OUTCOME_COLORS = ["#3b82f6", "#6b7280", "#f59e0b"]
    bar_texts = [
        f"{FLAGS.get(home_team, '')} {proba[0]*100:.1f}%".strip(),
        f"{proba[1]*100:.1f}%",
        f"{FLAGS.get(away_team, '')} {proba[2]*100:.1f}%".strip(),
    ]
    fig_bar = go.Figure()
    for i, (label, color) in enumerate(zip(outcome_labels, OUTCOME_COLORS)):
        fig_bar.add_trace(go.Bar(
            x=[proba[i]], y=[""], orientation="h",
            name=label, marker_color=color,
            text=bar_texts[i], textposition="inside",
            insidetextanchor="middle",
            insidetextfont=dict(size=14, color="#ffffff"),
        ))
    fig_bar.update_layout(
        barmode="stack", height=70, showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, range=[0, 1]),
        yaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config=_PLOTLY_CONFIG)

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
    st.subheader("Why this prediction? (SHAP)")
    st.caption(
        f"Top feature contributions toward **{outcome_labels[predicted_class]}** "
        f"({proba[predicted_class]*100:.1f}%)"
    )

    # sv shape: (n_samples, n_features, n_classes)
    sv = explainer.shap_values(X)
    sv_for_class = sv[0, :, predicted_class]   # shape (n_features,)

    sv_series = pd.Series(sv_for_class, index=bundle["feature_cols"])
    top_feats  = sv_series.abs().nlargest(15).index
    sv_top     = sv_series[top_feats].sort_values()

    fig_shap = go.Figure(go.Bar(
        x=sv_top.values,
        y=sv_top.index,
        orientation="h",
        marker_color=["#e74c3c" if v > 0 else "#2980b9" for v in sv_top.values],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig_shap.update_layout(
        xaxis_title="SHAP value",
        height=480,
        margin=dict(l=10, r=20, t=45, b=40),
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

    ctrl_l, ctrl_r = st.columns([3, 2])
    with ctrl_l:
        n_sims = st.slider("Number of simulations", 500, 10_000, 2_000, step=500)
    with ctrl_r:
        seed = int(st.number_input("Random seed", value=42, step=1))

    run_btn = st.button("▶ Run simulations", type="primary")

    if run_btn:
        with st.spinner(f"Simulating {n_sims:,} tournaments …"):
            wins = monte_carlo(n_sims, predictor, seed=seed)
        st.session_state["sim_wins"] = dict(wins)
        st.session_state["sim_n"]    = n_sims

    if "sim_wins" not in st.session_state:
        st.info("Set parameters above and click **Run simulations**.")
        st.stop()

    wins  = st.session_state["sim_wins"]
    total = sum(wins.values())

    # ── championship probability chart ────────────────────────────────────
    st.subheader(f"Championship win probability  ({total:,} simulations)")

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

    df_wins["label"] = df_wins["team"].map(
        lambda t: f"{FLAGS.get(t, '')} {t}".strip()
    )
    # reverse so highest probability appears at top of horizontal chart
    dw = df_wins.iloc[::-1].reset_index(drop=True)

    fig_wins = go.Figure(go.Bar(
        x=dw["pct"],
        y=dw["label"],
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

        gs_rows = []
        for grp, teams in sorted(GROUPS.items()):
            for t in teams:
                gs_rows.append({
                    "Group": grp,
                    "Team":  t,
                    "Top-2 qual %": f"{qc.get(t, 0) / n * 100:.0f}%",
                    "3rd place %":  f"{tc.get(t, 0) / n * 100:.0f}%",
                    "Eliminated %": f"{(n - qc.get(t,0) - tc.get(t,0)) / n * 100:.0f}%",
                })
        gs_df = pd.DataFrame(gs_rows)

        cols = st.columns(3)
        for idx, grp in enumerate(sorted(GROUPS.keys())):
            with cols[idx % 3]:
                st.markdown(f"**Group {grp}**")
                sub = gs_df[gs_df["Group"] == grp][["Team","Top-2 qual %","3rd place %"]].set_index("Team")
                st.table(sub)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — TEAM RANKINGS
# ══════════════════════════════════════════════════════════════════════════

else:
    hero("📊", "TEAM RANKINGS",
         "Elo Ratings &nbsp;·&nbsp; 49,000+ Matches &nbsp;·&nbsp; 1872–2026",
         "336 national teams ranked by predictive strength")

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

    top_df["label"] = top_df["team"].map(lambda t: f"{FLAGS.get(t, '')} {t}".strip())
    # reverse so highest Elo appears at top
    td = top_df.iloc[::-1].reset_index(drop=True)

    fig_elo = go.Figure(go.Bar(
        x=td["elo"],
        y=td["label"],
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

    wc_df = (
        elo_df[elo_df["team"].isin(WC_TEAMS)]
        .reset_index(drop=True)
        .rename(columns={"team": "Team", "elo": "Elo", "confederation": "Confederation"})
    )
    wc_df.index = wc_df.index + 1

    # Add group and flag
    team_to_group = {t: g for g, ts in GROUPS.items() for t in ts}
    wc_df.insert(0, "Group", wc_df["Team"].map(team_to_group))
    wc_df.insert(1, "Flag", wc_df["Team"].map(lambda t: FLAGS.get(t, "")))
    wc_df = wc_df.sort_values("Group")
    wc_df["Elo"] = wc_df["Elo"].map("{:.1f}".format)

    st.table(wc_df)
