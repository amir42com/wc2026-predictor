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
from train import make_X

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
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

_bootstrap_if_needed()

bundle, predictor, explainer, elo_df = load_resources()
ALL_TEAMS = sorted(predictor._state.keys())

# ── sidebar navigation ─────────────────────────────────────────────────────

st.sidebar.title("⚽ WC 2026 Predictor")
page = st.sidebar.radio(
    "Navigate",
    ["Match Predictor", "Tournament Simulator", "Team Rankings"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: XGBoost · "
    f"{len(bundle['feature_cols'])} features · "
    "data up to 2026-06-10"
)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════

if page == "Match Predictor":
    st.title("Match Predictor")
    st.caption("Win / draw / loss probabilities for any international fixture.")

    # ── team selectors ─────────────────────────────────────────────────────
    c1, cmid, c2 = st.columns([5, 1, 5])
    with c1:
        home_default = ALL_TEAMS.index("Argentina") if "Argentina" in ALL_TEAMS else 0
        home_team = st.selectbox("Home team", ALL_TEAMS, index=home_default)
    with cmid:
        st.markdown("<div style='text-align:center;padding-top:2rem;font-size:1.5rem'>vs</div>",
                    unsafe_allow_html=True)
    with c2:
        away_default = ALL_TEAMS.index("France") if "France" in ALL_TEAMS else 1
        away_team = st.selectbox("Away team", ALL_TEAMS, index=away_default)

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
    predicted_class = int(proba.argmax())
    outcome_labels  = [f"{home_team} Win", "Draw", f"{away_team} Win"]

    # ── probability display ────────────────────────────────────────────────
    st.markdown("---")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(outcome_labels[0], f"{proba[0]*100:.1f}%")
    mc2.metric("Draw",             f"{proba[1]*100:.1f}%")
    mc3.metric(outcome_labels[2], f"{proba[2]*100:.1f}%")

    # Stacked probability bar
    fig_bar = go.Figure()
    for i, (label, color) in enumerate(zip(outcome_labels, ["#2ecc71", "#f39c12", "#e74c3c"])):
        fig_bar.add_trace(go.Bar(
            x=[proba[i]], y=[""], orientation="h",
            name=label, marker_color=color,
            text=f"{proba[i]*100:.1f}%", textposition="inside",
            insidetextanchor="middle",
        ))
    fig_bar.update_layout(
        barmode="stack", height=70, showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, range=[0, 1]),
        yaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

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
    st.dataframe(comp_df, use_container_width=True)

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
        xaxis_title="SHAP value  (red = pushes toward this outcome, blue = pushes away)",
        height=480,
        margin=dict(l=10, r=20, t=10, b=40),
    )
    st.plotly_chart(fig_shap, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — TOURNAMENT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════

elif page == "Tournament Simulator":
    st.title("Tournament Simulator")
    st.caption("Monte Carlo simulation of the full WC 2026 bracket.")

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

    fig_wins = go.Figure(go.Bar(
        x=df_wins["team"],
        y=df_wins["pct"],
        marker_color=df_wins["color"].tolist(),
        text=df_wins["pct"].map("{:.1f}%".format),
        textposition="outside",
        hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
    ))
    fig_wins.update_layout(
        yaxis_title="Win probability (%)",
        xaxis_tickangle=-35,
        height=500,
        margin=dict(t=30, b=10),
        yaxis=dict(range=[0, df_wins["pct"].max() * 1.18]),
    )
    st.plotly_chart(fig_wins, use_container_width=True)

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
                st.dataframe(sub, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — TEAM RANKINGS
# ══════════════════════════════════════════════════════════════════════════

else:
    st.title("Team Rankings")
    st.caption("Elo ratings computed from all international results 1872 – 2026-06-10.")

    confs = sorted(elo_df["confederation"].dropna().unique())
    sel   = st.multiselect("Filter by confederation", confs, default=confs)

    filtered = elo_df[elo_df["confederation"].isin(sel)].reset_index(drop=True)
    top_n    = st.slider("Show top N teams", 10, min(100, len(filtered)), 30, step=5)
    top_df   = filtered.head(top_n).copy()
    top_df["color"] = top_df["confederation"].map(CONF_COLOR).fillna(CONF_COLOR["Other"])
    top_df["wc2026"] = top_df["team"].isin(WC_TEAMS)

    fig_elo = go.Figure(go.Bar(
        x=top_df["team"],
        y=top_df["elo"],
        marker_color=top_df["color"].tolist(),
        marker_line_width=top_df["wc2026"].map(lambda b: 2.5 if b else 0).tolist(),
        marker_line_color="white",
        text=top_df["elo"].map("{:.0f}".format),
        textposition="outside",
        hovertemplate="%{x}<br>Elo: %{y:.1f}<extra></extra>",
    ))
    fig_elo.update_layout(
        yaxis_title="Elo rating",
        xaxis_tickangle=-40,
        height=540,
        margin=dict(t=30, b=10),
        yaxis=dict(range=[top_df["elo"].min() * 0.97, top_df["elo"].max() * 1.025]),
    )
    st.plotly_chart(fig_elo, use_container_width=True)

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

    # Add group
    team_to_group = {t: g for g, ts in GROUPS.items() for t in ts}
    wc_df.insert(0, "Group", wc_df["Team"].map(team_to_group))
    wc_df = wc_df.sort_values("Group")

    st.dataframe(wc_df, use_container_width=True, height=600)
