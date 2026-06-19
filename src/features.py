"""
Feature engineering pipeline for WC 2026 predictor.

Reads:  data/raw/results.csv
Writes: data/processed/features.csv    — one row per match, pre-match features + outcome
        data/processed/elo_ratings.csv — final Elo snapshot for every team

No lookahead: all features are computed from data available strictly before
each match, then state is updated afterwards.

Usage:
    python src/features.py
"""

from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from team_names import canonical

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

ELO_BASE = 1500.0
ELO_K = 40.0

CONFEDERATION: dict[str, str] = {
    **dict.fromkeys([
        "Germany", "France", "Spain", "England", "Italy", "Portugal",
        "Netherlands", "Belgium", "Croatia", "Denmark", "Switzerland",
        "Austria", "Sweden", "Norway", "Poland", "Czech Republic",
        "Hungary", "Romania", "Serbia", "Slovakia", "Slovenia", "Ukraine",
        "Russia", "Turkey", "Greece", "Albania", "Bosnia and Herzegovina",
        "Bosnia-Herzegovina", "Bulgaria", "Finland", "Georgia", "Iceland",
        "Israel", "Kosovo", "Latvia", "Lithuania", "Luxembourg", "Malta",
        "Moldova", "Montenegro", "North Macedonia", "Wales", "Scotland",
        "Republic of Ireland", "Northern Ireland", "Armenia", "Azerbaijan",
        "Belarus", "Cyprus", "Estonia", "Faroe Islands", "Gibraltar",
        "Kazakhstan", "Liechtenstein", "San Marino", "Andorra",
    ], "UEFA"),
    **dict.fromkeys([
        "Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Peru",
        "Ecuador", "Paraguay", "Venezuela", "Bolivia",
    ], "CONMEBOL"),
    **dict.fromkeys([
        "United States", "Mexico", "Canada", "Costa Rica", "Honduras",
        "Jamaica", "Panama", "Trinidad and Tobago", "El Salvador",
        "Guatemala", "Haiti", "Cuba", "Curaçao", "Bermuda", "Barbados",
        "Nicaragua", "Belize", "Dominican Republic", "Guyana", "Suriname",
    ], "CONCACAF"),
    **dict.fromkeys([
        "Morocco", "Senegal", "Nigeria", "Ghana", "Egypt", "Cameroon",
        "Algeria", "Tunisia", "Ivory Coast", "Mali", "South Africa",
        "DR Congo", "Guinea", "Burkina Faso", "Cape Verde", "Zambia",
        "Zimbabwe", "Uganda", "Tanzania", "Kenya", "Angola", "Mozambique",
        "Gabon", "Ethiopia", "Benin", "Gambia", "Mauritania", "Sierra Leone",
        "Equatorial Guinea", "Libya", "Sudan", "Madagascar", "Rwanda",
        "Comoros", "Central African Republic", "Congo", "Malawi", "Namibia",
        "Niger", "Togo",
    ], "CAF"),
    **dict.fromkeys([
        "Japan", "South Korea", "Iran", "Australia", "Saudi Arabia",
        "Qatar", "China", "Iraq", "Jordan", "Oman", "Bahrain",
        "United Arab Emirates", "UAE", "Uzbekistan", "Indonesia", "Thailand",
        "Vietnam", "India", "Syria", "Palestine", "Kuwait", "Tajikistan",
        "Kyrgyzstan", "Lebanon", "Yemen", "Afghanistan", "Bangladesh",
        "Pakistan", "Nepal", "Sri Lanka", "Maldives", "Myanmar", "Cambodia",
        "Malaysia", "Singapore", "Philippines", "Mongolia", "Hong Kong",
        "Chinese Taipei", "North Korea", "Macau", "Guam",
    ], "AFC"),
    **dict.fromkeys([
        "New Zealand", "Fiji", "Papua New Guinea", "Solomon Islands",
        "Vanuatu", "Tahiti", "New Caledonia", "Samoa", "American Samoa",
        "Cook Islands", "Tonga",
    ], "OFC"),
}


def _conf(team: str) -> str:
    # Normalize aliases/accents first so e.g. "Curacao" and "Curaçao" both land
    # in CONCACAF rather than falling through to "Other".
    return CONFEDERATION.get(canonical(team), "Other")


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _elo_update(ra: float, rb: float, sa: float) -> tuple[float, float]:
    """Update both sides. sa ∈ {1, 0.5, 0} is score from team-a's perspective."""
    ea = _elo_expected(ra, rb)
    return ra + ELO_K * (sa - ea), rb + ELO_K * ((1 - sa) - (1 - ea))


def _form_stats(recent: list[tuple[float, float]]) -> tuple[float, float]:
    """Return (win_rate, avg_goal_diff) from list of (goals_for, goals_against)."""
    if not recent:
        return 0.5, 0.0
    return (
        sum(1 for gf, ga in recent if gf > ga) / len(recent),
        sum(gf - ga for gf, ga in recent) / len(recent),
    )


def build_features(df: pd.DataFrame,
                   state_cutoff: "pd.Timestamp | None" = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Walk matches chronologically, snapshot pre-match features, append a row,
    then update state. Returns (feature_df, elo_df, team_state_df).

    feature_df rows hold PRE-match features (the training/backtest target — its
    column DEFINITIONS are unchanged). team_state_df is the SERVING snapshot:
    each team's state AFTER its last observed match (post-match form + Elo +
    conf_elo), so the live Predictor isn't one match stale. When `state_cutoff`
    is given, only matches strictly before it contribute to team_state_df (the
    serving leakage cutoff); feature_df is unaffected.
    """
    elo: dict[str, float] = defaultdict(lambda: ELO_BASE)
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    # h2h key: tuple(sorted([home, away])) — stores raw results for lookback
    h2h: dict[tuple, list] = defaultdict(list)

    # Extended state (rest days, competitive form, qualification record, stage)
    last_date: dict[str, pd.Timestamp] = {}
    comp_form: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    qual_hist: dict[str, deque] = defaultdict(deque)   # (date, won) within 730 days
    tourn_n: dict[tuple, int] = defaultdict(int)        # (team, tournament, year) -> matches played

    def _days_since(team: str, date: pd.Timestamp) -> int:
        if team not in last_date:
            return 365
        return min((date - last_date[team]).days, 365)

    def _qual_wr(team: str, date: pd.Timestamp) -> float:
        q = qual_hist[team]
        while q and (date - q[0][0]).days > 730:
            q.popleft()
        if not q:
            return 0.5
        return sum(w for _, w in q) / len(q)

    # Confederation Elo: running sum + count of every registered team
    conf_sum: dict[str, float] = defaultdict(float)
    conf_cnt: dict[str, int] = defaultdict(int)
    registered: set[str] = set()

    def _register(team: str) -> None:
        if team not in registered:
            registered.add(team)
            conf_sum[_conf(team)] += ELO_BASE
            conf_cnt[_conf(team)] += 1

    def _conf_elo(confederation: str) -> float:
        n = conf_cnt[confederation]
        return conf_sum[confederation] / n if n else ELO_BASE

    # Serving snapshot: each team's POST-match state after its last (pre-cutoff)
    # match. Overwritten chronologically, so the final write per team wins.
    team_state: dict[str, dict] = {}

    rows: list[dict] = []
    for _, m in df.iterrows():
        home, away = m["home_team"], m["away_team"]
        hs, as_ = float(m["home_score"]), float(m["away_score"])

        _register(home)
        _register(away)

        h_elo, a_elo = elo[home], elo[away]
        h_conf, a_conf = _conf(home), _conf(away)

        f_h = list(form[home])
        f_a = list(form[away])
        h_wr5, h_gd5 = _form_stats(f_h[-5:])
        a_wr5, a_gd5 = _form_stats(f_a[-5:])
        h_wr10, h_gd10 = _form_stats(f_h)
        a_wr10, a_gd10 = _form_stats(f_a)

        pair = tuple(sorted([home, away]))
        hist = h2h[pair][-10:]
        n_h2h = len(hist)
        if n_h2h:
            h2h_hw = sum(
                1 for r in hist
                if (r["h"] == home and r["hs"] > r["as"])
                or (r["h"] == away and r["as"] > r["hs"])
            )
            h2h_d = sum(1 for r in hist if r["hs"] == r["as"])
            h2h_aw = n_h2h - h2h_hw - h2h_d
            h2h_hwr = h2h_hw / n_h2h
        else:
            h2h_hw = h2h_d = h2h_aw = 0
            h2h_hwr = 0.5

        tourn_year = (m["tournament"], m["date"].year)
        is_comp = m["tournament"] != "Friendly"
        is_qual = "qualification" in str(m["tournament"]).lower()

        h_comp_wr, _ = _form_stats(list(comp_form[home]))
        a_comp_wr, _ = _form_stats(list(comp_form[away]))

        rows.append({
            "date": m["date"],
            "home_team": home,
            "away_team": away,
            "tournament": m["tournament"],
            "neutral": int(m["neutral"]),
            "home_elo": round(h_elo, 2),
            "away_elo": round(a_elo, 2),
            "elo_diff": round(h_elo - a_elo, 2),
            "home_win_rate_5": round(h_wr5, 4),
            "away_win_rate_5": round(a_wr5, 4),
            "home_gd_5": round(h_gd5, 4),
            "away_gd_5": round(a_gd5, 4),
            "home_win_rate_10": round(h_wr10, 4),
            "away_win_rate_10": round(a_wr10, 4),
            "home_gd_10": round(h_gd10, 4),
            "away_gd_10": round(a_gd10, 4),
            "h2h_n": n_h2h,
            "h2h_home_wr": round(h2h_hwr, 4),
            "h2h_home_wins": h2h_hw,
            "h2h_draws": h2h_d,
            "h2h_away_wins": h2h_aw,
            "home_confederation": h_conf,
            "away_confederation": a_conf,
            "home_conf_elo": round(_conf_elo(h_conf), 2),
            "away_conf_elo": round(_conf_elo(a_conf), 2),
            "is_world_cup": int(m["tournament"] == "FIFA World Cup"),
            "home_days_since_last": _days_since(home, m["date"]),
            "away_days_since_last": _days_since(away, m["date"]),
            "home_comp_wr_10": round(h_comp_wr, 4),
            "away_comp_wr_10": round(a_comp_wr, 4),
            "home_qual_wr": round(_qual_wr(home, m["date"]), 4),
            "away_qual_wr": round(_qual_wr(away, m["date"]), 4),
            "home_tourn_match_n": tourn_n[(home, *tourn_year)],
            "away_tourn_match_n": tourn_n[(away, *tourn_year)],
            "outcome": 0 if hs > as_ else (1 if hs == as_ else 2),
        })

        # Post-match state updates
        sa = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        new_h, new_a = _elo_update(h_elo, a_elo, sa)

        conf_sum[h_conf] += new_h - h_elo
        conf_sum[a_conf] += new_a - a_elo
        elo[home], elo[away] = new_h, new_a

        form[home].append((hs, as_))
        form[away].append((as_, hs))
        h2h[pair].append({"h": home, "hs": hs, "as": as_})

        last_date[home] = last_date[away] = m["date"]
        if is_comp:
            comp_form[home].append((hs, as_))
            comp_form[away].append((as_, hs))
        if is_qual:
            qual_hist[home].append((m["date"], 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)))
            qual_hist[away].append((m["date"], 1.0 if as_ > hs else (0.5 if hs == as_ else 0.0)))
        tourn_n[(home, *tourn_year)] += 1
        tourn_n[(away, *tourn_year)] += 1

        # Serving snapshot AFTER this match (post-match form/Elo/conf_elo). Only
        # pre-cutoff matches update it, so served state never sees the tournament.
        if state_cutoff is None or m["date"] < state_cutoff:
            for team, t_elo, t_conf in ((home, new_h, h_conf), (away, new_a, a_conf)):
                fr = list(form[team])
                wr5, gd5 = _form_stats(fr[-5:])
                wr10, gd10 = _form_stats(fr)
                team_state[team] = {
                    "team": team,
                    "elo": round(t_elo, 2),
                    "win_rate_5": round(wr5, 4),
                    "win_rate_10": round(wr10, 4),
                    "gd_5": round(gd5, 4),
                    "gd_10": round(gd10, 4),
                    "confederation": t_conf,
                    "conf_elo": round(_conf_elo(t_conf), 2),
                    "last_match_date": m["date"],
                }

    features = pd.DataFrame(rows)
    elo_df = (
        pd.DataFrame(sorted(elo.items(), key=lambda x: -x[1]), columns=["team", "elo"])
        .assign(elo=lambda d: d["elo"].round(1), confederation=lambda d: d["team"].map(_conf))
    )
    team_state_df = pd.DataFrame(
        team_state.values(),
        columns=["team", "elo", "win_rate_5", "win_rate_10", "gd_5", "gd_10",
                 "confederation", "conf_elo", "last_match_date"],
    ).sort_values("team").reset_index(drop=True)
    return features, elo_df, team_state_df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results.csv ...")
    df = (
        pd.read_csv(RAW_DIR / "results.csv", parse_dates=["date"])
        .dropna(subset=["home_score", "away_score"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    print(f"  {len(df):,} matches  ({df['date'].min().date()} – {df['date'].max().date()})")

    print("Building features ...")
    # Freeze the SERVING snapshot strictly before the tournament (same cutoff
    # the live Predictor enforces). feature_df itself is the full history.
    from simulate import PRE_TOURNAMENT_CUTOFF
    features, elo_df, team_state_df = build_features(df, state_cutoff=PRE_TOURNAMENT_CUTOFF)

    out_feat = PROCESSED_DIR / "features.csv"
    out_elo = PROCESSED_DIR / "elo_ratings.csv"
    out_state = PROCESSED_DIR / "team_state.csv"
    features.to_csv(out_feat, index=False)
    elo_df.to_csv(out_elo, index=False)
    team_state_df.to_csv(out_state, index=False)

    print(f"\nfeatures.csv    {len(features):,} rows x {features.shape[1]} cols  ->  {out_feat}")
    print(f"elo_ratings.csv {len(elo_df):,} teams                         ->  {out_elo}")
    print(f"team_state.csv  {len(team_state_df):,} teams (serving snapshot)        ->  {out_state}")

    print("\nOutcome split:")
    labels = {0: "Home win", 1: "Draw    ", 2: "Away win"}
    for k, v in features["outcome"].value_counts().sort_index().items():
        print(f"  {labels[k]}  {v:>6,}  ({v / len(features) * 100:.1f}%)")

    print("\nTop 15 teams by Elo:")
    print(elo_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
