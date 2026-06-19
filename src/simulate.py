"""
Tournament simulation for WC 2026.

Loads the trained XGBoost model and current team state, then simulates the
full 48-team bracket.  Runs a Monte Carlo to estimate championship win
probabilities.

Usage:
    python src/simulate.py              # 10 000 simulations (default)
    python src/simulate.py --n 50000
    python src/simulate.py --once       # single verbose walkthrough
    python src/simulate.py --seed 7
"""

import argparse
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import ELO_BLEND_W, elo_prior_proba, make_X
from third_place_mapping import lookup as annexe_c_lookup, check_eligibility

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR    = Path(__file__).resolve().parents[1] / "models"

# ── WC 2026 draw (5 December 2025, Washington D.C.) ───────────────────────
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico",        "South Africa",          "South Korea",  "Czech Republic"],
    "B": ["Canada",        "Bosnia and Herzegovina","Qatar",        "Switzerland"],
    "C": ["Brazil",        "Morocco",               "Haiti",        "Scotland"],
    "D": ["United States", "Paraguay",              "Australia",    "Turkey"],
    "E": ["Germany",       "Curaçao",          "Ivory Coast",  "Ecuador"],
    "F": ["Netherlands",   "Japan",                 "Sweden",       "Tunisia"],
    "G": ["Belgium",       "Egypt",                 "Iran",         "New Zealand"],
    "H": ["Spain",         "Cape Verde",            "Saudi Arabia", "Uruguay"],
    "I": ["France",        "Senegal",               "Iraq",         "Norway"],
    "J": ["Argentina",     "Algeria",               "Austria",      "Jordan"],
    "K": ["Portugal",      "DR Congo",              "Uzbekistan",   "Colombia"],
    "L": ["England",       "Croatia",               "Ghana",        "Panama"],
}

# ── R32 bracket (FIFA official knockout schedule) ─────────────────────────
# "1X" = group-X winner, "2X" = runner-up, "3XYZ..." = best eligible 3rd place.
R32_SLOTS: list[tuple[str, str]] = [
    ("1E", "3ABCDF"),   #  0
    ("1I", "3CDFGH"),   #  1
    ("2A", "2B"),        #  2
    ("1F", "2C"),        #  3
    ("2K", "2L"),        #  4
    ("1H", "2J"),        #  5
    ("1D", "3BEFIJ"),   #  6
    ("1G", "3AEHIJ"),   #  7
    ("1C", "2F"),        #  8
    ("2E", "2I"),        #  9
    ("1A", "3CEFHI"),   # 10
    ("1L", "3EHIJK"),   # 11
    ("1J", "2H"),        # 12
    ("2D", "2G"),        # 13
    ("1B", "3EFGIJ"),   # 14
    ("1K", "3DEIJL"),   # 15
]

# R32 slot index → eligible source groups for third-place qualifier
THIRD_ELIGIBLE: dict[int, frozenset] = {
     0: frozenset("ABCDF"),
     1: frozenset("CDFGH"),
     6: frozenset("BEFIJ"),
     7: frozenset("AEHIJ"),
    10: frozenset("CEFHI"),
    11: frozenset("EHIJK"),
    14: frozenset("EFGIJ"),
    15: frozenset("DEIJL"),
}

# Per-winner-slot eligible third-place groups, derived from the repo's own R32
# constraints above: each third-place R32 tuple pairs a "1X" group winner with
# its "3..." placeholder, and THIRD_ELIGIBLE lists that slot's eligible source
# groups. Built once here so the official Annexe C table is validated against
# the SAME constraints the bracket actually uses — not a second hardcoded copy.
ELIGIBLE_THIRD_BY_WINNER: dict[str, frozenset] = {}
for _idx, (_a, _b) in enumerate(R32_SLOTS):
    if _idx in THIRD_ELIGIBLE:
        _winner = _a if _a.startswith("1") else _b
        ELIGIBLE_THIRD_BY_WINNER[_winner] = THIRD_ELIGIBLE[_idx]

# Fail-fast at import: the Annexe C mapping must respect these eligibility
# constraints for every one of its 495 rows (raises ThirdPlaceMappingError
# otherwise). Structural/bijection validation already ran inside the data module.
check_eligibility(ELIGIBLE_THIRD_BY_WINNER)

# Round pairings: indices into the previous round's winner list
R16_PAIRS: list[tuple[int, int]] = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)]
QF_PAIRS:  list[tuple[int, int]] = [(0,1),(2,3),(4,5),(6,7)]
SF_PAIRS:  list[tuple[int, int]] = [(0,1),(2,3)]


# ── Predictor ─────────────────────────────────────────────────────────────

class Predictor:
    """
    Wraps the model bundle and current team state.

    predict(a, b) returns (p_a_win, p_draw, p_b_win) for a neutral-venue
    World Cup match, averaged over both home/away orderings so neither team
    gets a spurious venue advantage.  Results are cached per unordered pair.
    """

    def __init__(self, features_path: Path, elo_path: Path, bundle: dict) -> None:
        self._model        = bundle["model"]
        self._feature_cols = bundle["feature_cols"]

        df     = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date")
        elo_df = pd.read_csv(elo_path).set_index("team")

        # Latest pre-match state for each team (last row they appeared in)
        state: dict[str, dict] = {}
        for _, row in df.iterrows():
            for side in ("home", "away"):
                t = row[f"{side}_team"]
                state[t] = {
                    "win_rate_5":     row[f"{side}_win_rate_5"],
                    "gd_5":           row[f"{side}_gd_5"],
                    "win_rate_10":    row[f"{side}_win_rate_10"],
                    "gd_10":          row[f"{side}_gd_10"],
                    "confederation":  row[f"{side}_confederation"],
                    "conf_elo":       row[f"{side}_conf_elo"],
                    "elo":            row[f"{side}_elo"],
                }

        # Override Elo with post-match final ratings
        for t in state:
            if t in elo_df.index:
                state[t]["elo"]           = float(elo_df.at[t, "elo"])
                state[t]["confederation"] = str(elo_df.at[t, "confederation"])

        self._state = state

        # H2H history: frozenset({a,b}) → list of (home_team, outcome) last 10
        h2h: dict[frozenset, list] = {}
        for _, row in df.iterrows():
            key = frozenset({row["home_team"], row["away_team"]})
            h2h.setdefault(key, []).append((row["home_team"], int(row["outcome"])))
        self._h2h = {k: v[-10:] for k, v in h2h.items()}

        self._cache: dict[frozenset, np.ndarray] = {}

    # ------------------------------------------------------------------
    def _default_state(self) -> dict:
        return {
            "elo": 1500.0, "win_rate_5": 0.5, "gd_5": 0.0,
            "win_rate_10": 0.5, "gd_10": 0.0,
            "confederation": "Other", "conf_elo": 1500.0,
        }

    def _h2h_stats(self, home: str, away: str) -> tuple[int, float]:
        hist = self._h2h.get(frozenset({home, away}), [])
        n = len(hist)
        if n == 0:
            return 0, 0.5
        hw = sum(
            1 for h, o in hist
            if (h == home and o == 0) or (h == away and o == 2)
        )
        return n, hw / n

    def feature_X(self, home: str, away: str) -> pd.DataFrame:
        """Model-ready feature matrix for a single home-away fixture (one row)."""
        hs  = self._state.get(home, self._default_state())
        as_ = self._state.get(away, self._default_state())
        n_h2h, hwr = self._h2h_stats(home, away)
        row = {
            "home_elo":          hs["elo"],
            "away_elo":          as_["elo"],
            "elo_diff":          hs["elo"] - as_["elo"],
            "home_win_rate_5":   hs["win_rate_5"],
            "away_win_rate_5":   as_["win_rate_5"],
            "home_gd_5":         hs["gd_5"],
            "away_gd_5":         as_["gd_5"],
            "home_win_rate_10":  hs["win_rate_10"],
            "away_win_rate_10":  as_["win_rate_10"],
            "home_gd_10":        hs["gd_10"],
            "away_gd_10":        as_["gd_10"],
            "h2h_n":             n_h2h,
            "h2h_home_wr":       hwr,
            "home_conf_elo":     hs["conf_elo"],
            "away_conf_elo":     as_["conf_elo"],
            "neutral":           1,
            "is_world_cup":      1,
            "home_confederation":  hs["confederation"],
            "away_confederation":  as_["confederation"],
        }
        X, _ = make_X(pd.DataFrame([row]), self._feature_cols)
        return X

    def _raw_proba(self, home: str, away: str) -> np.ndarray:
        return self._model.predict_proba(self.feature_X(home, away))[0]

    def predict(self, team_a: str, team_b: str) -> np.ndarray:
        """
        Return (p_a_win, p_draw, p_b_win). Cached, symmetry-corrected.
        Canonical order (alphabetical) is stored; reversed on demand.
        """
        key = frozenset({team_a, team_b})
        if key not in self._cache:
            a, b   = sorted([team_a, team_b])
            p_ab   = self._raw_proba(a, b)
            p_ba   = self._raw_proba(b, a)
            # Average both orderings to cancel any residual home-field asymmetry
            p_model = (p_ab + p_ba[[2, 1, 0]]) / 2  # (a_win, draw, b_win)
            # Shrink toward the Elo-logistic prior (improves WC calibration)
            elo_a = self._state.get(a, self._default_state())["elo"]
            elo_b = self._state.get(b, self._default_state())["elo"]
            prior = elo_prior_proba(elo_a, elo_b)
            blended = ELO_BLEND_W * p_model + (1 - ELO_BLEND_W) * prior
            self._cache[key] = blended / blended.sum()

        p = self._cache[key]
        a = min(team_a, team_b)
        return p if team_a == a else p[[2, 1, 0]]


# ── Group-stage helpers ────────────────────────────────────────────────────

def _safe_p(p: np.ndarray) -> np.ndarray:
    """Sanitize a probability vector for rng.choice: no NaN/negatives, sums to 1."""
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=0.0)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0:
        return np.full(len(p), 1.0 / len(p))
    return p / s


def _score_for_outcome(outcome: int, rng: np.random.Generator) -> tuple[int, int]:
    """Sample a plausible scoreline given the match outcome (0/1/2)."""
    if outcome == 0:
        return int(1 + rng.poisson(0.8)), int(rng.poisson(0.5))
    if outcome == 1:
        g = int(rng.poisson(1.0))
        return g, g
    return int(rng.poisson(0.5)), int(1 + rng.poisson(0.8))


def simulate_group(
    teams: list[str],
    predictor: Predictor,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Simulate a 4-team round-robin.  Returns standings sorted by:
    points → goal-diff → goals-for → head-to-head (points, then goal-diff)
    → random shuffle, applied ONLY to teams still tied after head-to-head.
    Each record: {team, pts, gd, gf, h2h_pts, h2h_gd, group}.
    """
    rec = {t: {"team": t, "pts": 0, "gd": 0, "gf": 0} for t in teams}
    h2h_pts: dict[str, dict[str, int]] = {t: {o: 0 for o in teams if o != t} for t in teams}
    h2h_gd:  dict[str, dict[str, int]] = {t: {o: 0 for o in teams if o != t} for t in teams}

    for home, away in combinations(teams, 2):
        p = _safe_p(predictor.predict(home, away))
        outcome = int(rng.choice(3, p=p))
        hg, ag = _score_for_outcome(outcome, rng)

        rec[home]["gd"] += hg - ag;  rec[home]["gf"] += hg
        rec[away]["gd"] += ag - hg;  rec[away]["gf"] += ag
        h2h_gd[home][away] += hg - ag
        h2h_gd[away][home] += ag - hg

        if outcome == 0:
            rec[home]["pts"]      += 3
            h2h_pts[home][away]   += 3
        elif outcome == 1:
            rec[home]["pts"]      += 1;  rec[away]["pts"]      += 1
            h2h_pts[home][away]   += 1;  h2h_pts[away][home]   += 1
        else:
            rec[away]["pts"]      += 3
            h2h_pts[away][home]   += 3

    standings = sorted(rec.values(), key=lambda r: (r["pts"], r["gd"], r["gf"]), reverse=True)

    # Break ties within equal-score clusters using H2H then random
    final: list[dict] = []
    i = 0
    while i < len(standings):
        j = i + 1
        while j < len(standings) and (
            standings[j]["pts"] == standings[i]["pts"]
            and standings[j]["gd"]  == standings[i]["gd"]
            and standings[j]["gf"]  == standings[i]["gf"]
        ):
            j += 1
        cluster = standings[i:j]
        if len(cluster) > 1:
            cluster_teams = [r["team"] for r in cluster]

            def h2h_key(r: dict) -> tuple[int, int]:
                # Mini-table among the tied teams only: H2H points, then H2H GD.
                others = [o for o in cluster_teams if o != r["team"]]
                return (sum(h2h_pts[r["team"]][o] for o in others),
                        sum(h2h_gd[r["team"]][o] for o in others))

            cluster.sort(key=h2h_key, reverse=True)

            # Random tiebreak ONLY within sub-groups that are STILL tied after
            # head-to-head — never reshuffle the whole cluster (that would undo
            # the H2H ordering). Shuffling each tied run independently keeps the
            # H2H ranking between runs and stays deterministic given the rng.
            resolved: list[dict] = []
            k = 0
            while k < len(cluster):
                m = k + 1
                while m < len(cluster) and h2h_key(cluster[m]) == h2h_key(cluster[k]):
                    m += 1
                tied = cluster[k:m]
                if len(tied) > 1:
                    rng.shuffle(tied)
                resolved.extend(tied)
                k = m
            cluster = resolved

        final.extend(cluster)
        i = j

    return final


# ── R32 bracket resolver ───────────────────────────────────────────────────

def _table_key(record: dict) -> tuple:
    return (record["pts"], record["gd"], record["gf"])


def resolve_r32(group_tables: dict[str, list[dict]]) -> list[tuple[str, str]]:
    """
    Map R32 slot strings to actual team names using the official FIFA World Cup
    2026 Annexe C third-place table (src/third_place_mapping.py).

    The 8 best third-place teams qualify (ranked by points -> goal-diff ->
    goals-for, unchanged); the sorted letters of their groups form the Annexe C
    key, and the table dictates EXACTLY which qualifying third each group winner
    faces. There is no greedy heuristic and no fallback: an invalid or unknown
    combination raises (annexe_c_lookup validates the 8 distinct A-L letters).
    Group-winner (1X) and runner-up (2X) resolution is unchanged.
    """
    winners = {g: t[0]["team"] for g, t in group_tables.items()}
    runners = {g: t[1]["team"] for g, t in group_tables.items()}
    thirds  = {g: t[2]         for g, t in group_tables.items()}

    # The 8 best third-place records qualify (points -> GD -> GF).
    ranked = sorted(thirds.items(), key=lambda x: _table_key(x[1]), reverse=True)
    qualified = [g for g, _ in ranked[:8]]

    # Official Annexe C assignment for this exact set of eight groups:
    # {"1A": "3X", ...} mapping each winner slot to the third it faces.
    assignment = annexe_c_lookup(qualified)

    def _resolve(slot: str, idx: int) -> str:
        if slot.startswith("1"):   return winners[slot[1]]
        if slot.startswith("2"):   return runners[slot[1]]
        # Third-place placeholder: the winner it faces is the "1X" partner in
        # this R32 tuple; Annexe C names which group's third that is.
        a, b = R32_SLOTS[idx]
        winner_slot = a if a.startswith("1") else b
        third_group = assignment[winner_slot][1]
        return thirds[third_group]["team"]

    return [(_resolve(a, i), _resolve(b, i)) for i, (a, b) in enumerate(R32_SLOTS)]


# ── Knockout helpers ───────────────────────────────────────────────────────

def knockout_match(
    team_a: str,
    team_b: str,
    predictor: Predictor,
    rng: np.random.Generator,
) -> str:
    """Simulate a single-elimination match.  Draws go to a penalty coin-flip."""
    p = _safe_p(predictor.predict(team_a, team_b))
    outcome = int(rng.choice(3, p=p))
    if outcome == 0:
        return team_a
    if outcome == 2:
        return team_b
    # Penalty shootout: renormalise win probabilities as the tiebreaker
    p_a = p[0] / (p[0] + p[2]) if (p[0] + p[2]) > 0 else 0.5
    return team_a if rng.random() < p_a else team_b


def play_round(
    matches: list[tuple[str, str]],
    predictor: Predictor,
    rng: np.random.Generator,
    label: str = "",
    verbose: bool = False,
) -> list[str]:
    winners = []
    for a, b in matches:
        w = knockout_match(a, b, predictor, rng)
        winners.append(w)
        if verbose:
            print(f"    {a:30s} vs {b:30s}  ->  {w}")
    return winners


# ── Full tournament simulation ─────────────────────────────────────────────

def simulate_tournament(
    predictor: Predictor,
    rng: np.random.Generator,
    verbose: bool = False,
    rounds: dict[str, Counter] | None = None,
) -> str:
    """If `rounds` is given, count each team's appearance per knockout stage."""
    # Group stage
    group_tables: dict[str, list[dict]] = {}
    for grp, teams in GROUPS.items():
        group_tables[grp] = simulate_group(teams, predictor, rng)
        if verbose:
            print(f"  Group {grp}:")
            for rank, row in enumerate(group_tables[grp]):
                q = " *" if rank < 2 else ("  (3rd)" if rank == 2 else "")
                print(f"    {rank+1}. {row['team']:<30s} {row['pts']}pts  GD{row['gd']:+d}{q}")

    r32 = resolve_r32(group_tables)
    if rounds is not None:
        for a, b in r32:
            rounds["R32"][a] += 1
            rounds["R32"][b] += 1

    if verbose:
        print(f"\n  Round of 32:")
    r32w = play_round(r32, predictor, rng, verbose=verbose)
    if rounds is not None:
        rounds["R16"].update(r32w)

    r16 = [(r32w[i], r32w[j]) for i, j in R16_PAIRS]
    if verbose:
        print(f"\n  Round of 16:")
    r16w = play_round(r16, predictor, rng, verbose=verbose)
    if rounds is not None:
        rounds["QF"].update(r16w)

    qf = [(r16w[i], r16w[j]) for i, j in QF_PAIRS]
    if verbose:
        print(f"\n  Quarter-finals:")
    qfw = play_round(qf, predictor, rng, verbose=verbose)
    if rounds is not None:
        rounds["SF"].update(qfw)

    sf = [(qfw[i], qfw[j]) for i, j in SF_PAIRS]
    if verbose:
        print(f"\n  Semi-finals:")
    sfw = play_round(sf, predictor, rng, verbose=verbose)
    if rounds is not None:
        rounds["Final"].update(sfw)

    finalist_a, finalist_b = sfw[0], sfw[1]
    champion = knockout_match(finalist_a, finalist_b, predictor, rng)
    if rounds is not None:
        rounds["Champion"][champion] += 1
    if verbose:
        print(f"\n  FINAL:  {finalist_a}  vs  {finalist_b}  ->  {champion}")

    return champion


# ── Monte Carlo ────────────────────────────────────────────────────────────

ROUND_NAMES = ["R32", "R16", "QF", "SF", "Final", "Champion"]


def monte_carlo(n: int, predictor: Predictor, seed: int = 42,
                track_rounds: bool = False):
    """
    Returns Counter of champions; with track_rounds=True returns
    (champions, rounds) where rounds maps stage name -> Counter of teams
    that reached it.
    """
    rng  = np.random.default_rng(seed)
    wins: Counter = Counter()
    rounds: dict[str, Counter] | None = (
        {r: Counter() for r in ROUND_NAMES} if track_rounds else None
    )
    step = max(1, n // 10)
    for i in range(n):
        wins[simulate_tournament(predictor, rng, rounds=rounds)] += 1
        if (i + 1) % step == 0:
            print(f"  {i+1:>6,} / {n:,} simulations done ...", flush=True)
    if track_rounds:
        return wins, rounds
    return wins


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate WC 2026")
    parser.add_argument("--n",    type=int, default=10_000, help="Monte Carlo iterations")
    parser.add_argument("--once", action="store_true",      help="single verbose walkthrough")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bundle_path = MODELS_DIR / "xgb_wc2026.joblib"
    if not bundle_path.exists():
        sys.exit(f"Model not found: {bundle_path}\nRun src/train.py first.")

    print("Loading model and team state ...")
    bundle    = joblib.load(bundle_path)
    predictor = Predictor(
        PROCESSED_DIR / "features.csv",
        PROCESSED_DIR / "elo_ratings.csv",
        bundle,
    )
    print(f"  {len(predictor._state)} teams loaded")

    if args.once:
        rng = np.random.default_rng(args.seed)
        print("\n=== Single tournament walkthrough ===\n")
        champion = simulate_tournament(predictor, rng, verbose=True)
        print(f"\nChampion: {champion}")
        return

    print(f"\nRunning {args.n:,} Monte Carlo simulations ...")
    wins = monte_carlo(args.n, predictor, seed=args.seed)
    total = sum(wins.values())

    print(f"\nWC 2026 championship win probabilities ({total:,} sims)\n")
    print(f"  {'Team':<30s}  {'Wins':>5}  {'%':>6}")
    print(f"  {'-'*30}  {'-'*5}  {'-'*6}")
    for team, count in wins.most_common(20):
        pct = count / total * 100
        bar = "#" * int(pct / 1.5)
        print(f"  {team:<30s}  {count:>5}  {pct:>5.2f}%  {bar}")

    never = sorted(t for g in GROUPS.values() for t in g if t not in wins)
    if never:
        print(f"\n  Never won: {', '.join(never)}")


if __name__ == "__main__":
    main()
