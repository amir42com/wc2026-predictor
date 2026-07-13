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
from scorelines import scoreline_grid
from train import ELO_BLEND_W, elo_prior_proba, make_X, predict_neutral_proba
from third_place_mapping import lookup as annexe_c_lookup, check_eligibility

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR    = Path(__file__).resolve().parents[1] / "models"

# ── Leakage cutoff for the live serving path (single source of truth) ──────
# WC 2026's first match is 2026-06-11. Every prediction the app/tracker serves
# must be made from team state built ONLY from matches STRICTLY BEFORE this
# date, so a cold-start rebuild from the live martj42 feed (which gains 2026 WC
# results during the tournament) can never fold a tournament result into the
# "pre-tournament" model state. Predictor truncates its loaded state here and
# refuses to serve if any tournament-dated match leaks in; the Streamlit
# bootstrap truncates raw data here before rebuilding features/elo. This is the
# SERVING path only — backtest.py / experiments.py use their own per-tournament
# historical cutoffs and never import this constant.
PRE_TOURNAMENT_CUTOFF = pd.Timestamp("2026-06-11")

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

    def __init__(self, features_path: Path, elo_path: Path, bundle: dict,
                 cutoff: "pd.Timestamp | None" = PRE_TOURNAMENT_CUTOFF,
                 state_path: "Path | None" = None) -> None:
        self._model        = bundle["model"]
        self._feature_cols = bundle["feature_cols"]
        state_path = state_path or features_path.parent / "team_state.csv"

        df = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date")

        # Leakage guard on the H2H source: hard-cap features at the pre-tournament
        # cutoff so a rebuilt features.csv can't fold tournament matches into H2H.
        if cutoff is not None:
            n_leaked = int((df["date"] >= cutoff).sum())
            if n_leaked:
                print(f"  WARNING: features source has {n_leaked} match(es) dated "
                      f">= {cutoff.date()}; dropping them (leakage prevented).")
            df = df[df["date"] < cutoff].reset_index(drop=True)
            if df.empty:
                raise RuntimeError(
                    f"Predictor: no matches strictly before {cutoff.date()} in "
                    f"{features_path} — refusing to serve.")

        # Serving team state: the POST-final-match snapshot (team_state.csv), so
        # recent-form features include each team's LAST pre-tournament match
        # rather than the one-behind value in their last pre-match feature row.
        ts = pd.read_csv(state_path, parse_dates=["last_match_date"])
        if cutoff is not None:
            leaked = ts[ts["last_match_date"] >= cutoff]
            if len(leaked):
                # A per-team snapshot can't be repaired by dropping rows (it would
                # silently degrade real teams to defaults), so a leaked snapshot
                # means the file was built without the cutoff — refuse to serve.
                raise RuntimeError(
                    f"Predictor: team_state.csv has {len(leaked)} team(s) whose last "
                    f"match is >= {cutoff.date()} ({sorted(leaked['team'])[:5]}…); "
                    f"rebuild features with the cutoff. Refusing to serve.")
        self._state_max_date = ts["last_match_date"].max()
        if cutoff is not None:
            assert self._state_max_date < cutoff, "leak guard failed"
            print(f"  Predictor state capped at < {cutoff.date()} "
                  f"(latest match {self._state_max_date.date()})")
        self._cutoff = cutoff

        state: dict[str, dict] = {}
        for _, r in ts.iterrows():
            state[r["team"]] = {
                "elo":           float(r["elo"]),
                "win_rate_5":    float(r["win_rate_5"]),
                "gd_5":          float(r["gd_5"]),
                "win_rate_10":   float(r["win_rate_10"]),
                "gd_10":         float(r["gd_10"]),
                "confederation": str(r["confederation"]),
                "conf_elo":      float(r["conf_elo"]),
            }
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

    def predict(self, team_a: str, team_b: str) -> np.ndarray:
        """
        Return (p_a_win, p_draw, p_b_win). Cached, symmetry-corrected.
        Canonical order (alphabetical) is stored; reversed on demand.
        """
        key = frozenset({team_a, team_b})
        if key not in self._cache:
            a, b = sorted([team_a, team_b])
            elo_a = self._state.get(a, self._default_state())["elo"]
            elo_b = self._state.get(b, self._default_state())["elo"]
            # Shared deployed inference: symmetry-average both orderings, then
            # blend toward the Elo-logistic prior. build_x = feature_X.
            self._cache[key] = predict_neutral_proba(
                self._model, self.feature_X, a, b, elo_a, elo_b)

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


def _scoreline_grid_for(predictor: Predictor, home: str, away: str) -> np.ndarray:
    """
    Dixon-Coles scoreline grid consistent with the predictor's W/D/L for this
    pair, cached per unordered pair (the lambda solve is the expensive part).
    grid[i, j] = P(home scores i, away scores j).
    """
    cache = getattr(predictor, "_scoreline_grids", None)
    if cache is None:
        cache = {}
        predictor._scoreline_grids = cache
    a, b = sorted([home, away])
    if (a, b) not in cache:
        p = _safe_p(predictor.predict(a, b))
        cache[(a, b)] = scoreline_grid(p[0], p[1], p[2])[0]
    g = cache[(a, b)]
    return g if home == a else g.T


def sample_scoreline(grid: np.ndarray, outcome: int,
                     rng: np.random.Generator) -> tuple[int, int]:
    """
    Sample a scoreline from the DC grid CONDITIONAL on the already-sampled
    W/D/L outcome: restrict the grid to the outcome's region (home-win lower
    triangle / draw diagonal / away-win upper triangle), renormalise, sample
    one cell. Points, GD and GF all derive from this one scoreline, so a
    decisive outcome can never carry a level scoreline (and vice versa).
    """
    n = grid.shape[0]
    if outcome == 0:
        mask = np.tril(np.ones((n, n)), -1)
    elif outcome == 1:
        mask = np.eye(n)
    else:
        mask = np.triu(np.ones((n, n)), 1)
    flat = (grid * mask).ravel()
    total = flat.sum()
    if total <= 0:   # degenerate grid: fall back to the minimal consistent score
        return (1, 0) if outcome == 0 else ((0, 0) if outcome == 1 else (0, 1))
    idx = int(rng.choice(len(flat), p=flat / total))
    i, j = divmod(idx, n)
    return int(i), int(j)


def _mini_table(team_names: list[str],
                results: list[tuple[str, str, int, int]]) -> dict[str, tuple]:
    """(pts, gd, gf) per team over the results BETWEEN the given teams only."""
    names = set(team_names)
    t = {x: [0, 0, 0] for x in team_names}
    for home, away, hg, ag in results:
        if home not in names or away not in names:
            continue
        t[home][1] += hg - ag; t[home][2] += hg
        t[away][1] += ag - hg; t[away][2] += ag
        if hg > ag:
            t[home][0] += 3
        elif hg == ag:
            t[home][0] += 1; t[away][0] += 1
        else:
            t[away][0] += 3
    return {x: tuple(v) for x, v in t.items()}


def _rank_tied(cluster: list[dict], results: list, rng: np.random.Generator) -> list[dict]:
    """
    Rank a set of teams tied on ALL overall criteria (pts, GD, GF) per the
    FIFA 2026 regulations' among-tied-teams block: points, then goal
    difference, then goals scored in the matches BETWEEN the tied teams —
    REAPPLIED to any subset that remains tied after a partial separation.
    When the mini-table separates nothing, the remaining criteria (fair-play
    conduct points, FIFA drawing of lots) are unavailable to a simulator, so
    a documented RANDOM DRAW stands in for them.
    """
    if len(cluster) <= 1:
        return cluster
    names = [r["team"] for r in cluster]
    mini = _mini_table(names, results)
    ordered = sorted(cluster, key=lambda r: mini[r["team"]], reverse=True)

    runs: list[list[dict]] = []
    for r in ordered:
        if runs and mini[runs[-1][-1]["team"]] == mini[r["team"]]:
            runs[-1].append(r)
        else:
            runs.append([r])

    if len(runs) == 1:
        # Mini-table fully tied: random draw stands in for fair play / lots.
        rng.shuffle(cluster)
        return cluster
    out: list[dict] = []
    for run in runs:
        # Reapplication: criteria d-f re-run on the still-tied subset only.
        out.extend(_rank_tied(run, results, rng))
    return out


def _rank_group(records: dict[str, dict], results: list,
                rng: np.random.Generator) -> list[dict]:
    """
    FIFA 2026 group ranking. Overall criteria first — points, goal
    difference, goals scored (the lexicographic sort) — then the
    among-tied-teams block with reapplication (_rank_tied), then the
    documented random fallback. NOTE: the regulations rank overall GD/GF
    ABOVE head-to-head (unlike UEFA); flipping that order is a one-line
    change to the sort key below if ever ruled otherwise.
    """
    standings = sorted(records.values(),
                       key=lambda r: (r["pts"], r["gd"], r["gf"]), reverse=True)
    final: list[dict] = []
    i = 0
    while i < len(standings):
        j = i + 1
        while j < len(standings) and (
            (standings[j]["pts"], standings[j]["gd"], standings[j]["gf"])
            == (standings[i]["pts"], standings[i]["gd"], standings[i]["gf"])
        ):
            j += 1
        final.extend(_rank_tied(standings[i:j], results, rng))
        i = j
    return final


def simulate_group(
    teams: list[str],
    predictor: Predictor,
    rng: np.random.Generator,
    completed: "list[tuple[str, str, int, int]] | None" = None,
) -> list[dict]:
    """
    Simulate a 4-team round-robin. `completed` (remaining-tournament
    forecast): (home, away, home_goals, away_goals) FACTS for already-played
    fixtures in this group — those matches are not simulated, their real
    scores enter the table, and only the remaining fixtures sample from the
    model. Scorelines come from the Dixon-Coles grid conditional on the
    sampled W/D/L (sample_scoreline), so points/GD/GF are all consequences of
    one sampled score. Ranking per _rank_group (FIFA 2026 criteria).
    """
    rec = {t: {"team": t, "pts": 0, "gd": 0, "gf": 0} for t in teams}
    results: list[tuple[str, str, int, int]] = []

    done: dict[frozenset, tuple[str, str, int, int]] = {}
    for home, away, hg, ag in (completed or []):
        if home not in rec or away not in rec:
            raise ValueError(f"completed result {home} v {away} not in group {teams}")
        done[frozenset({home, away})] = (home, away, int(hg), int(ag))

    for a, b in combinations(teams, 2):
        key = frozenset({a, b})
        if key in done:
            home, away, hg, ag = done[key]
        else:
            home, away = a, b
            p = _safe_p(predictor.predict(home, away))
            outcome = int(rng.choice(3, p=p))
            hg, ag = sample_scoreline(
                _scoreline_grid_for(predictor, home, away), outcome, rng)
        results.append((home, away, hg, ag))

        rec[home]["gd"] += hg - ag;  rec[home]["gf"] += hg
        rec[away]["gd"] += ag - hg;  rec[away]["gf"] += ag
        if hg > ag:
            rec[home]["pts"] += 3
        elif hg == ag:
            rec[home]["pts"] += 1;  rec[away]["pts"] += 1
        else:
            rec[away]["pts"] += 3

    return _rank_group(rec, results, rng)


# ── R32 bracket resolver ───────────────────────────────────────────────────

def _table_key(record: dict) -> tuple:
    return (record["pts"], record["gd"], record["gf"])


def resolve_r32(group_tables: dict[str, list[dict]],
                rng: "np.random.Generator | None" = None) -> list[tuple[str, str]]:
    """
    Map R32 slot strings to actual team names using the official FIFA World Cup
    2026 Annexe C third-place table (src/third_place_mapping.py).

    The 8 best third-place teams qualify, ranked points -> goal-diff ->
    goals-for; thirds still tied after goals-for are separated by a RANDOM
    draw (the documented stand-in for the regulations' fair-play-conduct and
    drawing-of-lots criteria, which a simulator cannot know). Ties must NEVER
    fall through to incidental group-letter order — that silently favoured
    early-alphabet groups in every simulation (pre-remediation defect).

    The sorted letters of the qualified groups form the Annexe C key, and the
    table dictates EXACTLY which qualifying third each group winner faces.
    There is no greedy heuristic and no fallback: an invalid or unknown
    combination raises (annexe_c_lookup validates the 8 distinct A-L letters).
    Group-winner (1X) and runner-up (2X) resolution is unchanged.
    """
    winners = {g: t[0]["team"] for g, t in group_tables.items()}
    runners = {g: t[1]["team"] for g, t in group_tables.items()}
    thirds  = {g: t[2]         for g, t in group_tables.items()}

    # The 8 best third-place records qualify (points -> GD -> GF -> random
    # draw for residual ties; never group-letter order).
    if rng is None:
        rng = np.random.default_rng(0)
    jitter = {g: rng.random() for g in sorted(thirds)}
    ranked = sorted(thirds.items(),
                    key=lambda x: (*_table_key(x[1]), jitter[x[0]]), reverse=True)
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
    known: "dict[int, str] | None" = None,
) -> list[str]:
    """`known` (remaining-tournament forecast): match-index -> real winner for
    fixtures already decided; those are facts, not simulated."""
    winners = []
    for i, (a, b) in enumerate(matches):
        if known and i in known:
            w = known[i]
            if w not in (a, b):
                raise ValueError(
                    f"{label or 'KO'} match {i}: recorded winner {w!r} is not "
                    f"one of the pairing ({a!r}, {b!r}) — tournament state is "
                    f"inconsistent with the simulated bracket path")
        else:
            w = knockout_match(a, b, predictor, rng)
        winners.append(w)
        if verbose:
            print(f"    {a:30s} vs {b:30s}  ->  {w}")
    return winners


# ── Full tournament simulation ─────────────────────────────────────────────

def _group_facts(state: "dict | None") -> dict[str, list]:
    """Split state['group_results'] facts by group membership (validated)."""
    facts: dict[str, list] = {g: [] for g in GROUPS}
    if not state:
        return facts
    membership = {t: g for g, ts in GROUPS.items() for t in ts}
    for home, away, hg, ag in state.get("group_results", []):
        gh, ga = membership.get(home), membership.get(away)
        if gh is None or ga is None or gh != ga:
            raise ValueError(f"group result {home} v {away}: not a same-group pairing")
        facts[gh].append((home, away, hg, ag))
    return facts


def simulate_tournament(
    predictor: Predictor,
    rng: np.random.Generator,
    verbose: bool = False,
    rounds: dict[str, Counter] | None = None,
    state: "dict | None" = None,
) -> str:
    """
    If `rounds` is given, count each team's appearance per knockout stage.

    `state` turns the run into a REMAINING-TOURNAMENT FORECAST: played matches
    enter as facts, only remaining fixtures are simulated, and team ratings
    stay frozen at the pre-tournament snapshot (deliberately — mid-tournament
    rating updates would silently rebuild the model). Keys, all optional:
      "group_results": [(home, away, home_goals, away_goals), ...]
      "r32_pairs":     the REAL 16 R32 pairings (facts) — supply these when
                       the group stage is complete, because reality has
                       already resolved the tiebreak/lots randomness that
                       resolve_r32 would otherwise re-sample;
      "ko_winners":    {"R32": {match_idx: winner}, "R16": ..., "QF": ...,
                        "SF": ..., "Final": {0: winner}}
    """
    facts = _group_facts(state)
    ko_known = (state or {}).get("ko_winners", {})

    # Group stage
    group_tables: dict[str, list[dict]] = {}
    for grp, teams in GROUPS.items():
        group_tables[grp] = simulate_group(teams, predictor, rng,
                                           completed=facts[grp])
        if verbose:
            print(f"  Group {grp}:")
            for rank, row in enumerate(group_tables[grp]):
                q = " *" if rank < 2 else ("  (3rd)" if rank == 2 else "")
                print(f"    {rank+1}. {row['team']:<30s} {row['pts']}pts  GD{row['gd']:+d}{q}")

    if state and state.get("r32_pairs"):
        r32 = [tuple(p) for p in state["r32_pairs"]]
        if len(r32) != 16:
            raise ValueError("r32_pairs must list all 16 pairings")
    else:
        r32 = resolve_r32(group_tables, rng)
    if rounds is not None:
        for a, b in r32:
            rounds["R32"][a] += 1
            rounds["R32"][b] += 1

    if verbose:
        print(f"\n  Round of 32:")
    r32w = play_round(r32, predictor, rng, "R32", verbose,
                      known=ko_known.get("R32"))
    if rounds is not None:
        rounds["R16"].update(r32w)

    r16 = [(r32w[i], r32w[j]) for i, j in R16_PAIRS]
    if verbose:
        print(f"\n  Round of 16:")
    r16w = play_round(r16, predictor, rng, "R16", verbose,
                      known=ko_known.get("R16"))
    if rounds is not None:
        rounds["QF"].update(r16w)

    qf = [(r16w[i], r16w[j]) for i, j in QF_PAIRS]
    if verbose:
        print(f"\n  Quarter-finals:")
    qfw = play_round(qf, predictor, rng, "QF", verbose,
                     known=ko_known.get("QF"))
    if rounds is not None:
        rounds["SF"].update(qfw)

    sf = [(qfw[i], qfw[j]) for i, j in SF_PAIRS]
    if verbose:
        print(f"\n  Semi-finals:")
    sfw = play_round(sf, predictor, rng, "SF", verbose,
                     known=ko_known.get("SF"))
    if rounds is not None:
        rounds["Final"].update(sfw)

    finalist_a, finalist_b = sfw[0], sfw[1]
    final_known = ko_known.get("Final", {})
    if 0 in final_known:
        champion = final_known[0]
        if champion not in (finalist_a, finalist_b):
            raise ValueError("recorded champion is not one of the finalists")
    else:
        champion = knockout_match(finalist_a, finalist_b, predictor, rng)
    if rounds is not None:
        rounds["Champion"][champion] += 1
    if verbose:
        print(f"\n  FINAL:  {finalist_a}  vs  {finalist_b}  ->  {champion}")

    return champion


# ── Monte Carlo ────────────────────────────────────────────────────────────

ROUND_NAMES = ["R32", "R16", "QF", "SF", "Final", "Champion"]


def monte_carlo(n: int, predictor: Predictor, seed: int = 42,
                track_rounds: bool = False, state: "dict | None" = None):
    """
    Returns Counter of champions; with track_rounds=True returns
    (champions, rounds) where rounds maps stage name -> Counter of teams
    that reached it. With `state`, every run is a REMAINING-TOURNAMENT
    FORECAST conditioned on the recorded facts (see simulate_tournament).
    """
    if state:
        print("  MODE: remaining-tournament forecast — "
              f"{len(state.get('group_results', []))} group results, "
              f"{sum(len(v) for v in state.get('ko_winners', {}).values())} "
              "knockout results enter as facts; ratings frozen pre-tournament.")
    rng  = np.random.default_rng(seed)
    wins: Counter = Counter()
    rounds: dict[str, Counter] | None = (
        {r: Counter() for r in ROUND_NAMES} if track_rounds else None
    )
    step = max(1, n // 10)
    for i in range(n):
        wins[simulate_tournament(predictor, rng, rounds=rounds, state=state)] += 1
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
