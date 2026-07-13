"""
Phase 5 simulator tests: conditional scoreline sampling (the _score_for_outcome
replacement), FIFA 2026 tiebreaks with reapplication, third-place ranking
without group-letter fallthrough, tournament-state conditioning, seed
determinism, and the Dixon-Coles rho fit recipe.

All simulator tests run against a stub predictor — no model bundle, no WC 2026
data. Runs standalone (`python tests/test_simulator.py`, exit 0/1) and under
pytest.
"""

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import simulate as sim  # noqa: E402
from scorelines import dc_grid, grid_wdl  # noqa: E402


class StubPredictor:
    """predict(a,b) from fixed per-team strengths; no model, no state files."""

    def __init__(self, strengths: dict[str, float] | None = None):
        self._s = strengths or {}
        self._cache: dict = {}

    def predict(self, a: str, b: str) -> np.ndarray:
        sa, sb = self._s.get(a, 1500.0), self._s.get(b, 1500.0)
        e = 1.0 / (1.0 + 10.0 ** ((sb - sa) / 400.0))
        p = np.array([0.75 * e, 0.25, 0.75 * (1.0 - e)])
        return p / p.sum()


# ── conditional scoreline sampling ──────────────────────────────────────────

def test_sampled_scorelines_consistent_with_outcome():
    grid = dc_grid(1.4, 1.1)
    rng = np.random.default_rng(7)
    for outcome, cmp in [(0, lambda h, a: h > a),
                         (1, lambda h, a: h == a),
                         (2, lambda h, a: h < a)]:
        for _ in range(500):
            h, a = sim.sample_scoreline(grid, outcome, rng)
            assert cmp(h, a), f"outcome {outcome} produced {h}-{a}"


def test_scoreline_marginals_match_grid_regions():
    """Sampling outcome~WDL then scoreline|outcome must reproduce the grid's
    own cell probabilities (law of total probability), checked on 0-0."""
    grid = dc_grid(1.3, 1.0)
    wdl = np.array(grid_wdl(grid))
    rng = np.random.default_rng(11)
    n = 40_000
    hits = 0
    for _ in range(n):
        outcome = int(rng.choice(3, p=wdl))
        if sim.sample_scoreline(grid, outcome, rng) == (0, 0):
            hits += 1
    p_emp = hits / n
    p_true = float(grid[0, 0])
    se = np.sqrt(p_true * (1 - p_true) / n)
    assert abs(p_emp - p_true) < 4 * se, f"{p_emp:.4f} vs {p_true:.4f}"


def test_group_points_gd_gf_derive_from_sampled_scores():
    """Every simulated group table must be reconstructible from integer
    scorelines: pts consistent with gd sign sums, 4 teams, 6 matches."""
    pred = StubPredictor({"A1": 1700, "B1": 1600, "C1": 1500, "D1": 1400})
    rng = np.random.default_rng(3)
    for _ in range(50):
        table = sim.simulate_group(["A1", "B1", "C1", "D1"], pred, rng)
        assert len(table) == 4
        total_pts = sum(r["pts"] for r in table)
        assert 12 <= total_pts <= 18          # 6 matches x (2 draw .. 3 decisive)
        assert sum(r["gd"] for r in table) == 0
        assert all(r["gf"] >= 0 for r in table)


# ── FIFA 2026 tiebreaks ─────────────────────────────────────────────────────

def _records(vals: dict[str, tuple[int, int, int]]) -> dict[str, dict]:
    return {t: {"team": t, "pts": p, "gd": d, "gf": f}
            for t, (p, d, f) in vals.items()}


def test_overall_gd_ranks_above_h2h():
    """FIFA order: a team ahead on OVERALL GD outranks a team that beat it
    head-to-head (regulations rank overall GD/GF above the H2H block)."""
    rec = _records({"X": (6, 5, 7), "Y": (6, 2, 4), "Z": (3, -3, 2), "W": (0, -4, 1)})
    results = [("Y", "X", 1, 0)]     # Y beat X, but X has the better overall GD
    rng = np.random.default_rng(0)
    order = [r["team"] for r in sim._rank_group(rec, results, rng)]
    assert order[:2] == ["X", "Y"]


def test_h2h_breaks_full_overall_tie():
    rec = _records({"X": (4, 1, 3), "Y": (4, 1, 3), "Z": (4, 1, 3), "W": (0, -3, 0)})
    # Mini-table among X,Y,Z: X beat Y, Y beat Z, X drew Z -> X 4pts, Y 3, Z 1.
    results = [("X", "Y", 1, 0), ("Y", "Z", 1, 0), ("X", "Z", 0, 0)]
    rng = np.random.default_rng(0)
    order = [r["team"] for r in sim._rank_group(rec, results, rng)]
    assert order[:3] == ["X", "Y", "Z"]


def test_reapplication_after_partial_separation():
    """Three-way tie (cycle) where the 3-team mini-table leaves X and Z tied
    and drops Y; the X-Z run must then be re-ranked by THEIR mutual result
    (reapplication) — Z beat X, so Z finishes above X, deterministically,
    for every seed. Without reapplication this would fall to the random
    draw and flicker across seeds."""
    # Full group: X 1-0 Y, Z 2-1 X, Y 1-0 Z (the cycle), and W-losses tuned
    # so X, Y, Z end identical overall: pts 6, GD +1, GF 3.
    results = [("X", "Y", 1, 0), ("Z", "X", 2, 1), ("Y", "Z", 1, 0),
               ("X", "W", 1, 0), ("Y", "W", 2, 1), ("Z", "W", 1, 0)]
    rec = _records({"X": (6, 1, 3), "Y": (6, 1, 3), "Z": (6, 1, 3),
                    "W": (0, -3, 1)})
    # 3-way mini: X (3, 0, 2), Z (3, 0, 2), Y (3, 0, 1) -> runs [X,Z], [Y].
    for seed in range(8):
        order = [r["team"] for r in
                 sim._rank_group(_records({"X": (6, 1, 3), "Y": (6, 1, 3),
                                           "Z": (6, 1, 3), "W": (0, -3, 1)}),
                                 results, np.random.default_rng(seed))]
        assert order == ["Z", "X", "Y", "W"], f"seed {seed}: {order}"


def test_random_fallback_documented_and_reachable():
    """A fully symmetric tie (all draws) must fall to the random draw — and
    different seeds must be able to produce different orders (i.e. no hidden
    deterministic fallthrough to input order)."""
    results = [(a, b, 1, 1) for a, b in combinations("XYZW", 2)]
    orders = {tuple(r["team"] for r in
                    sim._rank_group(_records({t: (3, 0, 3) for t in "XYZW"}),
                                    results, np.random.default_rng(s)))
              for s in range(12)}
    assert len(orders) > 1, "tie fallback is not random (hidden fallthrough?)"


def test_third_place_ranking_no_group_letter_fallthrough():
    """Twelve identical third-place records: the 8 qualifiers must vary with
    the rng, never resolving to the first eight group letters by default."""
    def tables_all_tied():
        tables = {}
        for g, teams in sim.GROUPS.items():
            rows = [{"team": t, "pts": p, "gd": d, "gf": f}
                    for t, (p, d, f) in zip(
                        teams, [(9, 5, 9), (6, 1, 4), (3, -2, 3), (0, -4, 1)])]
            tables[g] = rows
        return tables

    qualified_sets = set()
    for s in range(10):
        r32 = sim.resolve_r32(tables_all_tied(), np.random.default_rng(s))
        thirds = frozenset(t for pair in r32 for t in pair
                           if any(t == rows[2]["team"]
                                  for rows in tables_all_tied().values()))
        qualified_sets.add(thirds)
    assert len(qualified_sets) > 1, (
        "tied third-place ranking always picks the same groups — "
        "group-letter fallthrough is back"
    )


# ── tournament-state conditioning (remaining-tournament forecast) ───────────

def _stub_for_all_teams() -> StubPredictor:
    s = {}
    for teams in sim.GROUPS.values():
        for i, t in enumerate(teams):
            s[t] = 1750 - 150 * i
    return StubPredictor(s)


def test_completed_group_results_enter_as_facts():
    pred = _stub_for_all_teams()
    teams = sim.GROUPS["A"]
    # Weakest team has already beaten everyone 5-0: facts, not simulated.
    facts = [(teams[3], o, 5, 0) for o in teams[:3]]
    for seed in range(5):
        table = sim.simulate_group(teams, pred, np.random.default_rng(seed),
                                   completed=facts)
        assert table[0]["team"] == teams[3]
        assert table[0]["pts"] == 9 and table[0]["gf"] >= 15


def test_fully_conditioned_tournament_is_deterministic():
    """With every group result and every knockout winner recorded, the
    champion is a fact — identical across seeds."""
    pred = _stub_for_all_teams()
    group_results = []
    for teams in sim.GROUPS.values():
        for a, b in combinations(teams, 2):
            ga = 3 if teams.index(a) < teams.index(b) else 0
            gb = 0 if ga else 3
            group_results.append((a, b, ga, gb))

    # Resolve the bracket once to learn the pairings, then record every
    # winner as "first team of the pair".
    state = {"group_results": group_results}
    rng0 = np.random.default_rng(0)
    tables = {g: sim.simulate_group(ts, pred, rng0,
                                    completed=[r for r in group_results
                                               if r[0] in ts and r[1] in ts])
              for g, ts in sim.GROUPS.items()}
    r32 = sim.resolve_r32(tables, np.random.default_rng(0))
    state["r32_pairs"] = r32
    r32w = [a for a, _ in r32]
    r16 = [(r32w[i], r32w[j]) for i, j in sim.R16_PAIRS]
    r16w = [a for a, _ in r16]
    qf = [(r16w[i], r16w[j]) for i, j in sim.QF_PAIRS]
    qfw = [a for a, _ in qf]
    sf = [(qfw[i], qfw[j]) for i, j in sim.SF_PAIRS]
    sfw = [a for a, _ in sf]
    state["ko_winners"] = {
        "R32": dict(enumerate(r32w)), "R16": dict(enumerate(r16w)),
        "QF": dict(enumerate(qfw)), "SF": dict(enumerate(sfw)),
        "Final": {0: sfw[0]},
    }

    champs = {sim.simulate_tournament(pred, np.random.default_rng(s), state=state)
              for s in range(5)}
    assert champs == {sfw[0]}


def test_seed_determinism():
    pred = _stub_for_all_teams()
    a = [sim.simulate_tournament(pred, np.random.default_rng(42)) for _ in range(3)]
    b = [sim.simulate_tournament(pred, np.random.default_rng(42)) for _ in range(3)]
    # Note: each call above uses a FRESH generator, so element-wise equality
    # holds; and two full sequences from one seeded generator must also agree.
    assert a == b
    rng1, rng2 = np.random.default_rng(7), np.random.default_rng(7)
    seq1 = [sim.simulate_tournament(pred, rng1) for _ in range(3)]
    seq2 = [sim.simulate_tournament(pred, rng2) for _ in range(3)]
    assert seq1 == seq2


# ── rho fit recipe ──────────────────────────────────────────────────────────

def test_rho_fit_reproduces_frozen_constant():
    from scorelines import DIXON_COLES_RHO, fit_rho_from_data
    assert abs(fit_rho_from_data() - DIXON_COLES_RHO) < 0.005


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted((k, v) for k, v in globals().items()
                           if k.startswith("test_") and callable(v)):
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    sys.exit(1 if failed else 0)
