"""
Pre-push smoke test: model loads, prediction is sane, Monte Carlo completes.

Exits 0 on success, 1 on any failure (safe to wire into a pre-push hook).

Usage:
    python src/smoke_test.py
"""

import contextlib
import io
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np

from simulate import GROUPS, Predictor, monte_carlo
from scorelines import grid_wdl, scoreline_grid, top_scorelines

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR    = Path(__file__).resolve().parents[1] / "models"


def check(label: str, fn) -> None:
    print(f"  {label} ... ", end="", flush=True)
    fn()
    print("OK")


def main() -> int:
    bundle = {}
    predictor = {}

    def load_bundle():
        b = joblib.load(MODELS_DIR / "xgb_wc2026.joblib")
        assert {"model", "feature_cols", "label_map"} <= set(b), "bundle keys missing"
        bundle["b"] = b

    def load_predictor():
        predictor["p"] = Predictor(
            PROCESSED_DIR / "features.csv",
            PROCESSED_DIR / "elo_ratings.csv",
            bundle["b"],
        )

    def predict_fixture():
        p = predictor["p"].predict("Argentina", "France")
        assert p.shape == (3,), f"bad shape {p.shape}"
        assert np.all(p >= 0) and np.all(p <= 1), f"probs out of range: {p}"
        assert abs(p.sum() - 1.0) < 1e-9, f"probs sum to {p.sum()}"
        # symmetry: reversed fixture must mirror exactly
        q = predictor["p"].predict("France", "Argentina")
        assert np.allclose(p, q[[2, 1, 0]]), "predict() not symmetric"
        print(f"(A {p[0]*100:.1f}% / D {p[1]*100:.1f}% / F {p[2]*100:.1f}%) ", end="")

    def run_monte_carlo():
        with contextlib.redirect_stdout(io.StringIO()):  # hide progress prints
            wins = monte_carlo(100, predictor["p"], seed=0)
        n = sum(wins.values())
        assert n == 100, f"expected 100 champions, got {n}"
        wc_teams = {t for ts in GROUPS.values() for t in ts}
        assert set(wins) <= wc_teams, f"non-WC champion: {set(wins) - wc_teams}"
        top, cnt = wins.most_common(1)[0]
        print(f"(top: {top} {cnt}/100) ", end="")

    def scoreline_consistency():
        # Scoreline grid must sum to ~1 and roll up to the SAME W/D/L (<=1.5pp).
        cases = [
            tuple(predictor["p"].predict("Argentina", "France")),  # real fixture
            (0.86, 0.10, 0.04),   # lopsided home favourite
            (0.061, 0.162, 0.777),  # away favourite
            (0.34, 0.33, 0.33),   # near coin-flip
        ]
        for p_home, p_draw, p_away in cases:
            grid, _ = scoreline_grid(p_home, p_draw, p_away)
            assert abs(grid.sum() - 1.0) < 1e-6, f"grid sums to {grid.sum()}"
            iw, idr, ial = grid_wdl(grid)
            gap = max(abs(iw - p_home), abs(idr - p_draw), abs(ial - p_away))
            assert gap <= 0.015, f"W/D/L rollup off by {gap*100:.2f}pp for {(p_home,p_draw,p_away)}"
            tops = top_scorelines(p_home, p_draw, p_away, top_n=3)
            assert len(tops) == 3 and all(0 <= pr <= 1 for _, pr in tops)
        print("(rollup <=1.5pp, grid sums to 1) ", end="")

    steps = [
        ("load model bundle",        load_bundle),
        ("build Predictor",          load_predictor),
        ("predict Argentina-France", predict_fixture),
        ("scoreline consistency",    scoreline_consistency),
        ("100-sim Monte Carlo",      run_monte_carlo),
    ]

    print("Smoke test:")
    for label, fn in steps:
        try:
            check(label, fn)
        except Exception:
            print("FAIL")
            traceback.print_exc()
            return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
