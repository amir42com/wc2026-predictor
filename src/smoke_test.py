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
from explain import build_reasons, group_contributions

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

    def scoreline_logging():
        # Pre-match scorelines recorded for the Tracker are a deterministic
        # function of the W/D/L only (never the result): top-5, home-away,
        # sorted high->low, and reproducible from the same probabilities.
        from fetch_results import scorelines_for_proba

        p = tuple(predictor["p"].predict("Argentina", "France"))
        sl = scorelines_for_proba(*p, top_n=5)
        assert len(sl) == 5, f"expected 5 scorelines, got {len(sl)}"
        assert all(isinstance(h, int) and isinstance(a, int) and 0 < pr <= 1
                   for h, a, pr in sl), f"bad scoreline entry in {sl}"
        probs = [pr for *_, pr in sl]
        assert probs == sorted(probs, reverse=True), "scorelines not sorted desc"
        assert scorelines_for_proba(*p, top_n=5) == sl, "scorelines not deterministic"
        print(f"(top-5 home-away, sorted, deterministic; #1 {sl[0][0]}-{sl[0][1]}) ", end="")

    def next_fixture_selection():
        # The homepage default must pick the earliest FUTURE-kickoff fixture,
        # never a finished/already-started one, prefer a live match, and skip
        # fixtures whose team names don't normalize.
        from datetime import datetime, timezone
        from fetch_results import _pick_next_fixture

        now = datetime(2026, 6, 15, 21, 0, 0, tzinfo=timezone.utc)

        def M(h, a, status, utc):
            return {"homeTeam": {"name": h} if h else None,
                    "awayTeam": {"name": a} if a else None,
                    "status": status, "utcDate": utc}

        base = [
            M("Belgium", "Egypt", "FINISHED", "2026-06-14T18:00:00Z"),     # finished
            M("Spain", "Cape Verde", "TIMED", "2026-06-15T20:00:00Z"),     # kicked off, status lag
            M("Saudi Arabia", "Uruguay", "TIMED", "2026-06-15T22:00:00Z"), # earliest FUTURE
            M("Iran", "New Zealand", "TIMED", "2026-06-16T01:00:00Z"),     # later future
        ]
        fx = _pick_next_fixture(base, now)
        assert fx and (fx["home"], fx["away"]) == ("Saudi Arabia", "Uruguay"), \
            f"expected Saudi Arabia/Uruguay, got {fx}"

        # a live match is the current one even if a future TIMED exists
        live = _pick_next_fixture(
            base + [M("France", "Senegal", "IN_PLAY", "2026-06-15T19:00:00Z")], now)
        assert live and live["status"] == "IN_PLAY" and live["home"] == "France", \
            f"live match not prioritised, got {live}"

        # an unmappable earliest fixture is skipped, not fallen back from
        skip = _pick_next_fixture(
            [M(None, None, "TIMED", "2026-06-15T21:30:00Z")] + base, now)
        assert skip and skip["home"] == "Saudi Arabia", f"did not skip unmapped, got {skip}"
        print("(future-earliest, excludes finished/past, live-priority, skips unmapped) ", end="")

    def grouping_reconciliation():
        # The plain-language reasons SUM raw SHAP per group. Grouping must lose
        # nothing (group sums == total SHAP) and the SHAP must stay additive to
        # the model output (base + sum -> softmax == predict_proba). Also verify
        # each reason's favoured side matches the sign of its summed contribution.
        import shap

        model = bundle["b"]["model"]
        fcols = bundle["b"]["feature_cols"]
        ex = shap.TreeExplainer(model)

        def softmax(z):
            z = z - z.max()
            e = np.exp(z)
            return e / e.sum()

        cases = [("Germany", "Cape Verde"), ("Argentina", "France"),
                 ("Haiti", "Brazil"), ("Mexico", "Norway")]
        for home, away in cases:
            X = predictor["p"].feature_X(home, away)
            proba = model.predict_proba(X)[0]
            sv = np.array(ex.shap_values(X))          # (1, n_feats, 3)
            base = np.array(ex.expected_value)        # valid only AFTER the call
            shap_home = dict(zip(fcols, sv[0, :, 0]))

            # 1) grouping loses nothing
            grp_total = sum(group_contributions(shap_home).values())
            assert abs(grp_total - sum(shap_home.values())) < 1e-6, \
                f"group sums != total SHAP for {home}-{away}"

            # 2) SHAP stays additive to the model's W/D/L
            recon = softmax(base + sv[0].sum(axis=0))
            assert np.max(np.abs(recon - proba)) < 1e-4, \
                f"SHAP not additive to model proba for {home}-{away}"

            # 3) each reason's favoured side matches its contribution sign.
            # home_confederation is consumed by make_X, so read it from state.
            hs = predictor["p"]._state.get(home, predictor["p"]._default_state())
            as_ = predictor["p"]._state.get(away, predictor["p"]._default_state())
            ctx = {"home_team": home, "away_team": away,
                   "home_conf": hs["confederation"], "away_conf": as_["confederation"]}
            feat_vals = dict(zip(fcols, X.iloc[0].values))
            for r in build_reasons(shap_home, feat_vals, ctx):
                assert r["home_favoured"] == (r["shap_sum"] > 0), \
                    f"reason side != sign for {home}-{away}: {r['group']}"
        print("(group sums == total SHAP, additive, signs consistent) ", end="")

    steps = [
        ("load model bundle",        load_bundle),
        ("build Predictor",          load_predictor),
        ("predict Argentina-France", predict_fixture),
        ("scoreline consistency",    scoreline_consistency),
        ("pre-match scoreline log",  scoreline_logging),
        ("next-fixture selection",   next_fixture_selection),
        ("SHAP grouping reconcile",  grouping_reconciliation),
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
