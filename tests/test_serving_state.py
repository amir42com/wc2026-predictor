"""
Task 1 regression: the live serving state must include each team's FINAL
pre-tournament match, not the one-behind value from its last pre-match feature
row. Runs standalone (`python tests/test_serving_state.py`, exit 0/1) and under
pytest.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import features as feat      # noqa: E402
from simulate import Predictor  # noqa: E402


def _synthetic_history() -> pd.DataFrame:
    """Alpha plays 10 matches: wins the first 9, LOSES the 10th (2026-01-10).

    Pre-match-10 form (the old last-feature-row state) = 9 wins / 9 = win_rate_10
    1.0. Post-match-10 form (the correct served state) includes the loss =
    9/10 = 0.9, and win_rate_5 over the last five = 4 wins/5 = 0.8.
    """
    rows = []
    for i in range(1, 11):
        alpha_wins = i <= 9          # match 10 is a loss
        rows.append({
            "date": pd.Timestamp(f"2026-01-{i:02d}"),
            "home_team": "Alpha",
            "away_team": f"Opp{i}",
            "home_score": 2 if alpha_wins else 0,
            "away_score": 0 if alpha_wins else 2,
            "tournament": "Friendly",
            "neutral": 0,
        })
    return pd.DataFrame(rows)


def test_team_state_includes_final_match():
    df = _synthetic_history()
    _, _, team_state = feat.build_features(df)
    alpha = team_state.set_index("team").loc["Alpha"]
    # Includes the final (10th) match — NOT the stale 1.0 / 1.0 one-behind value.
    assert alpha["win_rate_10"] == 0.9, alpha["win_rate_10"]
    assert alpha["win_rate_5"] == 0.8, alpha["win_rate_5"]
    assert alpha["win_rate_10"] != 1.0, "served win_rate_10 is one match stale!"


def test_predictor_serves_post_final_match_state():
    """The same through the full serving path (team_state.csv -> Predictor)."""
    df = _synthetic_history()
    features, elo_df, team_state = feat.build_features(df)
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        features.to_csv(dd / "features.csv", index=False)
        elo_df.to_csv(dd / "elo_ratings.csv", index=False)
        team_state.to_csv(dd / "team_state.csv", index=False)
        # cutoff=None: synthetic dates are arbitrary; we only test state loading.
        bundle = {"model": None, "feature_cols": []}
        p = Predictor(dd / "features.csv", dd / "elo_ratings.csv", bundle, cutoff=None)
        assert p._state["Alpha"]["win_rate_10"] == 0.9, p._state["Alpha"]
        assert p._state["Alpha"]["win_rate_5"] == 0.8, p._state["Alpha"]


def main() -> int:
    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    print(f"Serving-state tests ({len(tests)} cases):")
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
