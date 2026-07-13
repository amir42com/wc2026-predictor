"""
Phase 2 venue-contract tests.

Covers: the MatchContext inference semantics (neutral symmetry vs host
single-orientation), byte-equivalence of the deprecated predict_neutral_proba
shim (the frozen serving path must be unchanged until the serving rebuild),
the HFA fit recipe reproducing the frozen production constant, the venue
parity of the Elo baseline, the PROTOCOL-INCOMPLETE gate, and the structural
invariants of the provisional WC 2026 schedule table.

Runs standalone (`python tests/test_match_context.py`, exit 0/1) and under
pytest.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import backtest as bt  # noqa: E402
import train  # noqa: E402
from train import MatchContext, predict_match_proba, predict_neutral_proba  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

TEAM_NUM = {"Alpha": 1600.0, "Beta": 1500.0}


def _build_x3(home: str, away: str, neutral: int) -> np.ndarray:
    """Minimal 3-arg build_x: (home rating, away rating, neutral flag)."""
    return np.array([[TEAM_NUM[home], TEAM_NUM[away], float(neutral)]])


class StubModel:
    """Deterministic, orientation- and neutral-sensitive predict_proba."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        h, a, n = np.asarray(X, dtype=float)[0]
        z = np.array([(h - a) / 400.0 + 0.08 * (1.0 - n), 0.1, (a - h) / 400.0])
        e = np.exp(z)
        return (e / e.sum())[None, :]


MODEL = StubModel()


def test_neutral_shim_is_byte_identical():
    """The deprecated shim must reproduce predict_match_proba's neutral path
    exactly — the frozen serving path depends on it."""
    legacy_build_x = lambda h, a: _build_x3(h, a, 1)  # noqa: E731 (serving hardcodes neutral=1)
    old = predict_neutral_proba(MODEL, legacy_build_x, "Alpha", "Beta", 1600.0, 1500.0)
    new = predict_match_proba(MODEL, _build_x3,
                              MatchContext("Alpha", "Beta", neutral=True),
                              1600.0, 1500.0)
    assert np.array_equal(old, new)


def test_neutral_orientation_invariance():
    p_ab = predict_match_proba(MODEL, _build_x3,
                               MatchContext("Alpha", "Beta", neutral=True),
                               1600.0, 1500.0)
    p_ba = predict_match_proba(MODEL, _build_x3,
                               MatchContext("Beta", "Alpha", neutral=True),
                               1500.0, 1600.0)
    assert np.allclose(p_ab, p_ba[[2, 1, 0]], atol=1e-12)


def test_host_path_single_orientation():
    calls: list[tuple] = []

    def spy_build_x(home, away, neutral):
        calls.append((home, away, neutral))
        return _build_x3(home, away, neutral)

    predict_match_proba(MODEL, spy_build_x,
                        MatchContext("Alpha", "Beta", neutral=False),
                        1600.0, 1500.0)
    assert calls == [("Alpha", "Beta", 0)], (
        "host inference must run exactly one orientation, host at home, neutral=0"
    )


def test_hfa_raises_host_win_probability():
    ctx = MatchContext("Alpha", "Beta", neutral=False)
    with_hfa = predict_match_proba(MODEL, _build_x3, ctx, 1600.0, 1500.0)
    without = predict_match_proba(MODEL, _build_x3, ctx, 1600.0, 1500.0, hfa_elo=0.0)
    assert with_hfa[0] > without[0]
    assert with_hfa[2] < without[2]


def test_hfa_fit_reproduces_frozen_constant():
    """The frozen production constant must equal a fresh run of the documented
    recipe on the canonical data (provenance, not tuning)."""
    df = pd.read_csv(ROOT / "data" / "processed" / "features.csv")
    assert abs(train.fit_hfa_elo(df) - train.HFA_ELO) < 0.05


def test_protocol_gate_blocks_backtest_numbers():
    df = pd.DataFrame({"home_elo": [1600.0], "away_elo": [1500.0], "neutral": [1]})
    for fn in (lambda: bt.elo_baseline_proba(df, 0.25),
               lambda: bt.production_blend_prior(df),
               lambda: bt.deployed_model_and_blend(MODEL, df, []),
               lambda: bt.backtest(df)):
        try:
            fn()
        except RuntimeError as exc:
            assert "PROTOCOL-INCOMPLETE" in str(exc)
        else:
            raise AssertionError("gated backtest entry point ran without Phase 4")


def test_baseline_venue_parity():
    """Unit test of the gated math (gate lifted locally, never globally):
    the baseline must apply the same fixed HFA on host rows only."""
    df = pd.DataFrame({
        "home_elo": [1600.0, 1600.0],
        "away_elo": [1500.0, 1500.0],
        "neutral":  [1, 0],          # row 0 neutral, row 1 host
    })
    old_flag = bt.PROTOCOL_COMPLETE
    bt.PROTOCOL_COMPLETE = True
    try:
        p = bt.elo_baseline_proba(df, draw_rate=0.25)
    finally:
        bt.PROTOCOL_COMPLETE = old_flag
    # Neutral row: unchanged classic formula.
    e = 1.0 / (1.0 + 10.0 ** ((1500.0 - 1600.0) / 400.0))
    assert np.isclose(p[0, 0], 0.75 * e)
    # Host row: same formula at elo_home + HFA_ELO.
    e_h = 1.0 / (1.0 + 10.0 ** ((1500.0 - 1600.0 - train.HFA_ELO) / 400.0))
    assert np.isclose(p[1, 0], 0.75 * e_h)
    assert p[1, 0] > p[0, 0]


def test_schedule_table_structure():
    sched = pd.read_csv(ROOT / "data" / "wc2026_schedule.csv",
                        comment="#", dtype=str).fillna("")
    assert len(sched) == 104
    counts = sched["stage"].value_counts()
    assert counts["group"] == 72 and counts["R32"] == 16 and counts["R16"] == 8
    assert counts["QF"] == 4 and counts["SF"] == 2
    assert counts["third_place"] == 1 and counts["final"] == 1

    # PROVISIONAL: nothing may claim verification yet.
    assert (sched["verified"] == "").all()

    # venue_country only ever a host country or blank (= neutral fallback).
    assert set(sched["venue_country"]) <= {"United States", "Mexico", "Canada", ""}

    # The 9 host group rows: host in its own country, marked high confidence.
    host_rows = sched[sched["host_team"] != ""]
    assert len(host_rows) == 9
    assert (host_rows["host_team"] == host_rows["venue_country"]).all()
    assert (host_rows["confidence"] == "high").all()
    assert set(host_rows["group"]) == {"A", "B", "D"}

    # Blank venue_country must coincide with confidence == unknown (no
    # half-filled rows that could silently mis-derive a context).
    blank = sched["venue_country"] == ""
    assert (sched.loc[blank, "confidence"] == "unknown").all()
    assert (sched.loc[~blank, "confidence"] != "unknown").all()


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
