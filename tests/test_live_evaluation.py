"""
Phase 6b: live_evaluation logic on SYNTHETIC fixtures only — home win / draw /
away win / extra-time / shootout-stripped / missing / duplicate / incomplete /
exactly-104 final-mode gate (complete joins, no problems), per-tier metrics.
No model bundle loaded, no WC 2026 results read.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import live_evaluation as le  # noqa: E402


def _res(fid, home="Alpha", away="Beta", hs=1, as_=0, duration="REGULAR"):
    return {"fixture_id": fid, "home_team": home, "away_team": away,
            "home_score": hs, "away_score": as_, "duration": duration}


def _pred(fid, a="Alpha", b="Beta", blend=(0.5, 0.3, 0.2), elo=(0.4, 0.25, 0.35),
          retro=False):
    return {"fixture_id": fid, "team_a": a, "team_b": b,
            "blend_a": blend[0], "blend_draw": blend[1], "blend_b": blend[2],
            "elo_a": elo[0], "elo_draw": elo[1], "elo_b": elo[2],
            "retrospective": retro}


# ── validation ──────────────────────────────────────────────────────────────

def test_validation_cases():
    records = [
        _res("F1", hs=2, as_=1),                                  # home win
        _res("F2", hs=1, as_=1),                                  # draw
        _res("F3", hs=0, as_=3),                                  # away win
        _res("F4", hs=2, as_=1, duration="EXTRA_TIME"),           # after-ET win
        _res("F5", hs=1, as_=1, duration="PENALTY_SHOOTOUT"),     # stripped draw
        _res("F6", hs=None, as_=None),                            # unfinished
        _res("F7", hs=3, as_=0, duration="PENALTY_SHOOTOUT"),     # bad strip
        _res("F8", home=""),                                      # incomplete
        _res("F9"), _res("F9"),                                   # duplicate
        {"home_team": "X", "away_team": "Y",
         "home_score": 1, "away_score": 0},                       # no id
    ]
    valid, problems = le.validate_results(records)
    assert set(valid) == {"F1", "F2", "F3", "F4", "F5"}
    joined_problems = "\n".join(problems)
    assert "F6" in joined_problems and "F7" in joined_problems
    assert "F8" in joined_problems and "F9" in joined_problems
    assert "without fixture_id" in joined_problems


def test_shootout_scored_as_draw_and_et_as_decisive():
    valid, _ = le.validate_results([
        _res("S1", hs=2, as_=2, duration="PENALTY_SHOOTOUT"),
        _res("S2", hs=2, as_=1, duration="EXTRA_TIME"),
    ])
    assert le.outcome_vs_orientation(valid["S1"], "Alpha", "Beta") == 1
    assert le.outcome_vs_orientation(valid["S2"], "Alpha", "Beta") == 0


def test_orientation_mapping():
    r = _res("F1", home="Beta", away="Alpha", hs=0, as_=2)   # Alpha away, wins
    assert le.outcome_vs_orientation(r, "Alpha", "Beta") == 0
    assert le.outcome_vs_orientation(r, "Beta", "Alpha") == 2
    assert le.outcome_vs_orientation(r, "Alpha", "Gamma") is None


# ── join + metrics ──────────────────────────────────────────────────────────

def test_join_and_metrics():
    valid, _ = le.validate_results([
        _res("F1", hs=2, as_=1),                     # Alpha (team_a) wins -> y=0
        _res("F2", hs=0, as_=0),                     # draw -> y=1
        _res("F3", home="Beta", away="Alpha", hs=1, as_=0),  # Beta wins -> y=2
    ])
    preds = [
        _pred("F1", blend=(0.6, 0.25, 0.15), elo=(0.3, 0.3, 0.4)),
        _pred("F2", blend=(0.2, 0.5, 0.3), elo=(0.5, 0.2, 0.3), retro=True),
        _pred("F3", blend=(0.1, 0.2, 0.7), elo=(0.6, 0.2, 0.2)),
        _pred("F4"),                                  # unplayed: not evaluable
    ]
    rows, problems = le.join_predictions(preds, valid)
    assert not problems and len(rows) == 3
    assert [r["y"] for r in rows] == [0, 1, 2]
    assert [r["tier"] for r in rows] == ["prospective", "retrospective", "prospective"]

    m = le.evaluate(rows)
    assert m["n"] == 3 and m["n_prospective"] == 2 and m["n_retrospective"] == 1
    assert m["blend"]["correct"] == 3 and m["blend"]["accuracy"] == 1.0
    assert m["elo"]["correct"] == 0
    # hand-check blend pooled log-loss
    expected_ll = -np.mean(np.log([0.6, 0.5, 0.7]))
    assert abs(m["blend"]["log_loss"] - expected_ll) < 1e-12
    # 3 rows < MIN_PAIRED_N: paired tests are not applicable at this size
    # (test_per_tier_metrics_hand_checked covers them with enough rows).
    assert m["mcnemar"] is None and m["bootstrap"] is None
    assert set(m["tiers"]) == {"prospective", "retrospective"}


def test_join_flags_team_mismatch():
    valid, _ = le.validate_results([_res("F1", home="Gamma", away="Delta")])
    rows, problems = le.join_predictions([_pred("F1")], valid)
    assert rows == [] and len(problems) == 1 and "do not match" in problems[0]


# ── fixture-ID mapping (post-push review fix) ───────────────────────────────

def test_fixture_slug_maps_all_registry_dates():
    """Every ledger-registry fixture maps to its official M-number by kickoff
    date — M101/M102 included (the reviewed bug mapped only M103/M104)."""
    for date, fid in [("2026-07-14", "M101"), ("2026-07-15", "M102"),
                      ("2026-07-18", "M103"), ("2026-07-19", "M104")]:
        m = {"date": date, "home_team": "X", "away_team": "Y"}
        assert le.fixture_slug(m) == f"WC2026-{fid}", (date, fid)
    # Non-registry dates keep the retrospective date-teams slug.
    m = {"date": "2026-06-15", "home_team": "X", "away_team": "Y"}
    assert le.fixture_slug(m) == "WC2026-2026-06-15-X-v-Y"


def test_semifinal_results_join_ledger_entries_as_prospective():
    """Both finished semifinals must join their prospective ledger entries by
    ID — including M102, where the ledger stores Argentina/England but the
    result arrives as England/Argentina (reversed orientation)."""
    ledger = [
        _pred("WC2026-M101", a="France", b="Spain",
              blend=(0.4, 0.3, 0.3), elo=(0.35, 0.25, 0.40)),
        _pred("WC2026-M102", a="Argentina", b="England",
              blend=(0.5, 0.3, 0.2), elo=(0.45, 0.25, 0.30)),
    ]
    cache_records = [
        {"date": "2026-07-14", "home_team": "France", "away_team": "Spain",
         "home_score": 0, "away_score": 1, "duration": "REGULAR"},
        {"date": "2026-07-15", "home_team": "England", "away_team": "Argentina",
         "home_score": 1, "away_score": 2, "duration": "EXTRA_TIME"},
    ]
    results = [{"fixture_id": le.fixture_slug(m), **m} for m in cache_records]
    assert [r["fixture_id"] for r in results] == ["WC2026-M101", "WC2026-M102"]

    valid, problems = le.validate_results(results)
    assert not problems and set(valid) == {"WC2026-M101", "WC2026-M102"}

    rows, join_problems = le.join_predictions(ledger, valid)
    assert not join_problems, join_problems
    assert len(rows) == 2, "a semifinal failed to join its ledger entry"
    assert all(r["tier"] == "prospective" for r in rows)
    by_id = {r["fixture_id"]: r for r in rows}
    assert by_id["WC2026-M101"]["y"] == 2      # Spain (team_b) won
    assert by_id["WC2026-M102"]["y"] == 0      # Argentina (team_a) won, reversed result

    m = le.evaluate(rows)
    assert m["n"] == 2 and m["n_prospective"] == 2 and m["n_retrospective"] == 0


# ── final mode ──────────────────────────────────────────────────────────────

def _complete(n):
    """n synthetic finished fixtures, each with a matching prediction."""
    records = [_res(f"G{i}") for i in range(n)]
    valid, vprob = le.validate_results(records)
    rows, jprob = le.join_predictions([_pred(f"G{i}") for i in range(n)], valid)
    return records, valid, vprob, rows, jprob


def _refusal(valid, rows, vprob, jprob, records):
    try:
        le.final_mode_check(valid, rows, vprob, jprob,
                            record_ids=[r.get("fixture_id") for r in records])
    except RuntimeError as exc:
        assert "final mode refused" in str(exc)
        return str(exc)
    raise AssertionError("final mode accepted an incomplete evaluation")


def test_final_mode_requires_exactly_104():
    records, valid, vprob, rows, jprob = _complete(103)
    assert "103 valid finished fixtures" in _refusal(valid, rows, vprob, jprob, records)

    records, valid, vprob, rows, jprob = _complete(104)
    le.final_mode_check(valid, rows, vprob, jprob,
                        record_ids=[r["fixture_id"] for r in records])  # must not raise

    records, valid, vprob, rows, jprob = _complete(105)
    _refusal(valid, rows, vprob, jprob, records)


def test_final_mode_refuses_join_problem_and_names_fixture():
    """104 valid results, but one prediction's teams do not match its result:
    103 joined rows plus a join problem must refuse, naming the fixture."""
    records = [_res(f"G{i}") for i in range(104)]
    valid, vprob = le.validate_results(records)
    preds = [_pred(f"G{i}") for i in range(103)] + [_pred("G103", a="Gamma")]
    rows, jprob = le.join_predictions(preds, valid)
    assert len(valid) == 104 and len(rows) == 103 and len(jprob) == 1
    msg = _refusal(valid, rows, vprob, jprob, records)
    assert "103 joined rows" in msg and "unjoined: G103" in msg
    assert "1 join problem(s)" in msg


def test_final_mode_refuses_missing_prediction_and_names_fixture():
    """A valid fixture with no prediction at all (no join problem raised)
    still leaves the evaluation incomplete and must be named."""
    records = [_res(f"G{i}") for i in range(104)]
    valid, vprob = le.validate_results(records)
    rows, jprob = le.join_predictions([_pred(f"G{i}") for i in range(1, 104)], valid)
    assert not jprob and len(rows) == 103
    assert "unjoined: G0" in _refusal(valid, rows, vprob, jprob, records)


def test_final_mode_refuses_validation_problem_and_names_fixture():
    """104 valid fixtures plus one invalid record (bad shootout strip) is not
    complete: the validation problem refuses and the fixture is named."""
    records = [_res(f"G{i}") for i in range(104)] + [
        _res("BAD", hs=3, as_=0, duration="PENALTY_SHOOTOUT")]
    valid, vprob = le.validate_results(records)
    rows, jprob = le.join_predictions([_pred(f"G{i}") for i in range(104)], valid)
    assert len(valid) == 104 and len(rows) == 104 and not jprob
    msg = _refusal(valid, rows, vprob, jprob, records)
    assert "not valid/finished: BAD" in msg and "1 validation problem(s)" in msg


# ── per-tier metrics ────────────────────────────────────────────────────────

def _tiered_rows():
    """4 prospective rows (hand-checkable) + 30 identical retrospective rows."""
    pro = [  # (fid, result home/away score, blend, elo)
        ("P1", (2, 1), (0.6, 0.25, 0.15), (0.3, 0.3, 0.4)),   # y=0
        ("P2", (1, 1), (0.2, 0.5, 0.3), (0.5, 0.2, 0.3)),     # y=1
        ("P3", (0, 2), (0.1, 0.2, 0.7), (0.6, 0.2, 0.2)),     # y=2
        ("P4", (0, 1), (0.5, 0.3, 0.2), (0.2, 0.3, 0.5)),     # y=2
    ]
    results = [_res(f, hs=s[0], as_=s[1]) for f, s, _, _ in pro]
    preds = [_pred(f, blend=b, elo=e) for f, _, b, e in pro]
    results += [_res(f"R{i}", hs=1, as_=0) for i in range(30)]           # y=0
    preds += [_pred(f"R{i}", blend=(0.5, 0.3, 0.2), elo=(0.4, 0.25, 0.35),
                    retro=True) for i in range(30)]
    valid, vprob = le.validate_results(results)
    rows, jprob = le.join_predictions(preds, valid)
    assert not vprob and not jprob and len(rows) == 34
    return rows


def test_per_tier_metrics_hand_checked():
    m = le.evaluate(_tiered_rows())
    assert m["n"] == 34 and m["n_prospective"] == 4 and m["n_retrospective"] == 30

    pro = m["tiers"]["prospective"]
    assert pro["n"] == 4 and pro["n_prospective"] == 4 and pro["n_retrospective"] == 0
    # Blend picks: P1 a, P2 draw, P3 b, P4 a (actual b) -> 3/4.
    assert pro["blend"]["correct"] == 3 and pro["blend"]["accuracy"] == 0.75
    assert abs(pro["blend"]["log_loss"]
               - -np.mean(np.log([0.6, 0.5, 0.7, 0.2]))) < 1e-12
    # Brier per row: .245, .38, .14, .98
    assert abs(pro["blend"]["brier"] - 0.43625) < 1e-12
    # Elo picks: P1 b, P2 a, P3 a, P4 b -> only P4 right.
    assert pro["elo"]["correct"] == 1
    assert abs(pro["elo"]["log_loss"]
               - -np.mean(np.log([0.3, 0.2, 0.2, 0.5]))) < 1e-12
    # Brier per row: .74, .98, 1.04, .38
    assert abs(pro["elo"]["brier"] - 0.785) < 1e-12
    assert pro["mcnemar"] is None and pro["bootstrap"] is None   # n=4 < 30

    ret = m["tiers"]["retrospective"]
    assert ret["n"] == 30 and ret["blend"]["correct"] == 30 and ret["elo"]["correct"] == 30
    assert abs(ret["blend"]["log_loss"] - -np.log(0.5)) < 1e-12
    assert abs(ret["elo"]["log_loss"] - -np.log(0.4)) < 1e-12
    assert abs(ret["blend"]["brier"] - (0.25 + 0.09 + 0.04)) < 1e-12
    assert abs(ret["elo"]["brier"] - (0.36 + 0.0625 + 0.1225)) < 1e-12
    assert ret["mcnemar"]["b"] == 0 and ret["mcnemar"]["c"] == 0
    # every per-row LL diff is log(0.4) - log(0.5) = log(0.8)
    assert abs(ret["bootstrap"]["mean_diff"] - np.log(0.8)) < 1e-12

    # pooled = the 34 rows together
    assert m["blend"]["correct"] == 33 and m["elo"]["correct"] == 31
    assert m["mcnemar"]["b"] == 3 and m["mcnemar"]["c"] == 1


def test_tier_absent_when_no_rows():
    rows = [r for r in _tiered_rows() if r["tier"] == "retrospective"]
    m = le.evaluate(rows)
    assert set(m["tiers"]) == {"retrospective"}


def test_final_report_has_pooled_and_tier_blocks():
    import pandas as pd
    rows = _tiered_rows()
    metrics = le.evaluate(rows)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "live_evaluation.csv"
        le.write_final_report(rows, metrics, path=out)
        df = pd.read_csv(out)
    assert list(df["scope"].unique()) == ["pooled", "prospective", "retrospective"]
    assert (df.groupby("scope").size() == 4).all()
    pro = df[df["scope"] == "prospective"].set_index("system")
    assert pro.loc["Blend", "correct"] == 3 and pro.loc["Blend", "n"] == 4
    assert "not applicable" in pro.loc["McNemar exact (Blend vs Elo)", "detail"]
    assert pd.isna(pro.loc["McNemar exact (Blend vs Elo)", "p_value"])
    ret = df[df["scope"] == "retrospective"].set_index("system")
    assert ret.loc["McNemar exact (Blend vs Elo)", "p_value"] == 1.0
    assert abs(ret.loc["Bootstrap logloss diff (Blend-Elo)", "mean_diff"]
               - np.log(0.8)) < 1e-9


def test_final_report_written_only_in_final_mode():
    valid, _ = le.validate_results([_res(f"G{i}") for i in range(104)])
    preds = [_pred(f"G{i}") for i in range(104)]
    rows, _ = le.join_predictions(preds, valid)
    metrics = le.evaluate(rows)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "live_evaluation.csv"
        le.write_final_report(rows, metrics, path=out)
        assert out.exists()
        text = out.read_text()
        assert "Blend" in text and "Elo baseline" in text
        assert "McNemar" in text and "Bootstrap" in text


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
