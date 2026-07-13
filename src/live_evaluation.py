"""
Two-tier live evaluation of WC 2026 predictions (remediation Phase 6b).

Tier 1 — PROSPECTIVE: entries read from reports/prediction_ledger.csv
(generated before kickoff, externally timestamped by the branch push).
Tier 2 — RETROSPECTIVE: finished fixtures with no ledger entry are recomputed
from the frozen bundle at full precision and labeled retrospective=true.
The tiers are never pooled silently: every evaluated row carries its tier.

Results are a SEPARATE table joined by fixture_id. Scoring target: the
after-extra-time W/D/L, with penalty shootouts scored as DRAWS (the martj42 /
training-target convention; the tracker cache already strips shootout goals —
validation re-checks that a PENALTY_SHOOTOUT record carries a level score).

Metrics per system (Blend, Elo baseline): n, correct, accuracy, pooled
log-loss, pooled Brier; McNemar exact + paired bootstrap (10,000 resamples,
seed 12345) reusing paired_tests' implementations.

FINAL MODE (--final) refuses unless there are EXACTLY 104 valid finished
fixtures; only then is reports/live_evaluation.csv written. Anything less is
an interim read-out printed to the terminal, never a report artifact.

Usage:
    python src/live_evaluation.py            # interim read-out
    python src/live_evaluation.py --final    # 104-match report or refusal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_tests import bootstrap_logloss_ci, mcnemar_exact  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "reports" / "prediction_ledger.csv"
CACHE_PATH = ROOT / "data" / "wc2026_results.json"
OUT_PATH = ROOT / "reports" / "live_evaluation.csv"

TOTAL_FIXTURES = 104
SYSTEMS = ["blend", "elo"]


# ── result validation (pure, synthetic-testable) ────────────────────────────

def validate_results(records: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """
    Validate raw result records into {fixture_id: record}. A valid record has
    a fixture_id, both teams, integer scores, and — when duration is
    PENALTY_SHOOTOUT — a LEVEL after-ET score (the shootout-stripped
    convention; an unequal score there means the strip failed upstream).
    Duplicated fixture_ids invalidate every copy (ambiguous truth).
    Returns (valid, problems).
    """
    problems: list[str] = []
    seen: dict[str, dict] = {}
    dupes: set[str] = set()

    for r in records:
        fid = r.get("fixture_id")
        if not fid:
            problems.append("record without fixture_id skipped")
            continue
        if fid in seen:
            dupes.add(fid)
            problems.append(f"{fid}: duplicate result records — all copies dropped")
            continue
        seen[fid] = r

    valid: dict[str, dict] = {}
    for fid, r in seen.items():
        if fid in dupes:
            continue
        hs, as_ = r.get("home_score"), r.get("away_score")
        if r.get("home_team") in (None, "") or r.get("away_team") in (None, ""):
            problems.append(f"{fid}: missing team name — dropped")
            continue
        if hs is None or as_ is None:
            problems.append(f"{fid}: unfinished/missing score — dropped")
            continue
        if not (float(hs).is_integer() and float(as_).is_integer()):
            problems.append(f"{fid}: non-integer score — dropped")
            continue
        if r.get("duration") == "PENALTY_SHOOTOUT" and int(hs) != int(as_):
            problems.append(f"{fid}: PENALTY_SHOOTOUT with unlevel after-ET "
                            "score (shootout strip failed upstream) — dropped")
            continue
        valid[fid] = r
    return valid, problems


def outcome_vs_orientation(result: dict, team_a: str, team_b: str) -> int | None:
    """After-ET W/D/L from team_a's perspective (0 a-win / 1 draw / 2 b-win),
    or None when the result's teams don't match the prediction's."""
    hs, as_ = int(result["home_score"]), int(result["away_score"])
    if result["home_team"] == team_a and result["away_team"] == team_b:
        ga, gb = hs, as_
    elif result["home_team"] == team_b and result["away_team"] == team_a:
        ga, gb = as_, hs
    else:
        return None
    return 0 if ga > gb else (1 if ga == gb else 2)


# ── joining + metrics (pure) ────────────────────────────────────────────────

def join_predictions(predictions: list[dict],
                     valid_results: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """Join prediction entries to validated results by fixture_id. Each joined
    row: fixture_id, tier, y, blend/elo probability vectors (a, draw, b)."""
    rows, problems = [], []
    for p in predictions:
        fid = p["fixture_id"]
        r = valid_results.get(fid)
        if r is None:
            continue   # not finished / not valid: simply not evaluable yet
        y = outcome_vs_orientation(r, p["team_a"], p["team_b"])
        if y is None:
            problems.append(f"{fid}: result teams {r['home_team']}/{r['away_team']} "
                            f"do not match prediction {p['team_a']}/{p['team_b']}")
            continue
        rows.append({
            "fixture_id": fid,
            "tier": "retrospective" if _truthy(p.get("retrospective")) else "prospective",
            "y": y,
            "blend": np.array([float(p["blend_a"]), float(p["blend_draw"]),
                               float(p["blend_b"])]),
            "elo": np.array([float(p["elo_a"]), float(p["elo_draw"]),
                             float(p["elo_b"])]),
        })
    return rows, problems


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1")


def evaluate(rows: list[dict]) -> dict:
    """Per-system metrics + paired tests over the joined rows."""
    if not rows:
        return {"n": 0}
    y = np.array([r["y"] for r in rows])
    out: dict = {"n": len(rows),
                 "n_prospective": sum(r["tier"] == "prospective" for r in rows),
                 "n_retrospective": sum(r["tier"] == "retrospective" for r in rows)}
    proba = {s: np.vstack([r[s] for r in rows]) for s in SYSTEMS}
    for s in SYSTEMS:
        p = proba[s] / proba[s].sum(axis=1, keepdims=True)
        pred = p.argmax(axis=1)
        p_true = p[np.arange(len(y)), y]
        yb = np.zeros_like(p)
        yb[np.arange(len(y)), y] = 1.0
        out[s] = {
            "correct": int((pred == y).sum()),
            "accuracy": float((pred == y).mean()),
            "log_loss": float(-np.log(np.clip(p_true, 1e-15, None)).mean()),
            "brier": float(np.sum((p - yb) ** 2, axis=1).mean()),
        }
    correct = {s: (proba[s].argmax(axis=1) == y) for s in SYSTEMS}
    out["mcnemar"] = mcnemar_exact(correct["blend"], correct["elo"])
    ll = {s: -np.log(np.clip(
            (proba[s] / proba[s].sum(axis=1, keepdims=True))[np.arange(len(y)), y],
            1e-15, None)) for s in SYSTEMS}
    out["bootstrap"] = bootstrap_logloss_ci(ll["blend"] - ll["elo"])
    return out


def final_mode_check(valid_results: dict[str, dict]) -> None:
    """FINAL mode gate: exactly TOTAL_FIXTURES valid finished fixtures."""
    n = len(valid_results)
    if n != TOTAL_FIXTURES:
        raise RuntimeError(
            f"final mode refused: {n} valid finished fixtures, need exactly "
            f"{TOTAL_FIXTURES}. reports/live_evaluation.csv is only written "
            "for the complete tournament.")


def write_final_report(rows: list[dict], metrics: dict, path: Path = OUT_PATH) -> None:
    recs = []
    for s, label in [("blend", "Blend"), ("elo", "Elo baseline")]:
        m = metrics[s]
        recs.append({"system": label, "n": metrics["n"],
                     "n_prospective": metrics["n_prospective"],
                     "n_retrospective": metrics["n_retrospective"], **m})
    mc, bs = metrics["mcnemar"], metrics["bootstrap"]
    recs.append({"system": "McNemar exact (Blend vs Elo)", "n": metrics["n"],
                 "p_value": mc["p_value"],
                 "detail": f"b={mc['b']}, c={mc['c']}, discordant={mc['n_discordant']}"})
    recs.append({"system": "Bootstrap logloss diff (Blend-Elo)", "n": metrics["n"],
                 "mean_diff": bs["mean_diff"],
                 "detail": f"95% CI [{bs['ci_low']:.6f}, {bs['ci_high']:.6f}], "
                           f"{bs['n_boot']} resamples, seed {bs['seed']}"})
    pd.DataFrame(recs).to_csv(path, index=False)
    print(f"wrote {path}")


# ── data loading (live paths; NOT exercised by the synthetic tests) ─────────

def load_ledger_predictions() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    return pd.read_csv(LEDGER_PATH, dtype=str).to_dict("records")


def load_results_from_cache() -> list[dict]:
    """Map tracker cache records to result records with stable fixture_ids:
    knockout fixtures get their M-number when derivable from the schedule
    date-windows; every other fixture gets the date-teams slug used by the
    retrospective tier."""
    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    matches = data["matches"] if isinstance(data, dict) else data
    out = []
    for m in matches:
        out.append({
            "fixture_id": fixture_slug(m),
            "home_team": m["home_team"], "away_team": m["away_team"],
            "home_score": m.get("home_score"), "away_score": m.get("away_score"),
            "duration": m.get("duration"),
        })
    return out


def fixture_slug(m: dict) -> str:
    from prediction_ledger import FIXTURES
    for fid, info in FIXTURES.items():
        if m["date"] == info["kickoff_date"]:
            # date-unique fixtures only (M103/M104); SF days could collide if
            # schedules shift — the slug fallback below stays unambiguous.
            if fid in ("M103", "M104"):
                return f"WC2026-{fid}"
    return f"WC2026-{m['date']}-{m['home_team']}-v-{m['away_team']}"


def retrospective_predictions(valid_results: dict[str, dict],
                              ledgered: set[str]) -> list[dict]:
    """Tier 2: recompute frozen-bundle predictions for finished fixtures that
    were never ledgered, at full precision, labeled retrospective=true."""
    from prediction_ledger import load_serving, predict_pair, venue_context

    predictor, bundle, drb = load_serving()
    out = []
    for fid, r in sorted(valid_results.items()):
        if fid in ledgered:
            continue
        a, b = r["home_team"], r["away_team"]
        ctx = venue_context(fid.replace("WC2026-", ""), a, b)
        blend, elo, home, away = predict_pair(predictor, bundle, a, b, ctx, drb)
        out.append({
            "fixture_id": fid, "team_a": home, "team_b": away,
            "blend_a": repr(float(blend[0])), "blend_draw": repr(float(blend[1])),
            "blend_b": repr(float(blend[2])),
            "elo_a": repr(float(elo[0])), "elo_draw": repr(float(elo[1])),
            "elo_b": repr(float(elo[2])),
            "retrospective": True,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true",
                        help="write reports/live_evaluation.csv (exactly "
                             f"{TOTAL_FIXTURES} valid finished fixtures required)")
    args = parser.parse_args()

    valid, problems = validate_results(load_results_from_cache())
    for p in problems:
        print(f"PROBLEM: {p}")

    ledger = load_ledger_predictions()
    ledgered_ids = {p["fixture_id"] for p in ledger}
    predictions = ledger + retrospective_predictions(valid, ledgered_ids)

    rows, join_problems = join_predictions(predictions, valid)
    for p in join_problems:
        print(f"PROBLEM: {p}")

    metrics = evaluate(rows)
    if metrics["n"] == 0:
        print("no evaluable fixtures")
        return
    print(f"\nEvaluated {metrics['n']} fixtures "
          f"({metrics['n_prospective']} prospective / "
          f"{metrics['n_retrospective']} retrospective)")
    for s, label in [("blend", "Blend"), ("elo", "Elo baseline")]:
        m = metrics[s]
        print(f"  {label:<13} {m['correct']}/{metrics['n']} correct  "
              f"acc {m['accuracy']*100:.2f}%  ll {m['log_loss']:.4f}  "
              f"brier {m['brier']:.4f}")
    mc, bs = metrics["mcnemar"], metrics["bootstrap"]
    print(f"  McNemar p={mc['p_value']:.4f} (b={mc['b']}, c={mc['c']}); "
          f"bootstrap LL diff {bs['mean_diff']:+.4f} "
          f"[{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]")

    if args.final:
        final_mode_check(valid)
        write_final_report(rows, metrics)


if __name__ == "__main__":
    main()
