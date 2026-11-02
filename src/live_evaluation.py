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

Metrics per system (Blend, Elo baseline): n, correct, accuracy, log-loss,
Brier; McNemar exact + paired bootstrap (10,000 resamples, seed 12345)
reusing paired_tests' implementations. Metrics are reported pooled AND per
tier (prospective, retrospective); the paired tests run only for a scope with
at least MIN_PAIRED_N rows and are marked not applicable below that.

FINAL MODE (--final) refuses unless the evaluation is complete: EXACTLY 104
valid finished fixtures, exactly 104 joined rows (one per valid fixture),
zero validation problems and zero join problems. A refusal names the missing
or unjoined fixture ids. Only a complete evaluation writes
reports/live_evaluation.csv; anything less is an interim read-out printed to
the terminal, never a report artifact.

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
SYSTEM_LABELS = [("blend", "Blend"), ("elo", "Elo baseline")]
TIERS = ["prospective", "retrospective"]
# Below this many rows a McNemar / bootstrap read-out is not informative
# (the 4-row prospective tier); such a scope reports point metrics only.
MIN_PAIRED_N = 30


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


def _scope_metrics(rows: list[dict]) -> dict:
    """Per-system metrics for one scope (pooled or a single tier), plus the
    paired tests when the scope has at least MIN_PAIRED_N rows; below that
    "mcnemar" and "bootstrap" are None (not applicable)."""
    y = np.array([r["y"] for r in rows])
    idx = np.arange(len(y))
    out: dict = {"n": len(rows),
                 "n_prospective": sum(r["tier"] == "prospective" for r in rows),
                 "n_retrospective": sum(r["tier"] == "retrospective" for r in rows)}
    ll: dict[str, np.ndarray] = {}
    correct: dict[str, np.ndarray] = {}
    for s in SYSTEMS:
        p = np.vstack([r[s] for r in rows])
        p = p / p.sum(axis=1, keepdims=True)
        yb = np.zeros_like(p)
        yb[idx, y] = 1.0
        correct[s] = p.argmax(axis=1) == y
        ll[s] = -np.log(np.clip(p[idx, y], 1e-15, None))
        out[s] = {
            "correct": int(correct[s].sum()),
            "accuracy": float(correct[s].mean()),
            "log_loss": float(ll[s].mean()),
            "brier": float(np.sum((p - yb) ** 2, axis=1).mean()),
        }
    if len(rows) >= MIN_PAIRED_N:
        out["mcnemar"] = mcnemar_exact(correct["blend"], correct["elo"])
        out["bootstrap"] = bootstrap_logloss_ci(ll["blend"] - ll["elo"])
    else:
        out["mcnemar"] = out["bootstrap"] = None
    return out


def evaluate(rows: list[dict]) -> dict:
    """Pooled per-system metrics + paired tests over the joined rows, and the
    same metrics for each tier that has rows under "tiers"."""
    if not rows:
        return {"n": 0, "tiers": {}}
    out = _scope_metrics(rows)
    out["tiers"] = {t: _scope_metrics([r for r in rows if r["tier"] == t])
                    for t in TIERS if any(r["tier"] == t for r in rows)}
    return out


def final_mode_check(valid_results: dict[str, dict], rows: list[dict],
                     validation_problems: list[str], join_problems: list[str],
                     record_ids: list[str] | None = None) -> None:
    """FINAL mode gate: the evaluation must be complete. Requires exactly
    TOTAL_FIXTURES valid finished fixtures, exactly TOTAL_FIXTURES joined
    rows covering those fixtures once each, and no validation or join
    problems. record_ids (every fixture_id seen in the raw results) lets a
    refusal name fixtures that were present but not valid."""
    reasons: list[str] = []
    n_valid = len(valid_results)
    if n_valid != TOTAL_FIXTURES:
        reasons.append(f"{n_valid} valid finished fixtures, need exactly "
                       f"{TOTAL_FIXTURES}")
    missing = sorted({i for i in (record_ids or []) if i} - set(valid_results))
    if missing:
        reasons.append(f"not valid/finished: {', '.join(missing)}")
    joined_ids = [r["fixture_id"] for r in rows]
    if len(joined_ids) != TOTAL_FIXTURES:
        reasons.append(f"{len(joined_ids)} joined rows, need exactly "
                       f"{TOTAL_FIXTURES}")
    unjoined = sorted(set(valid_results) - set(joined_ids))
    if unjoined:
        reasons.append(f"unjoined: {', '.join(unjoined)}")
    dup = sorted({i for i in joined_ids if joined_ids.count(i) > 1})
    if dup:
        reasons.append(f"joined more than once: {', '.join(dup)}")
    if validation_problems:
        reasons.append(f"{len(validation_problems)} validation problem(s): "
                       + "; ".join(validation_problems))
    if join_problems:
        reasons.append(f"{len(join_problems)} join problem(s): "
                       + "; ".join(join_problems))
    if reasons:
        raise RuntimeError(
            "final mode refused: " + " | ".join(reasons)
            + ". reports/live_evaluation.csv is only written for the complete "
              "tournament.")


def _report_block(scope: str, m: dict) -> list[dict]:
    """CSV records for one scope: one per system, then the paired tests (or a
    not-applicable marker when the scope is too small)."""
    base = {"scope": scope, "n": m["n"], "n_prospective": m["n_prospective"],
            "n_retrospective": m["n_retrospective"]}
    recs = [{**base, "system": label, **m[s]} for s, label in SYSTEM_LABELS]
    mc, bs = m["mcnemar"], m["bootstrap"]
    if mc is None:
        na = f"not applicable (n={m['n']} < {MIN_PAIRED_N})"
        recs.append({**base, "system": "McNemar exact (Blend vs Elo)", "detail": na})
        recs.append({**base, "system": "Bootstrap logloss diff (Blend-Elo)",
                     "detail": na})
        return recs
    recs.append({**base, "system": "McNemar exact (Blend vs Elo)",
                 "p_value": mc["p_value"],
                 "detail": f"b={mc['b']}, c={mc['c']}, discordant={mc['n_discordant']}"})
    recs.append({**base, "system": "Bootstrap logloss diff (Blend-Elo)",
                 "mean_diff": bs["mean_diff"],
                 "detail": f"95% CI [{bs['ci_low']:.6f}, {bs['ci_high']:.6f}], "
                           f"{bs['n_boot']} resamples, seed {bs['seed']}"})
    return recs


def write_final_report(rows: list[dict], metrics: dict, path: Path = OUT_PATH) -> None:
    """Pooled block first, then one block per tier that has rows."""
    recs = _report_block("pooled", metrics)
    for t in TIERS:
        if t in metrics["tiers"]:
            recs += _report_block(t, metrics["tiers"][t])
    cols = ["scope", "system", "n", "n_prospective", "n_retrospective",
            "correct", "accuracy", "log_loss", "brier", "p_value", "mean_diff",
            "detail"]
    pd.DataFrame(recs, columns=cols).to_csv(path, index=False)
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
    """Stable fixture ID for a cache record: knockout fixtures in the ledger
    registry map to their official M-number BY KICKOFF DATE — every registry
    fixture (M101 2026-07-14, M102 2026-07-15, M103 2026-07-18, M104
    2026-07-19) sits alone on its date, and these IDs must match the ledger's
    exactly or finished semifinals would silently fall to the date-team
    fallback and never join their prospective entries (post-push review bug).
    Everything else gets the date-teams slug used by the retrospective tier."""
    from prediction_ledger import FIXTURES
    for fid, info in sorted(FIXTURES.items()):
        if m["date"] == info["kickoff_date"]:
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

    records = load_results_from_cache()
    valid, problems = validate_results(records)
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
    _print_scope("pooled", metrics)
    for t in TIERS:
        if t in metrics["tiers"]:
            _print_scope(t, metrics["tiers"][t])

    if args.final:
        final_mode_check(valid, rows, problems, join_problems,
                         record_ids=[r.get("fixture_id") for r in records])
        write_final_report(rows, metrics)


def _print_scope(scope: str, m: dict) -> None:
    print(f"[{scope}] n={m['n']}")
    for s, label in SYSTEM_LABELS:
        ms = m[s]
        print(f"  {label:<13} {ms['correct']}/{m['n']} correct  "
              f"acc {ms['accuracy']*100:.2f}%  ll {ms['log_loss']:.4f}  "
              f"brier {ms['brier']:.4f}")
    mc, bs = m["mcnemar"], m["bootstrap"]
    if mc is None:
        print(f"  paired tests not applicable (n={m['n']} < {MIN_PAIRED_N})")
        return
    print(f"  McNemar p={mc['p_value']:.4f} (b={mc['b']}, c={mc['c']}); "
          f"bootstrap LL diff {bs['mean_diff']:+.4f} "
          f"[{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
