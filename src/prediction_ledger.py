"""
Prospective prediction ledger (remediation Phase 6a).

Records full-precision Blend AND Elo-baseline probabilities for upcoming WC
2026 fixtures BEFORE kickoff, with complete provenance (bundle/data/code
hashes, every frozen constant, venue context, timestamps). The ledger is
APPEND-ONLY FROM BIRTH: existing entries are never rewritten — regeneration
appends missing fixtures only, and the file is only ever opened in append
mode after creation (tests/test_ledger.py enforces byte-stability).

Fixture identity: official match numbers (M101/M102 semifinals, M103 third
place, M104 final) — the remaining fixtures at ledger birth. Pairings are
derived from the tracker's cached results (data/wc2026_results.json) by
schedule date-windows; if the cache is stale or the derivation is ambiguous
the script says so and accepts explicit pairings:

    python src/prediction_ledger.py --refresh          # routine tracker
                                                       # refresh first, then
                                                       # ledger what is now
                                                       # determined
    python src/prediction_ledger.py                    # cache only
    python src/prediction_ledger.py --fixture M101 "Team A" "Team B"

Venue context comes from the PROVISIONAL schedule table (the SF/bronze/final
rows are asserted, high confidence) and is recorded in each entry. Inference
is the Phase 2 contract (predict_match_proba with MatchContext) on the frozen
serving state — the same procedure the benchmark validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import Predictor  # noqa: E402
from train import (ELO_BLEND_W, ELO_DRAW_RATE, HFA_ELO, MatchContext,  # noqa: E402
                   elo_prior_proba, make_X, predict_match_proba)
from scorelines import DIXON_COLES_RHO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
LEDGER_PATH = ROOT / "reports" / "prediction_ledger.csv"
CACHE_PATH = ROOT / "data" / "wc2026_results.json"
SCHEDULE_PATH = ROOT / "data" / "wc2026_schedule.csv"

HOSTS = {"Mexico", "Canada", "United States"}

# Remaining-fixture registry at ledger birth (official match numbers; kickoff
# dates from the PROVISIONAL schedule's asserted rows).
FIXTURES = {
    "M101": {"stage": "SF", "kickoff_date": "2026-07-14"},
    "M102": {"stage": "SF", "kickoff_date": "2026-07-15"},
    "M103": {"stage": "third_place", "kickoff_date": "2026-07-18"},
    "M104": {"stage": "final", "kickoff_date": "2026-07-19"},
}

# Date windows for classifying cached knockout results (PROVISIONAL schedule).
QF_WINDOW = ("2026-07-09", "2026-07-11")
SF_WINDOW = ("2026-07-14", "2026-07-15")

COLUMNS = [
    "fixture_id", "stage", "kickoff_date", "kickoff_utc",
    "team_a", "team_b",
    "blend_a", "blend_draw", "blend_b",
    "elo_a", "elo_draw", "elo_b",
    "neutral", "home_advantage_team", "venue_city", "venue_country",
    "venue_source", "pairing_source",
    "bundle_sha256", "features_csv_sha256", "code_git_sha",
    "elo_blend_w", "elo_draw_rate_prior", "elo_draw_rate_baseline",
    "hfa_elo", "dixon_coles_rho",
    "generated_utc", "generated_before_kickoff", "retrospective",
]


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def load_serving() -> tuple[Predictor, dict, float]:
    """Frozen bundle + Predictor + the baseline draw rate (training-window
    draw share on canonical features — the live 'fold' analogue, recorded in
    every entry)."""
    bundle = joblib.load(MODELS_DIR / "xgb_wc2026.joblib")
    predictor = Predictor(PROCESSED_DIR / "features.csv",
                          PROCESSED_DIR / "elo_ratings.csv", bundle)
    feats = pd.read_csv(PROCESSED_DIR / "features.csv", usecols=["outcome"])
    draw_rate_baseline = float((feats["outcome"] == 1).mean())
    return predictor, bundle, draw_rate_baseline


def venue_context(fixture_id: str, team_a: str, team_b: str) -> dict:
    """MatchContext inputs from the PROVISIONAL schedule row for this match
    number. Blank venue_country would fall back to neutral (logged) — the
    SF/bronze/final rows are asserted, so in practice country is present."""
    sched = pd.read_csv(SCHEDULE_PATH, comment="#", dtype=str).fillna("")
    row = sched[sched["match_number"] == fixture_id.lstrip("M")]
    if row.empty:
        return {"neutral": True, "home_advantage_team": "",
                "venue_city": "", "venue_country": "",
                "venue_source": "schedule row missing -> neutral fallback (logged)"}
    r = row.iloc[0]
    country = r["venue_country"]
    hosts_playing = [t for t in (team_a, team_b) if t in HOSTS and t == country]
    if country and hosts_playing:
        return {"neutral": False, "home_advantage_team": hosts_playing[0],
                "venue_city": r["venue_city"], "venue_country": country,
                "venue_source": f"wc2026_schedule.csv (PROVISIONAL, confidence={r['confidence']})"}
    return {"neutral": True, "home_advantage_team": "",
            "venue_city": r["venue_city"], "venue_country": country,
            "venue_source": f"wc2026_schedule.csv (PROVISIONAL, confidence={r['confidence']})"}


def predict_pair(predictor: Predictor, bundle: dict, team_a: str, team_b: str,
                 ctx_info: dict, draw_rate_baseline: float
                 ) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Full-precision (blend, elo_baseline) under the Phase 2 venue contract
    on the frozen serving state. Returns (blend, elo, home, away): for a
    non-neutral fixture the host occupies the home slot; neutral fixtures are
    oriented alphabetically (documented, orientation-invariant by symmetry
    averaging)."""
    if not ctx_info["neutral"]:
        home = ctx_info["home_advantage_team"]
        away = team_b if home == team_a else team_a
    else:
        home, away = sorted([team_a, team_b])

    state = predictor._state
    if home not in state or away not in state:
        raise ValueError(f"team missing from serving state: {home!r}/{away!r}")

    def build_x(h, a, neutral):
        hs, as_ = state[h], state[a]
        n_h2h, hwr = predictor._h2h_stats(h, a)
        row = {
            "home_elo": hs["elo"], "away_elo": as_["elo"],
            "elo_diff": hs["elo"] - as_["elo"],
            "home_win_rate_5": hs["win_rate_5"], "away_win_rate_5": as_["win_rate_5"],
            "home_gd_5": hs["gd_5"], "away_gd_5": as_["gd_5"],
            "home_win_rate_10": hs["win_rate_10"], "away_win_rate_10": as_["win_rate_10"],
            "home_gd_10": hs["gd_10"], "away_gd_10": as_["gd_10"],
            "h2h_n": n_h2h, "h2h_home_wr": hwr,
            "home_conf_elo": hs["conf_elo"], "away_conf_elo": as_["conf_elo"],
            "neutral": int(neutral), "is_world_cup": 1,
            "home_confederation": hs["confederation"],
            "away_confederation": as_["confederation"],
        }
        X, _ = make_X(pd.DataFrame([row]), bundle["feature_cols"])
        return X

    ctx = MatchContext(home, away, neutral=ctx_info["neutral"])
    elo_h, elo_a = state[home]["elo"], state[away]["elo"]
    blend = predict_match_proba(bundle["model"], build_x, ctx, elo_h, elo_a)

    h_off = 0.0 if ctx_info["neutral"] else HFA_ELO
    e = 1.0 / (1.0 + 10.0 ** ((elo_a - elo_h - h_off) / 400.0))
    p_h = (1.0 - draw_rate_baseline) * e
    p_a = (1.0 - draw_rate_baseline) * (1.0 - e)
    elo = np.array([p_h, draw_rate_baseline, p_a])
    elo = elo / elo.sum()
    return blend, elo, home, away


def derive_fixtures_from_cache() -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Pair the remaining fixtures from cached results, by schedule date
    windows. Returns (fixture_id -> (team_a, team_b), notes). Refuses to
    guess: ambiguity or a stale cache produces notes, not pairings."""
    notes: list[str] = []
    pairs: dict[str, tuple[str, str]] = {}
    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    matches = data["matches"] if isinstance(data, dict) else data

    def winner(m) -> str | None:
        if m.get("winner") == "HOME_TEAM":
            return m["home_team"]
        if m.get("winner") == "AWAY_TEAM":
            return m["away_team"]
        return None  # DRAW should not occur in knockouts (shootout has a winner)

    qf = sorted((m for m in matches if QF_WINDOW[0] <= m["date"] <= QF_WINDOW[1]),
                key=lambda m: m["utc"])
    if len(qf) != 4:
        notes.append(f"cache holds {len(qf)} QF-window matches (need 4) — "
                     f"cache fetched_at={data.get('fetched_at')}; run a routine "
                     "tracker refresh (--refresh) and retry, or supply "
                     "--fixture M101/M102 pairings explicitly")
    else:
        w = [winner(m) for m in qf]
        by_date: dict[str, list[str]] = {}
        for m, ww in zip(qf, w):
            by_date.setdefault(m["date"], []).append(ww)
        if None in w:
            notes.append("a QF-window match has no winner field — refusing")
        elif sorted(len(v) for v in by_date.values()) == [1, 1, 2]:
            # Schedule bracket: Jul 9 & Jul 10 winners -> M101 (Dallas);
            # the two Jul 11 winners -> M102 (Atlanta). PROVISIONAL mapping.
            singles = [v[0] for d, v in sorted(by_date.items()) if len(v) == 1]
            doubles = next(v for v in by_date.values() if len(v) == 2)
            pairs["M101"] = (singles[0], singles[1])
            pairs["M102"] = (doubles[0], doubles[1])
        else:
            notes.append(f"QF date distribution {sorted(by_date)} does not match "
                         "the provisional 1+1+2 bracket grouping — supply "
                         "--fixture pairings explicitly")

    sf = sorted((m for m in matches if SF_WINDOW[0] <= m["date"] <= SF_WINDOW[1]),
                key=lambda m: m["utc"])
    if len(sf) == 2:
        sw = [winner(m) for m in sf]
        sl = [(m["home_team"] if w_ == m["away_team"] else m["away_team"])
              for m, w_ in zip(sf, sw)]
        if None not in sw:
            pairs["M104"] = (sw[0], sw[1])
            pairs["M103"] = (sl[0], sl[1])
    elif sf:
        notes.append(f"cache holds {len(sf)} SF-window matches (need 2 for "
                     "M103/M104 pairings)")
    return pairs, notes


def make_entry(fixture_id: str, team_a: str, team_b: str, predictor, bundle,
               draw_rate_baseline: float, pairing_source: str) -> dict:
    info = FIXTURES[fixture_id]
    ctx_info = venue_context(fixture_id, team_a, team_b)
    blend, elo, home, away = predict_pair(predictor, bundle, team_a, team_b,
                                          ctx_info, draw_rate_baseline)
    now = datetime.now(timezone.utc)
    before = now.date().isoformat() < info["kickoff_date"]
    return {
        "fixture_id": f"WC2026-{fixture_id}",
        "stage": info["stage"],
        "kickoff_date": info["kickoff_date"],
        "kickoff_utc": "",   # official kickoff time not held in-repo; date-level bound
        "team_a": home, "team_b": away,
        "blend_a": repr(float(blend[0])), "blend_draw": repr(float(blend[1])),
        "blend_b": repr(float(blend[2])),
        "elo_a": repr(float(elo[0])), "elo_draw": repr(float(elo[1])),
        "elo_b": repr(float(elo[2])),
        "neutral": ctx_info["neutral"],
        "home_advantage_team": ctx_info["home_advantage_team"],
        "venue_city": ctx_info["venue_city"],
        "venue_country": ctx_info["venue_country"],
        "venue_source": ctx_info["venue_source"],
        "pairing_source": pairing_source,
        "bundle_sha256": _sha256(MODELS_DIR / "xgb_wc2026.joblib"),
        "features_csv_sha256": _sha256(PROCESSED_DIR / "features.csv"),
        "code_git_sha": _git_sha(),
        "elo_blend_w": ELO_BLEND_W,
        "elo_draw_rate_prior": ELO_DRAW_RATE,
        "elo_draw_rate_baseline": repr(draw_rate_baseline),
        "hfa_elo": HFA_ELO,
        "dixon_coles_rho": DIXON_COLES_RHO,
        "generated_utc": now.isoformat(timespec="seconds"),
        "generated_before_kickoff": before,
        "retrospective": False,
    }


def append_entries(path: Path, entries: list[dict]) -> list[str]:
    """APPEND-ONLY write. Existing fixture_ids are never touched; the file is
    opened in append mode only (header written at creation). Returns the
    fixture_ids actually appended."""
    existing: set[str] = set()
    if path.exists():
        existing = set(pd.read_csv(path)["fixture_id"].astype(str))
    new = [e for e in entries if e["fixture_id"] not in existing]
    if not path.exists():
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(COLUMNS) + "\n")
    if new:
        with open(path, "a", encoding="utf-8", newline="") as fh:
            for e in new:
                fh.write(",".join(_csv_cell(e[c]) for c in COLUMNS) + "\n")
    return [e["fixture_id"] for e in new]


def _csv_cell(v) -> str:
    s = "" if v is None else str(v)
    if "," in s or '"' in s or "\n" in s:
        s = '"' + s.replace('"', '""') + '"'
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="run the routine tracker refresh first "
                             "(fetch_results.main), then ledger")
    parser.add_argument("--fixture", nargs=3, action="append",
                        metavar=("ID", "TEAM_A", "TEAM_B"),
                        help="explicit pairing, e.g. --fixture M101 'X' 'Y'")
    args = parser.parse_args()

    if args.refresh:
        import fetch_results
        fetch_results.main()

    pairs, notes = derive_fixtures_from_cache()
    source = "tracker cache (date-window bracket mapping, PROVISIONAL schedule)"
    manual = {}
    for fid, a, b in (args.fixture or []):
        if fid not in FIXTURES:
            sys.exit(f"unknown fixture id {fid!r} (expected M101/M102/M103/M104)")
        manual[fid] = (a, b)
    pairs.update(manual)

    for n in notes:
        print(f"NOTE: {n}")
    if not pairs:
        print("No fixture pairings determinable — nothing ledgered "
              "(ledger unchanged; append-only).")
        return

    predictor, bundle, drb = load_serving()
    entries = []
    for fid, (a, b) in sorted(pairs.items()):
        src = "explicit operator input" if fid in manual else source
        entries.append(make_entry(fid, a, b, predictor, bundle, drb, src))

    appended = append_entries(LEDGER_PATH, entries)
    skipped = [e["fixture_id"] for e in entries if e["fixture_id"] not in appended]
    print(f"appended {len(appended)} entries -> {LEDGER_PATH}")
    for f in appended:
        print(f"  + {f}")
    for f in skipped:
        print(f"  = {f} already ledgered (immutable, untouched)")


if __name__ == "__main__":
    main()
