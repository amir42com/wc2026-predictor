"""
Fetch finished WC 2026 results from football-data.org, score them against the
model's pre-tournament predictions, and cache to data/wc2026_results.json.

The model's team state comes from data/processed/features.csv, which ends the
day before the tournament (2026-06-10).  As long as that file is NOT rebuilt to
include tournament matches, every prediction here is leakage-free: the model has
never seen a single WC 2026 result.

API key resolution (never hardcoded):
    1. environment variable FOOTBALL_DATA_KEY            (standalone / CI)
    2. st.secrets["api"]["football_data_key"]            (Streamlit context)
    3. .streamlit/secrets.toml [api] football_data_key   (local standalone)

Usage:
    python src/fetch_results.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import Predictor  # noqa: E402
from scorelines import top_scorelines  # noqa: E402

ROOT          = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
RESULTS_PATH  = ROOT / "data" / "wc2026_results.json"

API_URL = "https://api.football-data.org/v4/competitions/WC/matches"

# football-data.org team names -> our model's team keys
TEAM_NAME_MAP: dict[str, str] = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR":           "DR Congo",
    "Czechia":            "Czech Republic",
    "IR Iran":            "Iran",
    "Korea Republic":     "South Korea",
    "Republic of Korea":  "South Korea",
    "USA":                "United States",
}


def get_api_key() -> str | None:
    """Resolve the API key without ever hardcoding it. Returns None if absent."""
    if os.environ.get("FOOTBALL_DATA_KEY"):
        return os.environ["FOOTBALL_DATA_KEY"]

    try:  # only succeeds inside a running Streamlit app with secrets configured
        import streamlit as st

        return st.secrets["api"]["football_data_key"]
    except Exception:
        pass

    secrets_file = ROOT / ".streamlit" / "secrets.toml"
    if secrets_file.exists():
        try:
            import tomllib

            with open(secrets_file, "rb") as fh:
                return tomllib.load(fh)["api"]["football_data_key"]
        except Exception:
            return None
    return None


def map_team(name: str | None) -> str | None:
    if name is None:
        return None
    return TEAM_NAME_MAP.get(name, name)


def load_predictor() -> Predictor:
    bundle = joblib.load(MODELS_DIR / "xgb_wc2026.joblib")
    return Predictor(
        PROCESSED_DIR / "features.csv",
        PROCESSED_DIR / "elo_ratings.csv",
        bundle,
    )


def fetch_raw_matches(key: str, timeout: int = 30) -> list[dict]:
    resp = requests.get(API_URL, headers={"X-Auth-Token": key}, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("matches", [])


def _parse_utc(s: str | None) -> datetime | None:
    """Parse an ISO-8601 UTC timestamp (…Z) to an aware datetime, or None."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _pick_next_fixture(raw: list[dict], now: datetime) -> dict | None:
    """
    Choose the genuine next fixture from raw football-data.org matches.

    A currently-live match (IN_PLAY/PAUSED) takes priority as the "current" one;
    otherwise the earliest match whose kickoff is still in the FUTURE
    (TIMED/SCHEDULED with utcDate > now) is used. Selecting on the kickoff time —
    not just the status — means a finished or already-started match is never
    picked even if the feed is slow to flip its status, and the default keeps
    advancing as matches kick off. Matches whose team names don't normalize are
    skipped so one unmappable fixture can't break the default. Returns the
    normalized dict (home/away/utc/status) or None.
    """
    cands: list[tuple[int, str, dict]] = []
    for m in raw:
        status = m.get("status")
        ko = _parse_utc(m.get("utcDate"))
        is_live = status in ("IN_PLAY", "PAUSED")
        is_upcoming = (status in ("TIMED", "SCHEDULED")
                       and ko is not None and ko > now)
        if is_live or is_upcoming:
            # live first (rank 0), then earliest future kickoff
            cands.append((0 if is_live else 1, m.get("utcDate") or "9999-12-31", m))

    cands.sort(key=lambda t: (t[0], t[1]))
    for _, _, m in cands:
        home = map_team((m.get("homeTeam") or {}).get("name"))
        away = map_team((m.get("awayTeam") or {}).get("name"))
        if home and away:
            return {"home": home, "away": away,
                    "utc": m.get("utcDate"), "status": m.get("status")}
    return None


def get_next_fixture() -> dict | None:
    """
    Next upcoming WC 2026 fixture, with team names normalized to the app's names:
        {"home": str, "away": str, "utc": str, "status": str}

    Earliest future kickoff (a live match counts as current); see
    _pick_next_fixture for the selection rule. Returns None on any failure or
    when there is no upcoming fixture, so the caller can fall back gracefully.
    """
    try:
        key = get_api_key()
        if not key:
            return None
        raw = fetch_raw_matches(key)
    except Exception:
        return None
    return _pick_next_fixture(raw, datetime.now(timezone.utc))


def scorelines_for_proba(p_home: float, p_draw: float, p_away: float,
                         top_n: int = 5) -> list[list]:
    """
    Top-N most likely exact scorelines for a fixture, oriented home-away, as
    ``[[home_goals, away_goals, prob], ...]`` highest first.

    These are a deterministic function of the *pre-match* W/D/L probabilities
    via the calibrated Dixon-Coles layer (which is built to reproduce that
    exact W/D/L). They are NEVER computed from the actual result, so they carry
    the same pre-match integrity as the W/D/L predictions themselves.
    """
    sl = top_scorelines(float(p_home), float(p_draw), float(p_away), top_n=top_n)
    return [[int(hg), int(ag), round(float(pr), 4)] for (hg, ag), pr in sl]


def score_matches(raw_matches: list[dict], predictor: Predictor) -> list[dict]:
    """Keep FINISHED matches; attach model probabilities and correctness."""
    out: list[dict] = []
    unmapped: set[str] = set()

    for m in raw_matches:
        if m.get("status") != "FINISHED":
            continue

        home = map_team(m["homeTeam"]["name"])
        away = map_team(m["awayTeam"]["name"])
        if not home or not away:
            continue

        ft = m["score"]["fullTime"]
        hs, as_ = ft.get("home"), ft.get("away")
        if hs is None or as_ is None:
            continue

        actual = 0 if hs > as_ else (1 if hs == as_ else 2)

        # Pre-tournament, neutral-venue probabilities (p_home, p_draw, p_away)
        if home in predictor._state and away in predictor._state:
            p = predictor.predict(home, away)
        else:
            unmapped.update(t for t in (home, away) if t not in predictor._state)
            p = np.array([np.nan, np.nan, np.nan])

        predicted = int(np.nanargmax(p)) if not np.isnan(p).all() else None

        # Pre-match top-5 scorelines, derived only from the locked W/D/L above.
        scorelines = (None if predicted is None
                      else scorelines_for_proba(p[0], p[1], p[2], top_n=5))

        out.append({
            "date":       m["utcDate"][:10],
            "utc":        m["utcDate"],
            "home_team":  home,
            "away_team":  away,
            "home_score": int(hs),
            "away_score": int(as_),
            "winner":     m["score"].get("winner"),
            "actual_outcome":    actual,
            "predicted_outcome": predicted,
            "p_home": None if np.isnan(p[0]) else round(float(p[0]), 4),
            "p_draw": None if np.isnan(p[1]) else round(float(p[1]), 4),
            "p_away": None if np.isnan(p[2]) else round(float(p[2]), 4),
            "scorelines": scorelines,
            "correct": None if predicted is None else bool(predicted == actual),
        })

    if unmapped:
        print(f"  WARNING: no model state for {sorted(unmapped)} — add to TEAM_NAME_MAP")

    out.sort(key=lambda r: r["utc"], reverse=True)  # newest first
    return out


def build_payload(matches: list[dict]) -> dict:
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source":     "football-data.org",
        "n_matches":  len(matches),
        "matches":    matches,
    }


def fetch_and_save() -> dict:
    """Used by the Streamlit tracker. Raises on API/key failure."""
    key = get_api_key()
    if not key:
        raise RuntimeError("No football-data.org API key found.")
    raw = fetch_raw_matches(key)
    matches = score_matches(raw, load_predictor())
    payload = build_payload(matches)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    key = get_api_key()
    if not key:
        print("ERROR: no API key. Set FOOTBALL_DATA_KEY or add it to "
              ".streamlit/secrets.toml under [api] football_data_key.")
        sys.exit(1)

    print("Fetching WC 2026 matches from football-data.org ...")
    raw = fetch_raw_matches(key)
    statuses: dict[str, int] = {}
    for m in raw:
        statuses[m.get("status", "?")] = statuses.get(m.get("status", "?"), 0) + 1
    print(f"  {len(raw)} matches total — " +
          ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items())))

    matches = score_matches(raw, load_predictor())
    payload = build_payload(matches)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    scored = [m for m in matches if m["correct"] is not None]
    print(f"\n  {len(matches)} finished matches saved -> {RESULTS_PATH}")
    if scored:
        n_correct = sum(m["correct"] for m in scored)
        print(f"  Model called {n_correct}/{len(scored)} correctly "
              f"({n_correct/len(scored)*100:.0f}%)\n")
        for m in matches:
            mark = "OK " if m["correct"] else "X  "
            if m["correct"] is None:
                mark = "?  "
            pp = (f"[{m['p_home']*100:.0f}/{m['p_draw']*100:.0f}/{m['p_away']*100:.0f}]"
                  if m["p_home"] is not None else "[no pred]")
            print(f"  {mark}{m['date']}  {m['home_team']} {m['home_score']}-"
                  f"{m['away_score']} {m['away_team']:<22} {pp}")


if __name__ == "__main__":
    main()
