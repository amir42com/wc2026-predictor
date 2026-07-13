"""
One-off cross-check of the PROVISIONAL data/wc2026_schedule.csv venue table
against football-data.org's venue fields (remediation Phase 2, Checkpoint 2
decision).

Run this ALONGSIDE the next ROUTINE tracker refresh — it performs its own GET
of the same fixtures endpoint fetch_results.py uses (same API-key resolution),
because the cached data/wc2026_results.json does not store venue fields. It is
NOT a tracker refresh and writes nothing to the tracker cache.

Boundary hygiene: this tool reads ONLY date / stage / group / matchday / venue
(and team names for HOST group fixtures, whose participation was fixed by the
draw). It never reads, prints, or stores scores, winners, or anything derived
from results — knockout fixtures are compared by stage + date + venue alone.

Usage:
    python src/check_schedule_venues.py            # report only
    python src/check_schedule_venues.py --write    # also stamp verified="api"
                                                   # on rows that match

Report per drafted row: OK (venue agrees), MISMATCH (drafted venue disagrees —
shown side by side), or FILL (row was left blank; the API venue is suggested
so a human can complete the table). Manual fifa.com spot-check ("fifa" stamp)
remains required for fixtures the API has not scheduled/played yet.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_results import API_URL, get_api_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_PATH = ROOT / "data" / "wc2026_schedule.csv"

HOSTS = {"Mexico", "Canada", "United States"}

# API stage strings -> schedule-table stage values (2026 48-team format).
STAGE_MAP = {
    "GROUP_STAGE": "group",
    "LAST_32": "R32",
    "LAST_16": "R16",
    "QUARTER_FINALS": "QF",
    "SEMI_FINALS": "SF",
    "THIRD_PLACE": "third_place",
    "FINAL": "final",
}

# FIFA uses sponsor-free venue names; football-data.org may use either form.
# Both sides are normalized through these aliases before comparison.
VENUE_ALIASES = {
    "estadio ciudad de mexico": "estadio azteca",
    "mexico city stadium": "estadio azteca",
    "estadio guadalajara": "estadio akron",
    "guadalajara stadium": "estadio akron",
    "toronto stadium": "bmo field",
    "bc place vancouver": "bc place",
    "vancouver stadium": "bc place",
    "los angeles stadium": "sofi stadium",
    "seattle stadium": "lumen field",
    "boston stadium": "gillette stadium",
    "kansas city stadium": "arrowhead stadium",
    "geha field at arrowhead stadium": "arrowhead stadium",
    "miami stadium": "hard rock stadium",
    "dallas stadium": "at&t stadium",
    "atlanta stadium": "mercedes-benz stadium",
    "new york new jersey stadium": "metlife stadium",
}


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = " ".join(s.lower().replace("/", " ").split())
    return VENUE_ALIASES.get(s, s)


def _venues_agree(drafted: str, api: str) -> bool:
    a, b = _norm(drafted), _norm(api)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def fetch_fixture_venues() -> list[dict]:
    """GET the fixtures list; keep ONLY schedule fields (never scores)."""
    key = get_api_key()
    if not key:
        sys.exit("No football-data.org API key found (FOOTBALL_DATA_KEY / secrets).")
    resp = requests.get(API_URL, headers={"X-Auth-Token": key}, timeout=30)
    resp.raise_for_status()
    out = []
    for m in resp.json().get("matches", []):
        out.append({
            "date":     (m.get("utcDate") or "")[:10],
            "stage":    STAGE_MAP.get(m.get("stage"), m.get("stage")),
            "group":    (m.get("group") or "").replace("GROUP_", ""),
            "matchday": m.get("matchday"),
            "venue":    m.get("venue") or "",
            # Team names retained ONLY to locate host group fixtures (draw-
            # public information). Never scores.
            "home":     ((m.get("homeTeam") or {}).get("name") or ""),
            "away":     ((m.get("awayTeam") or {}).get("name") or ""),
        })
    return out


def _api_match_for_row(row: pd.Series, api: list[dict]) -> "dict | None":
    """Locate the API fixture for a drafted row without consulting results."""
    if row["stage"] == "group":
        cands = [m for m in api if m["stage"] == "group"
                 and m["group"] == row["group"]
                 and str(m["matchday"]) == str(row["matchday"])]
        host = row["host_team"]
        if host and not pd.isna(host):
            from team_names import canonical
            cands = [m for m in cands
                     if canonical(m["home"]) == host or canonical(m["away"]) == host]
        return cands[0] if len(cands) == 1 else None
    # Knockouts: stage + date only (teams deliberately not consulted).
    cands = [m for m in api if m["stage"] == row["stage"]
             and (pd.isna(row["date"]) or not row["date"] or m["date"] == row["date"])]
    if len(cands) == 1:
        return cands[0]
    # Same-day siblings (e.g. two QFs on July 11): disambiguate by venue.
    named = [m for m in cands if _venues_agree(row["stadium"], m["venue"])]
    return named[0] if len(named) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help='stamp verified="api" on rows whose venue matches')
    args = parser.parse_args()

    sched = pd.read_csv(SCHEDULE_PATH, comment="#", dtype=str).fillna("")
    api = fetch_fixture_venues()
    print(f"{len(sched)} drafted rows / {len(api)} API fixtures\n")

    ok = mismatch = fill = unresolved = 0
    for i, row in sched.iterrows():
        m = _api_match_for_row(row, api)
        label = f"{row['stage']:<12} {row['group'] or '-'}{row['matchday'] or ''} seq {row['seq']}"
        if m is None:
            unresolved += 1
            print(f"UNRESOLVED  {label}: no unique API fixture (not yet scheduled, "
                  f"or needs manual disambiguation)")
            continue
        if not row["stadium"]:
            fill += 1
            print(f"FILL        {label}: API venue '{m['venue']}' on {m['date']} "
                  f"(row was drafted blank — complete manually, then re-run)")
            continue
        if _venues_agree(row["stadium"], m["venue"]):
            ok += 1
            date_note = "" if (not row["date"] or row["date"] == m["date"]) else \
                f"  [DATE differs: drafted {row['date']}, API {m['date']}]"
            print(f"OK          {label}: '{row['stadium']}' == '{m['venue']}'{date_note}")
            if args.write and not sched.at[i, "verified"]:
                sched.at[i, "verified"] = "api"
        else:
            mismatch += 1
            print(f"MISMATCH    {label}: drafted '{row['stadium']}' "
                  f"({row['venue_city']}, {row['venue_country']}) vs API '{m['venue']}'")

    print(f"\nSummary: {ok} OK, {mismatch} MISMATCH, {fill} FILL, "
          f"{unresolved} UNRESOLVED of {len(sched)} rows")
    if mismatch:
        print("MISMATCH rows must be corrected before the table loses its "
              "PROVISIONAL flag.")

    if args.write:
        # Preserve the header comment block byte-for-byte, rewrite the data.
        header_lines = []
        with open(SCHEDULE_PATH, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("#"):
                    header_lines.append(line)
                else:
                    break
        with open(SCHEDULE_PATH, "w", encoding="utf-8", newline="") as fh:
            fh.writelines(header_lines)
            sched.to_csv(fh, index=False)
        print(f'verified="api" stamps written to {SCHEDULE_PATH}')


if __name__ == "__main__":
    main()
