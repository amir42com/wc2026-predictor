"""
Task 3: canonical team-name normalization. Asserts every WC 2026 team resolves
to a real model state and a real confederation (catching a Curaçao-style
accented fall-through), and that all known aliases canonicalize consistently.

Runs standalone (`python tests/test_team_names.py`, exit 0/1) and under pytest.
"""

import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from simulate import GROUPS                      # noqa: E402
from team_names import ALIASES, canonical        # noqa: E402
import features as feat                          # noqa: E402

WC_TEAMS = sorted({t for ts in GROUPS.values() for t in ts})
STATE = pd.read_csv(ROOT / "data" / "processed" / "team_state.csv")
STATE_TEAMS = set(STATE["team"])
STATE_CONF = dict(zip(STATE["team"], STATE["confederation"]))


def test_forty_eight_teams():
    assert len(WC_TEAMS) == 48, f"expected 48 WC teams, got {len(WC_TEAMS)}"


def test_all_teams_resolve_to_a_state():
    missing = [t for t in WC_TEAMS if canonical(t) not in STATE_TEAMS]
    assert not missing, f"WC teams with no Predictor state: {missing}"


def test_no_team_resolves_to_other():
    # Confederation via the SERVING snapshot AND via features._conf — both must
    # be a real confederation for every WC team (this is the Curaçao guard).
    bad_state = [t for t in WC_TEAMS if STATE_CONF.get(canonical(t)) == "Other"]
    bad_conf = [t for t in WC_TEAMS if feat._conf(t) == "Other"]
    assert not bad_state, f"WC teams in confederation 'Other' (state): {bad_state}"
    assert not bad_conf, f"WC teams in confederation 'Other' (_conf): {bad_conf}"


def test_groups_spellings_are_canonical():
    # The app's group spellings must already be canonical (idempotent).
    noncanon = [t for t in WC_TEAMS if canonical(t) != t]
    assert not noncanon, f"GROUPS uses non-canonical spellings: {noncanon}"


def test_aliases_map_consistently():
    # Every alias resolves to a canonical name that is itself a fixed point, and
    # the known accent/abbreviation/API families collapse to one name.
    for alias, canon in ALIASES.items():
        assert canonical(alias) == canon, f"{alias!r} -> {canonical(alias)!r} != {canon!r}"
        assert canonical(canon) == canon, f"canonical not idempotent for {canon!r}"
    # The Curaçao accent family specifically must agree.
    assert canonical("Curacao") == canonical("Curaçao") == "Curaçao"
    # Korea API variants both collapse to the app spelling.
    assert canonical("Korea Republic") == canonical("Republic of Korea") == "South Korea"


def main() -> int:
    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    print(f"Team-name tests ({len(tests)} cases):")
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
