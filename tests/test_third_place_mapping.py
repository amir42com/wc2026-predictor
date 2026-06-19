"""
Exhaustive tests for the official FIFA 2026 Annexe C third-place mapping.

Runs with plain Python (no pytest dependency) — `python tests/test_third_place_mapping.py`
exits 0 on success, 1 on any failure — and is also pytest-compatible (the
test_* functions use bare asserts). The minimal load-time/lookup checks are
mirrored into src/smoke_test.py for the pre-push hook; the exhaustive 495-row
coverage lives here.
"""

import sys
from itertools import combinations
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import third_place_mapping as tpm   # noqa: E402
import simulate as sim              # noqa: E402

ALL_COMBOS = ["".join(c) for c in combinations(tpm.GROUPS_ALL, 8)]  # 495, sorted

# Per-winner-slot eligible groups built from the repo's OWN R32 constraints
# (the simulator already does this at import; reuse it rather than re-derive).
ELIGIBLE = sim.ELIGIBLE_THIRD_BY_WINNER

# A handful of rows copied verbatim from the source file, as regression anchors
# (first key, last key, and several from the middle).
EXPECTED_ROWS = {
    "ABCDEFGH": {"1A": "3H", "1B": "3G", "1D": "3B", "1E": "3C",
                 "1G": "3A", "1I": "3F", "1K": "3D", "1L": "3E"},
    "ABCDEFGI": {"1A": "3C", "1B": "3G", "1D": "3B", "1E": "3D",
                 "1G": "3A", "1I": "3F", "1K": "3E", "1L": "3I"},
    "ABCEGIKL": {"1A": "3E", "1B": "3G", "1D": "3B", "1E": "3A",
                 "1G": "3I", "1I": "3C", "1K": "3L", "1L": "3K"},
    "ACDEFGHJ": {"1A": "3H", "1B": "3G", "1D": "3J", "1E": "3C",
                 "1G": "3A", "1I": "3F", "1K": "3D", "1L": "3E"},
    "EFGHIJKL": {"1A": "3E", "1B": "3J", "1D": "3I", "1E": "3F",
                 "1G": "3H", "1I": "3G", "1K": "3L", "1L": "3K"},
}


def _expect_raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} from {fn.__name__}{args!r}, none raised")


# ── structure & completeness ────────────────────────────────────────────────

def test_load_time_validation_passes():
    assert tpm.validate_structure() is True


def test_exactly_495_rows_all_combos():
    assert len(tpm.THIRD_PLACE_MAPPING) == 495
    assert set(tpm.THIRD_PLACE_MAPPING) == set(ALL_COMBOS)


def test_all_495_resolve_via_lookup():
    for combo in ALL_COMBOS:
        row = tpm.lookup(list(combo))
        assert set(row) == set(tpm.WINNER_SLOTS), combo


def test_every_row_is_a_bijection_on_its_key():
    for key, row in tpm.THIRD_PLACE_MAPPING.items():
        thirds = [row[s] for s in tpm.WINNER_SLOTS]
        groups = [t[1] for t in thirds]
        assert all(t[0] == "3" and t[1] in tpm.GROUPS_ALL for t in thirds), key
        assert len(set(groups)) == 8, f"{key}: a third assigned twice ({groups})"
        assert set(groups) == set(key), f"{key}: groups {sorted(set(groups))} != key"


# ── eligibility cross-check against the repo's R32 constraints ───────────────

def test_eligibility_all_495_rows():
    failures = tpm.check_eligibility(ELIGIBLE, raise_on_fail=False)
    assert failures == [], f"{len(failures)} eligibility failures, e.g. {failures[:3]}"


# ── representative rows (first / middle / last) ──────────────────────────────

def test_representative_rows_match_source():
    for key, expected in EXPECTED_ROWS.items():
        assert tpm.lookup(list(key)) == expected, key


def test_lookup_is_order_independent():
    assert tpm.lookup(list("HGFEDCBA")) == tpm.THIRD_PLACE_MAPPING["ABCDEFGH"]


# ── invalid inputs raise clearly ─────────────────────────────────────────────

def test_too_few_groups_raises():
    _expect_raises(tpm.ThirdPlaceMappingError, tpm.lookup, list("ABCDEFG"))   # 7


def test_too_many_groups_raises():
    _expect_raises(tpm.ThirdPlaceMappingError, tpm.lookup, list("ABCDEFGHI"))  # 9


def test_duplicate_letters_raise():
    _expect_raises(tpm.ThirdPlaceMappingError, tpm.lookup, list("ABCDEFGG"))


def test_unknown_letter_raises():
    _expect_raises(tpm.ThirdPlaceMappingError, tpm.lookup, list("ABCDEFGM"))


def test_missing_combination_raises():
    # Every valid 8-of-12 combination exists, so simulate a missing row by
    # swapping in a reduced copy (original object restored afterwards).
    original = tpm.THIRD_PLACE_MAPPING
    reduced = dict(original)
    reduced.pop("ABCDEFGH")
    tpm.THIRD_PLACE_MAPPING = reduced
    try:
        _expect_raises(tpm.ThirdPlaceMappingError, tpm.lookup, list("ABCDEFGH"))
    finally:
        tpm.THIRD_PLACE_MAPPING = original


def test_validate_structure_rejects_wrong_row_count():
    _expect_raises(tpm.ThirdPlaceMappingError, tpm.validate_structure,
                   {"ABCDEFGH": tpm.THIRD_PLACE_MAPPING["ABCDEFGH"]})


# ── resolve_r32 wiring: opponents come from the table, eligible, no dupes ────

def _fake_group_tables(qualifying_third_groups: set) -> dict:
    """12 groups; each group's third gets high points iff its group qualifies,
    so resolve_r32 selects exactly `qualifying_third_groups` as the 8 thirds."""
    tables = {}
    for g in tpm.GROUPS_ALL:
        third_pts = 5 if g in qualifying_third_groups else 0
        tables[g] = [
            {"team": f"{g}1", "pts": 9, "gd": 9, "gf": 9},   # winner
            {"team": f"{g}2", "pts": 6, "gd": 3, "gf": 5},   # runner-up
            {"team": f"{g}3", "pts": third_pts, "gd": 0, "gf": third_pts},  # third
            {"team": f"{g}4", "pts": 0, "gd": -9, "gf": 0},
        ]
    return tables


def test_resolve_r32_uses_annexe_c_opponents():
    qualifying = set("ABCDEFGH")
    tables = _fake_group_tables(qualifying)
    r32 = sim.resolve_r32(tables)
    assert len(r32) == 16
    assert len({t for m in r32 for t in m}) == 32  # 32 distinct teams, no TBD

    assignment = tpm.lookup(qualifying)
    third_teams_used = []
    for idx, (a, b) in enumerate(sim.R32_SLOTS):
        if idx in sim.THIRD_ELIGIBLE:
            winner_slot = a if a.startswith("1") else b
            grp = assignment[winner_slot][1]              # group letter
            opponent = r32[idx][1]                        # third sits second
            assert opponent == f"{grp}3", (idx, winner_slot, opponent)
            assert grp in sim.ELIGIBLE_THIRD_BY_WINNER[winner_slot]
            third_teams_used.append(opponent)
    assert len(set(third_teams_used)) == 8               # each qualifying third once


# ── runner ───────────────────────────────────────────────────────────────────

def main() -> int:
    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    print(f"Annexe C mapping tests ({len(tests)} cases):")
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
