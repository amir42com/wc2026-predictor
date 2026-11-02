"""
Phase 6a: the prediction ledger is append-only from birth — regenerating must
never mutate an existing entry, byte for byte. Synthetic entries only; no
model bundle, no WC 2026 data.
"""

import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import prediction_ledger as pl  # noqa: E402


def _entry(fid: str, pa: float) -> dict:
    e = {c: "" for c in pl.COLUMNS}
    e.update({
        "fixture_id": fid, "stage": "SF", "kickoff_date": "2026-07-14",
        "team_a": "Alpha", "team_b": "Beta",
        "blend_a": repr(pa), "blend_draw": repr(0.2), "blend_b": repr(0.8 - pa),
        "elo_a": repr(0.4), "elo_draw": repr(0.25), "elo_b": repr(0.35),
        "neutral": True, "generated_utc": "2026-07-13T00:00:00+00:00",
        "generated_before_kickoff": True, "retrospective": False,
    })
    return e


def test_ledger_is_append_only():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ledger.csv"

        added = pl.append_entries(path, [_entry("WC2026-M101", 0.5)])
        assert added == ["WC2026-M101"]
        first = path.read_bytes()

        # Re-appending the SAME fixture with DIFFERENT numbers must be a no-op.
        added = pl.append_entries(path, [_entry("WC2026-M101", 0.999)])
        assert added == []
        assert path.read_bytes() == first, "existing entry was mutated"

        # Appending a new fixture extends the file; the original bytes are a
        # strict prefix (nothing above them was rewritten).
        added = pl.append_entries(path, [_entry("WC2026-M101", 0.1),
                                         _entry("WC2026-M102", 0.3)])
        assert added == ["WC2026-M102"]
        assert path.read_bytes().startswith(first)

        # Idempotence of a full regeneration pass.
        snapshot = path.read_bytes()
        added = pl.append_entries(path, [_entry("WC2026-M101", 0.7),
                                         _entry("WC2026-M102", 0.7)])
        assert added == [] and path.read_bytes() == snapshot


def test_csv_cell_escaping():
    assert pl._csv_cell("plain") == "plain"
    assert pl._csv_cell('a,"b"') == '"a,""b"""'
    assert pl._csv_cell(None) == ""


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
