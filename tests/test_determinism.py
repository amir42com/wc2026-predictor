"""
Phase 1B regression tests: the feature pipeline must be a pure function of the
canonical data, not of the raw file's incidental row order.

  * shuffle-within-date invariance — permuting the raw file's rows within each
    date block and rebuilding yields byte-identical features/team_state/elo
    outputs (this is exactly the perturbation the old unstable date-only sort
    leaked: 94% of feature rows moved, up to ~36 Elo points);
  * idempotence — building twice from the same input is byte-identical;
  * cleaned-data row counts — the Checkpoint-1A duplicate policy numbers,
    pinned to the frozen raw snapshot.

By default the build-twice tests run on a bounded slice (dates < 1980, ~12k
rows) to stay CI-friendly; set DETERMINISM_FULL=1 to run them on the full
history (~1 min). The full run was executed and reported at Checkpoint 1B.

Runs standalone (`python tests/test_determinism.py`, exit 0/1) and under
pytest.
"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import features as feat  # noqa: E402

RAW_CSV = feat.RAW_DIR / "results.csv"
# Frozen snapshot pinned in reports/DATA_SNAPSHOT.md. The count assertions
# below are validated against THIS byte state; if the raw file is re-fetched,
# they must be re-validated against a fresh duplicate scan (Checkpoint 1A
# procedure), not silently updated.
PINNED_RAW_SHA256 = "59f49de6055179ebe8c7f4d9f31b579c938f9eafd25e925abdffd91885f08e43"
CUTOFF = pd.Timestamp("2026-06-11")

SLICE_END = pd.Timestamp("1980-01-01")
FULL = os.environ.get("DETERMINISM_FULL") == "1"


def _raw_sha256() -> str:
    return hashlib.sha256(RAW_CSV.read_bytes()).hexdigest()


def _build_bytes(df: pd.DataFrame) -> tuple[str, str, str]:
    """Build features and serialize the three outputs exactly as main() does."""
    features, elo_df, team_state_df = feat.build_features(df, state_cutoff=CUTOFF)
    return (
        features.to_csv(index=False),
        team_state_df.to_csv(index=False),
        elo_df.to_csv(index=False),
    )


def _bounded(df: pd.DataFrame) -> pd.DataFrame:
    if FULL:
        return df
    return df[df["date"] < SLICE_END].reset_index(drop=True)


def _shuffle_within_dates(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute rows within each date block, keeping dates chronological —
    the exact degree of freedom the raw feed does not guarantee. (Full random
    shuffle, then a STABLE date-only sort: dates return to order, the random
    within-date permutation survives.)"""
    return (
        df.sample(frac=1.0, random_state=seed)
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )


def test_cleaned_row_counts():
    assert _raw_sha256() == PINNED_RAW_SHA256, (
        "raw snapshot changed — re-run the Checkpoint 1A duplicate scan and "
        "re-validate these counts before updating them"
    )
    file_rows = pd.read_csv(RAW_CSV, parse_dates=["date"])
    assert len(file_rows) == 49_477

    # Policy headline: dedupe on MODEL_COLUMNS removes exactly one file row
    # (the 2026-06-06 Gibraltar v Cayman Islands venue-string double entry).
    deduped = file_rows.sort_values(feat.CANONICAL_SORT_KEY, kind="stable")
    deduped = deduped.drop_duplicates(subset=feat.MODEL_COLUMNS, keep="first")
    assert len(deduped) == 49_476

    # Loader outputs: scored rows only, then the published training cutoff.
    assert len(feat.load_canonical_results()) == 49_444
    assert len(feat.load_canonical_results(cutoff=CUTOFF)) == 49_404


def test_shuffle_within_date_invariance():
    canonical = _bounded(feat.load_canonical_results(cutoff=CUTOFF))
    base = _build_bytes(canonical)
    for seed in (0, 1):
        shuffled = _shuffle_within_dates(canonical, seed)
        assert not shuffled.equals(canonical), "shuffle was a no-op; test is vacuous"
        assert _build_bytes(shuffled) == base, (
            f"outputs depend on within-date input order (seed {seed})"
        )


def test_shuffled_raw_file_roundtrip():
    """End-to-end: a within-date-shuffled RAW FILE fed through the canonical
    loader (sort + dedupe + cutoff) rebuilds byte-identical outputs."""
    raw = pd.read_csv(RAW_CSV, parse_dates=["date"])
    with tempfile.TemporaryDirectory() as tmp:
        shuffled_path = Path(tmp) / "shuffled_raw.csv"
        _shuffle_within_dates(raw, seed=7).to_csv(shuffled_path, index=False)
        a = _bounded(feat.load_canonical_results(cutoff=CUTOFF))
        b = _bounded(feat.load_canonical_results(path=shuffled_path, cutoff=CUTOFF))
        pd.testing.assert_frame_equal(
            a[feat.MODEL_COLUMNS], b[feat.MODEL_COLUMNS], check_dtype=False
        )
        assert _build_bytes(a) == _build_bytes(b)


def test_idempotence():
    canonical = _bounded(feat.load_canonical_results(cutoff=CUTOFF))
    assert _build_bytes(canonical) == _build_bytes(canonical.copy())


if __name__ == "__main__":
    failed = 0
    for fn in (test_cleaned_row_counts, test_shuffle_within_date_invariance,
               test_shuffled_raw_file_roundtrip, test_idempotence):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    sys.exit(1 if failed else 0)
