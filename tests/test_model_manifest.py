"""
Phase 3: the shipped bundle must carry a provenance manifest that matches
reality — the bundle bytes, the training-input bytes, and the frozen
production constants the deployed inference uses.

Runs standalone (`python tests/test_model_manifest.py`, exit 0/1) and under
pytest.
"""

import hashlib
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "model-manifest.json"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_manifest_matches_reality():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))

    # The manifest must describe the bundle actually on disk...
    assert m["bundle_sha256"] == _sha256(ROOT / "models" / m["bundle_file"])
    # ...the exact training input...
    assert m["features_csv_sha256"] == _sha256(
        ROOT / "data" / "processed" / "features.csv")
    assert m["raw_results_sha256"] == _sha256(
        ROOT / "data" / "raw" / "results.csv")
    # ...and the frozen constants the deployed inference uses.
    assert m["elo_blend_w"] == train.ELO_BLEND_W
    assert m["elo_draw_rate"] == train.ELO_DRAW_RATE
    assert m["hfa_elo"] == train.HFA_ELO
    import scorelines
    assert m["dixon_coles_rho"] == scorelines.DIXON_COLES_RHO

    # The referenced blend-weight-search artifact must exist.
    assert (ROOT / m["blend_weight_search"]).exists()


if __name__ == "__main__":
    try:
        test_manifest_matches_reality()
        print("PASS  test_manifest_matches_reality")
    except AssertionError as exc:
        print(f"FAIL  test_manifest_matches_reality: {exc}")
        sys.exit(1)
