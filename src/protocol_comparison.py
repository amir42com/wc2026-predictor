"""
Protocol decomposition (remediation Phase 4): isolate WHERE the benchmark
numbers moved between the published baseline and the canonical frozen-state
protocol, by evaluating intermediate variants that change one thing at a time.

Variants (combined 192 WC 2014/2018/2022 matches, three systems each):

  A  published        old data (pre-dedupe, unstable ordering), rolling
                      recorded-row state, no venue contract. Values copied
                      VERBATIM from reports/remediation_baseline.md — the old
                      environment is gone by design and is not recomputed.
  B0 rolling-legacy   canonical data (Phase 1 dedupe + deterministic
                      ordering), rolling recorded-row state, PRE-remediation
                      inference (all matches symmetry-averaged, no HFA).
                      A -> B0 = the data/ordering delta.
  B1 rolling-venue    same rolling state, Phase 2 venue contract (host rows
                      single-orientation + HFA, baseline gets the same HFA).
                      B0 -> B1 = the venue-contract delta.
  C  frozen-venue     the canonical protocol (one pre-tournament snapshot,
                      venue contract) — read from the just-written
                      reports/backtest_predictions.csv, NOT recomputed.
                      B1 -> C = the state-protocol delta.

The rolling variants exist ONLY here, explicitly labelled sensitivity runs;
the canonical benchmark is variant C.

Writes reports/evaluation_protocol_comparison.csv.

Usage:
    python src/protocol_comparison.py   (run AFTER export_backtest.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
from export_backtest import SYSTEMS, _metrics
from train import train_model

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# Variant A: reports/remediation_baseline.md "Combined" rows, verbatim.
PUBLISHED = {
    "Raw XGBoost": {"accuracy": 0.562500, "log_loss": 0.976240, "brier": 0.575773},
    "Blend":       {"accuracy": 0.562500, "log_loss": 0.972037, "brier": 0.573491},
    "Elo":         {"accuracy": 0.578125, "log_loss": 0.976894, "brier": 0.577025},
}

VARIANT_META = {
    "A_published":     {"data": "pre-remediation", "row_ordering": "unstable",
                        "state_protocol": "rolling", "venue_contract": "none"},
    "B0_rolling_legacy": {"data": "canonical", "row_ordering": "deterministic",
                          "state_protocol": "rolling", "venue_contract": "none"},
    "B1_rolling_venue":  {"data": "canonical", "row_ordering": "deterministic",
                          "state_protocol": "rolling", "venue_contract": "phase2"},
    "C_frozen_venue":    {"data": "canonical", "row_ordering": "deterministic",
                          "state_protocol": "frozen", "venue_contract": "phase2"},
}


def rolling_variants(features_all: pd.DataFrame) -> dict:
    """Train each fold once; evaluate the rolling recorded-row state under the
    legacy inference (B0) and the venue contract (B1)."""
    out = {"B0_rolling_legacy": [], "B1_rolling_venue": []}
    y_parts = []
    for year in bt.BACKTEST_YEARS:
        wc_mask = ((features_all["is_world_cup"] == 1)
                   & (features_all["date"].dt.year == year))
        df_wc = features_all[wc_mask].reset_index(drop=True)
        cutoff = df_wc["date"].min()
        df_pre = features_all[features_all["date"] < cutoff].reset_index(drop=True)
        print(f"  WC {year}: {len(df_wc)} matches | rolling variants | "
              f"training on {len(df_pre):,} rows ...")
        model, fc = train_model(df_pre)
        draw_rate = float((df_pre["outcome"] == 1).mean())
        y_parts.append(df_wc["outcome"].values)

        for variant, venue_aware in [("B0_rolling_legacy", False),
                                     ("B1_rolling_venue", True)]:
            xgb, blend = bt.deployed_model_and_blend(model, df_wc, fc,
                                                     venue_aware=venue_aware)
            elo = bt.elo_baseline_proba(df_wc, draw_rate,
                                        hfa_elo=bt.HFA_ELO if venue_aware else 0.0)
            out[variant].append({"xgb": xgb, "blend": blend, "elo": elo})
    return out, np.concatenate(y_parts)


def frozen_from_export() -> tuple[np.ndarray, dict]:
    """Variant C from the canonical export artifact (never recomputed here)."""
    df = pd.read_csv(REPORTS_DIR / "backtest_predictions.csv")
    y = df["y_true"].values
    probs = {key: df[[f"{key}_home", f"{key}_draw", f"{key}_away"]].values
             for key, _ in SYSTEMS}
    return y, probs


def main() -> None:
    features_all = pd.read_csv(PROCESSED_DIR / "features.csv",
                               parse_dates=["date"])

    rows = []
    for label, m in PUBLISHED.items():
        rows.append({"variant": "A_published", **VARIANT_META["A_published"],
                     "system": label, "n": 192, **m})

    rolled, y_roll = rolling_variants(features_all)
    for variant, folds in rolled.items():
        for key, label in SYSTEMS:
            proba = np.vstack([f[key] for f in folds])
            rows.append({"variant": variant, **VARIANT_META[variant],
                         "system": label, "n": len(y_roll),
                         **_metrics(y_roll, proba)})

    y_frozen, probs = frozen_from_export()
    for key, label in SYSTEMS:
        rows.append({"variant": "C_frozen_venue",
                     **VARIANT_META["C_frozen_venue"],
                     "system": label, "n": len(y_frozen),
                     **_metrics(y_frozen, probs[key])})

    table = pd.DataFrame(rows)
    out = REPORTS_DIR / "evaluation_protocol_comparison.csv"
    header = (
        "# Protocol decomposition (remediation Phase 4). Deltas, per system:\n"
        "#   A->B0 data/ordering (Phase 1 dedupe + deterministic sort)\n"
        "#   B0->B1 venue contract (Phase 2: host rows + HFA, both systems)\n"
        "#   B1->C  state protocol (rolling recorded rows -> ONE frozen\n"
        "#          pre-tournament snapshot)\n"
        "# A is copied verbatim from reports/remediation_baseline.md. C is read\n"
        "# from reports/backtest_predictions.csv (the canonical benchmark).\n"
        "# Rolling variants are sensitivity runs, never headline numbers.\n"
    )
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write(header)
        table.to_csv(fh, index=False, float_format="%.6f")
    print(f"\nwrote {out}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
