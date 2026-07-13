"""
Canonical frozen-state benchmark export (remediation Phase 4).

Runs the frozen-state fold protocol (backtest.frozen_fold_run — the evaluator
validated on the 2006/2010 tune folds, imported verbatim from
blend_weight_search) over WC 2014/2018/2022 and writes:

  reports/backtest_predictions.csv     one row per match, all three systems
  reports/backtest_metrics_summary.csv per-tournament + combined metrics,
                                       plus diagnostics rows (Wilson 95% CIs,
                                       10-bin ECE, draw-pick counts) and the
                                       per-fold HFA-refit sensitivity rows
                                       promised at remediation Phase 2

Run src/paired_tests.py afterwards to append the McNemar + paired-bootstrap
rows (it strips and rewrites only its own rows).

There is deliberately NO reconciliation against previously published numbers:
the pre-remediation baseline is preserved verbatim in
reports/remediation_baseline.md, and the decomposition of old-vs-new lives in
reports/evaluation_protocol_comparison.csv (src/protocol_comparison.py).

Usage:
    python src/export_backtest.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

import backtest as bt
from backtest import BACKTEST_YEARS, brier_multiclass
from features import load_canonical_results
from train import ELO_BLEND_W, ELO_DRAW_RATE, HFA_ELO, elo_prior_proba, fit_hfa_elo

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

DIAG_TAG = "Diagnostics (192)"
SENS_TAG = "Sensitivity (HFA per-fold refit)"
SYSTEMS = [("xgb", "Raw XGBoost"), ("blend", "Blend"), ("elo", "Elo")]


def _fold_predictions(canonical: pd.DataFrame,
                      features_all: pd.DataFrame) -> list[dict]:
    """Run every benchmark fold under the canonical frozen-state protocol."""
    folds = []
    for year in BACKTEST_YEARS:
        r = bt.frozen_fold_run(canonical, features_all, year)
        n_host = int((r["fold_df"]["neutral"] == 0).sum())
        print(f"    host (non-neutral) rows: {n_host} of {len(r['y'])}")
        folds.append(r)
    return folds


def _row_frame(folds: list[dict]) -> pd.DataFrame:
    rows = []
    for f in folds:
        df_wc, y = f["fold_df"], f["y"]
        for i in range(len(df_wc)):
            rows.append({
                "date":       df_wc["date"].iloc[i].date().isoformat(),
                "tournament": f["tournament"],
                "home_team":  df_wc["home_team"].iloc[i],
                "away_team":  df_wc["away_team"].iloc[i],
                "y_true":     int(y[i]),
                "xgb_home":   f["xgb"][i, 0],
                "xgb_draw":   f["xgb"][i, 1],
                "xgb_away":   f["xgb"][i, 2],
                "blend_home": f["blend"][i, 0],
                "blend_draw": f["blend"][i, 1],
                "blend_away": f["blend"][i, 2],
                "elo_home":   f["elo"][i, 0],
                "elo_draw":   f["elo"][i, 1],
                "elo_away":   f["elo"][i, 2],
            })
    return pd.DataFrame(rows)


def _metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y, proba.argmax(axis=1)),
        "log_loss": log_loss(y, proba, labels=[0, 1, 2]),
        "brier":    brier_multiclass(y, proba),
    }


def _metrics_frame(folds: list[dict]) -> pd.DataFrame:
    rows = []
    for f in folds:
        for key, label in SYSTEMS:
            m = _metrics(f["y"], f[key])
            rows.append({"tournament": f["tournament"], "model": label,
                         "n": len(f["y"]), **m})

    y_all = np.concatenate([f["y"] for f in folds])
    for key, label in SYSTEMS:
        proba_all = np.vstack([f[key] for f in folds])
        m = _metrics(y_all, proba_all)
        rows.append({"tournament": "Combined", "model": label,
                     "n": len(y_all), **m})
    return pd.DataFrame(rows)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def _ece(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: 10 equal-width bins on top-class confidence."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf <= edges[1])
        if m.sum():
            ece += (m.sum() / len(y)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _diagnostics_rows(folds: list[dict]) -> list[dict]:
    y_all = np.concatenate([f["y"] for f in folds])
    n = len(y_all)
    n_draws = int((y_all == 1).sum())
    rows = []
    for key, label in SYSTEMS:
        proba = np.vstack([f[key] for f in folds])
        pred = proba.argmax(axis=1)
        k = int((pred == y_all).sum())
        lo, hi = _wilson(k, n)
        rows.append({"tournament": DIAG_TAG, "model": label, "n": n,
                     "statistic": "acc_wilson95", "value": k / n,
                     "detail": f"95% CI [{lo:.4f}, {hi:.4f}], {k} correct of {n}"})
        rows.append({"tournament": DIAG_TAG, "model": label, "n": n,
                     "statistic": "ece_10bin", "value": _ece(y_all, proba),
                     "detail": "10 equal-width bins on top-class confidence"})
        rows.append({"tournament": DIAG_TAG, "model": label, "n": n,
                     "statistic": "draw_picks", "value": int((pred == 1).sum()),
                     "detail": f"hard draw picks of {n}; actual draws {n_draws}"})
    return rows


def _hfa_refit_rows(folds: list[dict], features_all: pd.DataFrame) -> list[dict]:
    """
    The per-fold HFA refit sensitivity promised at Phase 2: refit the offset
    on each fold's own training window (production link, canonical data), then
    rescore Blend and Elo with the refit value on that fold's host rows. Only
    the prior/baseline shift — the model output is HFA-independent.
    """
    refit = {}
    blend_parts, elo_parts, y_parts = [], [], []
    for f in folds:
        cutoff = f["fold_df"]["date"].min()
        h = fit_hfa_elo(features_all[features_all["date"] < cutoff])
        refit[f["year"]] = h

        fd = f["fold_df"]
        prior = np.vstack([
            elo_prior_proba(
                he + (0.0 if nu else h), ae, ELO_DRAW_RATE)
            for he, ae, nu in zip(fd["frozen_home_elo"], fd["frozen_away_elo"],
                                  fd["neutral"])])
        b = ELO_BLEND_W * f["xgb"] + (1 - ELO_BLEND_W) * prior
        blend_parts.append(b / b.sum(axis=1, keepdims=True))

        base_frame = pd.DataFrame({"home_elo": fd["frozen_home_elo"].values,
                                   "away_elo": fd["frozen_away_elo"].values,
                                   "neutral": fd["neutral"].values})
        elo_parts.append(bt.elo_baseline_proba(base_frame, f["draw_rate"],
                                               hfa_elo=h))
        y_parts.append(f["y"])

    y_all = np.concatenate(y_parts)
    detail = ("refit per fold (production link, pre-fold canonical data): "
              + ", ".join(f"{yr}={h:.1f}" for yr, h in refit.items())
              + f"; production constant {HFA_ELO}")
    rows = []
    for label, proba in [("Blend", np.vstack(blend_parts)),
                         ("Elo", np.vstack(elo_parts))]:
        m = _metrics(y_all, proba)
        rows.append({"tournament": SENS_TAG, "model": label, "n": len(y_all),
                     **m, "statistic": "hfa_refit", "value": np.nan,
                     "detail": detail})
    return rows


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading canonical data + features.csv ...")
    canonical = load_canonical_results()
    features_all = pd.read_csv(PROCESSED_DIR / "features.csv",
                               parse_dates=["date"])
    print(f"  {len(features_all):,} feature rows\n")

    folds = _fold_predictions(canonical, features_all)

    rows = _row_frame(folds)
    rows_path = REPORTS_DIR / "backtest_predictions.csv"
    rows.to_csv(rows_path, index=False)

    metrics = _metrics_frame(folds)
    extra = pd.DataFrame(_diagnostics_rows(folds)
                         + _hfa_refit_rows(folds, features_all))
    summary = pd.concat([metrics, extra], ignore_index=True)
    metrics_path = REPORTS_DIR / "backtest_metrics_summary.csv"
    summary.to_csv(metrics_path, index=False, float_format="%.6f")

    print(f"\nWrote {len(rows)} match rows -> {rows_path}")
    print(f"Wrote metrics summary -> {metrics_path}")

    print("\nCombined metrics (192 matches, frozen-state protocol):")
    comb = metrics[metrics["tournament"] == "Combined"]
    for _, r in comb.iterrows():
        print(f"  {r['model']:<12} accuracy {r['accuracy']*100:5.1f}%   "
              f"log-loss {r['log_loss']:.4f}   Brier {r['brier']:.4f}")
    print("\n(old-vs-new decomposition: reports/evaluation_protocol_comparison.csv;"
          "\n pre-remediation baseline: reports/remediation_baseline.md)")


if __name__ == "__main__":
    main()
