"""
Row-level export of the leakage-free WC backtest.

Walks the SAME walk-forward folds as src/backtest.py (WC 2014/2018/2022, each
model trained only on matches strictly before that tournament's first match)
and writes, for every match:

  * the raw XGBoost probabilities   (model.predict_proba, BEFORE blending)
  * the production blend            (ELO_BLEND_W * XGB + (1-ELO_BLEND_W) * Elo)
  * the naive Elo-baseline          (elo_baseline_proba)

Outputs (tracked, publishable with the article):
  reports/backtest_predictions.csv       one row per match, all three models
  reports/backtest_metrics_summary.csv   accuracy / log-loss / Brier, per
                                         tournament and combined, per model

No modelling code is modified — this only re-runs the existing backtest logic
(train.train_model / make_X / ELO_BLEND_W, backtest.elo_baseline_proba /
brier_multiclass) and records what it produces, so the numbers reconcile to
the article exactly.

Usage:
    python src/export_backtest.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from train import ELO_BLEND_W, make_X, train_model
from backtest import BACKTEST_YEARS, brier_multiclass, elo_baseline_proba

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# Article headline numbers (combined, 192 matches) for reconciliation.
# (accuracy fraction, log-loss, multiclass Brier)
ARTICLE_COMBINED = {
    "xgb":   (0.542, 0.9897, 0.5859),
    "blend": (0.552, 0.9809, 0.5804),
    "elo":   (0.578, 0.9769, 0.5770),
}
RECONCILE_TOL = {"acc": 0.005, "ll": 0.0005, "brier": 0.0005}


def _fold_predictions(df: pd.DataFrame) -> list[dict]:
    """Run every backtest fold and collect per-match rows + pooled arrays."""
    folds = []
    for year in BACKTEST_YEARS:
        wc_mask = (df["is_world_cup"] == 1) & (df["date"].dt.year == year)
        df_wc = df[wc_mask].reset_index(drop=True)
        if df_wc.empty:
            print(f"  WC {year}: no matches found — skipping.")
            continue

        cutoff = df_wc["date"].min()
        df_pre = df[df["date"] < cutoff].reset_index(drop=True)
        print(f"  WC {year}: {len(df_wc)} matches | "
              f"training on {len(df_pre):,} matches before {cutoff.date()} ...")

        model, feature_cols = train_model(df_pre)

        X_wc, _ = make_X(df_wc, feature_cols)
        y_wc = df_wc["outcome"].values

        # Raw XGB (before the Elo blend)
        xgb_proba = model.predict_proba(X_wc)
        # Elo baseline — draw rate from the same pre-tournament window (no leakage)
        draw_rate = float((df_pre["outcome"] == 1).mean())
        elo_proba = elo_baseline_proba(df_wc, draw_rate)
        # Production blend (identical to backtest.py)
        blend_proba = ELO_BLEND_W * xgb_proba + (1 - ELO_BLEND_W) * elo_proba

        folds.append({
            "tournament": f"WC {year}",
            "df_wc": df_wc,
            "y": y_wc,
            "xgb": xgb_proba,
            "blend": blend_proba,
            "elo": elo_proba,
        })
    return folds


def _row_frame(folds: list[dict]) -> pd.DataFrame:
    rows = []
    for f in folds:
        df_wc, y = f["df_wc"], f["y"]
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
    model_keys = [("xgb", "Raw XGBoost"), ("blend", "Blend"), ("elo", "Elo")]
    rows = []

    # Per tournament
    for f in folds:
        for key, label in model_keys:
            m = _metrics(f["y"], f[key])
            rows.append({"tournament": f["tournament"], "model": label,
                         "n": len(f["y"]), **m})

    # Combined (pool every match so per-sample metrics are exact)
    y_all = np.concatenate([f["y"] for f in folds])
    for key, label in model_keys:
        proba_all = np.vstack([f[key] for f in folds])
        m = _metrics(y_all, proba_all)
        rows.append({"tournament": "Combined", "model": label,
                     "n": len(y_all), **m})

    return pd.DataFrame(rows)


def _reconcile(metrics: pd.DataFrame) -> None:
    print("\nReconciliation against article headline numbers (combined, 192):")
    key_to_label = {"xgb": "Raw XGBoost", "blend": "Blend", "elo": "Elo"}
    comb = metrics[metrics["tournament"] == "Combined"].set_index("model")
    any_off = False
    for key, (a_acc, a_ll, a_br) in ARTICLE_COMBINED.items():
        row = comb.loc[key_to_label[key]]
        d_acc = abs(row["accuracy"] - a_acc)
        d_ll = abs(row["log_loss"] - a_ll)
        d_br = abs(row["brier"] - a_br)
        off = (d_acc > RECONCILE_TOL["acc"] or d_ll > RECONCILE_TOL["ll"]
               or d_br > RECONCILE_TOL["brier"])
        flag = "  <-- DIFFERS" if off else "ok"
        any_off = any_off or off
        print(f"  {key_to_label[key]:<12} "
              f"acc {row['accuracy']*100:5.1f}% (article {a_acc*100:.1f}%, d {d_acc*100:+.2f}pp)  "
              f"ll {row['log_loss']:.4f} (art {a_ll:.4f}, d {d_ll:+.4f})  "
              f"brier {row['brier']:.4f} (art {a_br:.4f}, d {d_br:+.4f})  [{flag}]")
    if any_off:
        print("  WARNING: at least one metric differs by more than rounding.")
    else:
        print("  All three models reconcile to the article within rounding.")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features.csv ...")
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    print(f"  {len(df):,} rows total\n")

    folds = _fold_predictions(df)

    rows = _row_frame(folds)
    rows_path = REPORTS_DIR / "backtest_predictions.csv"
    rows.to_csv(rows_path, index=False)

    metrics = _metrics_frame(folds)
    metrics_path = REPORTS_DIR / "backtest_metrics_summary.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.6f")

    print(f"\nWrote {len(rows)} match rows -> {rows_path}")
    print(f"Wrote metrics summary -> {metrics_path}")

    print("\nCombined metrics (192 matches):")
    comb = metrics[metrics["tournament"] == "Combined"]
    for _, r in comb.iterrows():
        print(f"  {r['model']:<12} accuracy {r['accuracy']*100:5.1f}%   "
              f"log-loss {r['log_loss']:.4f}   Brier {r['brier']:.4f}")

    _reconcile(metrics)


if __name__ == "__main__":
    main()
