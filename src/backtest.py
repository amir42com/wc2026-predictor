"""
Walk-forward backtest of the WC prediction model on past World Cups.

For each tournament (2014, 2018, 2022):
  - Train a fresh XGBoost model on all matches strictly BEFORE the
    tournament's first match (no leakage — for WC 2018 the model has
    never seen June 2018 or later).
  - Evaluate on that tournament's matches only.
  - Compare against a naive Elo baseline: always predict the higher-Elo
    team wins.  Baseline probabilities come from the standard Elo
    logistic formula, with the draw share estimated from the same
    pre-tournament training window.

Metrics: accuracy, log-loss, multi-class Brier score.

Usage:
    python src/backtest.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from train import ELO_BLEND_W, make_X, train_model

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

BACKTEST_YEARS = [2014, 2018, 2022]


def brier_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean multi-class Brier score: mean over matches of sum_c (p_c - y_c)^2."""
    y_bin = np.zeros_like(proba)
    y_bin[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - y_bin) ** 2, axis=1)))


def elo_baseline_proba(df_wc: pd.DataFrame, draw_rate: float) -> np.ndarray:
    """
    Naive Elo baseline probabilities.

    Win probability from the standard Elo expectation
        E_home = 1 / (1 + 10^((away_elo - home_elo) / 400))
    then reserve `draw_rate` (estimated from pre-tournament data) for the
    draw class and split the rest proportionally.  argmax of these
    probabilities always picks the higher-Elo team, matching the hard
    "higher Elo wins" rule.
    """
    e_home = 1.0 / (1.0 + 10.0 ** ((df_wc["away_elo"] - df_wc["home_elo"]) / 400.0))
    p_home = (1.0 - draw_rate) * e_home
    p_away = (1.0 - draw_rate) * (1.0 - e_home)
    proba = np.column_stack([p_home, np.full(len(df_wc), draw_rate), p_away])
    return proba / proba.sum(axis=1, keepdims=True)


def evaluate_proba(y_true: np.ndarray, proba: np.ndarray) -> dict:
    return {
        "acc":   accuracy_score(y_true, proba.argmax(axis=1)),
        "ll":    log_loss(y_true, proba, labels=[0, 1, 2]),
        "brier": brier_multiclass(y_true, proba),
    }


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

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

        # Baseline draw rate from the same pre-tournament window — no leakage
        draw_rate = float((df_pre["outcome"] == 1).mean())
        base_proba = elo_baseline_proba(df_wc, draw_rate)

        # Production model = XGB blended with the Elo prior (weight tuned on
        # WC 2006/2010 — see notebooks/04_model_improvement.md)
        model_proba = (ELO_BLEND_W * model.predict_proba(X_wc)
                       + (1 - ELO_BLEND_W) * base_proba)

        rows.append({"tournament": f"WC {year}", "n": len(df_wc),
                     "model": evaluate_proba(y_wc, model_proba),
                     "base":  evaluate_proba(y_wc, base_proba),
                     "y": y_wc,
                     "model_proba": model_proba,
                     "base_proba": base_proba})

    # Combined: pool all matches so per-sample metrics are exact
    y_all     = np.concatenate([r["y"] for r in rows])
    model_all = np.vstack([r["model_proba"] for r in rows])
    base_all  = np.vstack([r["base_proba"] for r in rows])
    rows.append({"tournament": "Combined", "n": len(y_all),
                 "model": evaluate_proba(y_all, model_all),
                 "base":  evaluate_proba(y_all, base_all)})

    return rows


def print_table(rows: list) -> None:
    W = 78
    print()
    print("WC Backtest - production model (XGB+Elo blend) vs naive Elo baseline".center(W))
    print("(each model trained only on matches before that tournament)".center(W))
    print("=" * W)
    print(f"{'':14}{'':>4}  |{'Model':^28} |{'Elo baseline':^28}")
    print(f"{'Tournament':<14}{'N':>4}  |{'Acc':>8}{'LogLoss':>10}{'Brier':>9} |"
          f"{'Acc':>8}{'LogLoss':>10}{'Brier':>9}")
    print("-" * W)

    for r in rows:
        if r["tournament"] == "Combined":
            print("-" * W)
        m, b = r["model"], r["base"]
        print(f"{r['tournament']:<14}{r['n']:>4}  |"
              f"{m['acc']*100:>7.1f}%{m['ll']:>10.4f}{m['brier']:>9.4f} |"
              f"{b['acc']*100:>7.1f}%{b['ll']:>10.4f}{b['brier']:>9.4f}")

    print("=" * W)
    comb = rows[-1]
    d_acc = (comb["model"]["acc"] - comb["base"]["acc"]) * 100
    d_ll  = comb["base"]["ll"] - comb["model"]["ll"]
    print(f"Combined edge over baseline: {d_acc:+.1f} pp accuracy, "
          f"{d_ll:+.4f} log-loss (lower is better)")


def main() -> None:
    print("Loading features.csv ...")
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    print(f"  {len(df):,} rows total\n")

    rows = backtest(df)
    print_table(rows)


if __name__ == "__main__":
    main()
