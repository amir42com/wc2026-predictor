"""
Exact paired comparisons of Blend vs Elo on the same 192 backtest matches.

Reads  reports/backtest_predictions.csv  (written by src/export_backtest.py)
Prints, and appends to reports/backtest_metrics_summary.csv:

  1. EXACT two-sided McNemar test on per-match correctness (uses the actual
     discordant counts b and c, not the best-case bound). This replaces the
     conservative "best-case 0.0625" figure with the real p-value.
  2. Paired bootstrap 95% CI for the mean per-match log-loss difference
     (Blend - Elo), 10,000 resamples, fixed seed.

The append is idempotent: any previously-appended paired-test rows are stripped
and rewritten, so re-running this script never duplicates rows. (Re-running
export_backtest.py rewrites the summary from scratch — run this afterwards.)

Usage:
    python src/paired_tests.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
PRED_CSV = REPORTS_DIR / "backtest_predictions.csv"
SUMMARY_CSV = REPORTS_DIR / "backtest_metrics_summary.csv"

PAIRED_TAG = "Paired test (192)"   # tournament-column marker for appended rows
N_BOOT = 10_000
BOOT_SEED = 12345


def _per_match_logloss(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """-log probability assigned to the true outcome, per match."""
    proba = df[[f"{prefix}_home", f"{prefix}_draw", f"{prefix}_away"]].values
    y = df["y_true"].values
    p_true = proba[np.arange(len(y)), y]
    return -np.log(p_true)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """
    Exact two-sided McNemar test on paired binary correctness.

    b = A correct & B wrong, c = A wrong & B correct (the discordant pairs).
    Under H0 the b-type count ~ Binomial(b + c, 0.5); the exact two-sided
    p-value comes straight from that binomial (scipy binomtest).
    """
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n_disc = b + c
    if n_disc == 0:
        p = 1.0
    else:
        p = binomtest(min(b, c), n_disc, 0.5, alternative="two-sided").pvalue
    return {"b": b, "c": c, "n_discordant": n_disc, "p_value": float(p)}


def bootstrap_logloss_ci(diff: np.ndarray, n_boot: int = N_BOOT,
                         seed: int = BOOT_SEED) -> dict:
    """Paired bootstrap 95% CI for the mean of `diff` (Blend - Elo log-loss)."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = diff[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": n_boot, "seed": seed}


def _append_summary(mc: dict, boot: dict) -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    # Drop any previously-appended paired rows for idempotency
    if "tournament" in summary.columns:
        summary = summary[summary["tournament"] != PAIRED_TAG].copy()

    new_rows = pd.DataFrame([
        {"tournament": PAIRED_TAG, "model": "McNemar exact (Blend vs Elo)",
         "n": 192, "statistic": "p_value", "value": mc["p_value"],
         "detail": f"b={mc['b']} (blend-only correct), c={mc['c']} "
                   f"(elo-only correct), discordant={mc['n_discordant']}"},
        {"tournament": PAIRED_TAG, "model": "Bootstrap logloss diff (Blend-Elo)",
         "n": 192, "statistic": "mean_diff", "value": boot["mean_diff"],
         "detail": f"95% CI [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}], "
                   f"{boot['n_boot']} resamples, seed {boot['seed']}"},
    ])
    out = pd.concat([summary, new_rows], ignore_index=True)
    out.to_csv(SUMMARY_CSV, index=False, float_format="%.6f")
    print(f"\nAppended paired-test rows -> {SUMMARY_CSV}")


def main() -> None:
    df = pd.read_csv(PRED_CSV)
    y = df["y_true"].values
    print(f"Loaded {len(df)} match rows from backtest_predictions.csv\n")

    blend_pred = df[["blend_home", "blend_draw", "blend_away"]].values.argmax(1)
    elo_pred = df[["elo_home", "elo_draw", "elo_away"]].values.argmax(1)
    correct_blend = blend_pred == y
    correct_elo = elo_pred == y

    print(f"  Blend correct: {correct_blend.sum()}/{len(y)}   "
          f"Elo correct: {correct_elo.sum()}/{len(y)}")

    # 1) Exact McNemar
    mc = mcnemar_exact(correct_blend, correct_elo)
    print("\nExact two-sided McNemar (Blend vs Elo correctness):")
    print(f"  b (Blend correct, Elo wrong) = {mc['b']}")
    print(f"  c (Blend wrong, Elo correct) = {mc['c']}")
    print(f"  discordant pairs             = {mc['n_discordant']}")
    print(f"  exact two-sided p-value      = {mc['p_value']:.4f}")

    # 2) Paired bootstrap log-loss CI
    ll_blend = _per_match_logloss(df, "blend")
    ll_elo = _per_match_logloss(df, "elo")
    diff = ll_blend - ll_elo
    boot = bootstrap_logloss_ci(diff)
    print("\nPaired bootstrap — mean per-match log-loss difference (Blend - Elo):")
    print(f"  point estimate = {boot['mean_diff']:+.4f}  "
          f"(Blend mean {ll_blend.mean():.4f} vs Elo {ll_elo.mean():.4f})")
    print(f"  95% CI         = [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]"
          f"  ({boot['n_boot']} resamples, seed {boot['seed']})")
    print("  (positive => Blend has the higher/worse log-loss)")

    _append_summary(mc, boot)


if __name__ == "__main__":
    main()
