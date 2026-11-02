"""
Per-tournament breakdown of the WC backtest (supplementary to Figures 3-5).

Splits the SAME 192 leakage-free match predictions (produced via the shared
train.predict_match_proba inference and written to
reports/backtest_predictions.csv by src/export_backtest.py) by tournament year,
and scores all three systems with the IDENTICAL metric definitions used for the
combined summary (export_backtest._metrics → accuracy, log-loss with labels
[0,1,2], multiclass Brier). No predictions are recomputed here.

Reuses make_figures' website figure contract (light and dark pair).

Writes  reports/backtest_by_tournament.csv          (9 rows: 3 years x 3 systems)
        reports/figures/figure5_by_tournament-{light,dark}.{png,svg}
                                                    (log-loss + Brier dot plot)

Usage:
    python src/make_by_tournament.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Shared website figure contract and palette (single source of truth).
from make_figures import (REPORTS_DIR, FIG_DIR, NOTE_PT, SYSTEM_MARKER,
                          _clean_axes, render_pair, system_color)
# Identical metric definitions as the combined summary.
from export_backtest import _metrics

PRED_CSV = REPORTS_DIR / "backtest_predictions.csv"
OUT_CSV = REPORTS_DIR / "backtest_by_tournament.csv"

# (display system, probability columns) — fixed canonical order for the figure.
SYSTEMS = [
    ("Raw XGBoost",  ["xgb_home", "xgb_draw", "xgb_away"]),
    ("Blend",        ["blend_home", "blend_draw", "blend_away"]),
    ("Elo baseline", ["elo_home", "elo_draw", "elo_away"]),
]


def per_tournament_table() -> pd.DataFrame:
    """One row per (tournament_year x system), 9 rows, sorted by year then system."""
    df = pd.read_csv(PRED_CSV)
    rows = []
    for tourn, sub in df.groupby("tournament"):
        year = int(str(tourn).split()[-1])      # "WC 2014" -> 2014
        y = sub["y_true"].values
        for system, cols in SYSTEMS:
            m = _metrics(y, sub[cols].values)
            rows.append({"tournament_year": year, "n_matches": len(sub),
                         "system": system, "accuracy": m["accuracy"],
                         "log_loss": m["log_loss"], "brier": m["brier"]})
    out = pd.DataFrame(rows, columns=["tournament_year", "n_matches", "system",
                                      "accuracy", "log_loss", "brier"])
    return out.sort_values(["tournament_year", "system"]).reset_index(drop=True)


def write_csv(table: pd.DataFrame) -> None:
    table.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"Wrote {len(table)} rows -> {OUT_CSV}")


def figure5(table: pd.DataFrame) -> None:
    """Dot plot per tournament: each system's log-loss (top) and Brier
    (bottom), one marker shape per system. Lower is better. The differences
    are a few thousandths, so dots on a free scale show them honestly where
    bars would need a truncated baseline."""
    years = sorted(table["tournament_year"].unique())
    rows = {y: i for i, y in enumerate(reversed(years))}     # earliest on top
    dodge = {name: (1 - i) * 0.22 for i, (name, _) in enumerate(SYSTEMS)}

    def draw(fig, theme):
        axes = fig.subplots(2, 1, sharey=True)
        for ax, (metric, label) in zip(axes, (("log_loss", "Log-loss"),
                                              ("brier", "Brier score"))):
            _clean_axes(ax, theme, grid_axis="x")
            for y in years:
                ax.axhline(rows[y], color=theme["rule"], linewidth=1.0, zorder=0)
            for name, _ in SYSTEMS:
                sub = table[table["system"] == name]
                ax.scatter(sub[metric], [rows[y] + dodge[name]
                                         for y in sub["tournament_year"]],
                           s=420, marker=SYSTEM_MARKER[name],
                           color=system_color(theme, name), linewidth=0,
                           zorder=3, label=name)
            ax.set_yticks([rows[y] for y in years])
            ax.set_yticklabels([f"WC {y}" for y in years], color=theme["text"])
            ax.set_ylim(-0.6, len(years) - 0.4)
            ax.spines["left"].set_visible(False)
            ax.set_xlabel(f"{label} (lower is better)")
            ax.tick_params(axis="x", labelsize=NOTE_PT)
        axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3,
                       fontsize=NOTE_PT, borderaxespad=0, handletextpad=0.2,
                       columnspacing=1.0, markerscale=0.9)
        fig.supxlabel(f"{int(table['n_matches'].iloc[0])} matches per "
                      "tournament, predicted before each one.",
                      fontsize=NOTE_PT, color=theme["soft"])

    render_pair("figure5_by_tournament", draw, 1700)


def _print_table(table: pd.DataFrame) -> None:
    print("\nPer-tournament backtest metrics (192 matches; 64 per tournament):")
    print(f"  {'Year':<6}{'N':>4}  {'System':<13}{'Accuracy':>10}"
          f"{'Log-loss':>11}{'Brier':>9}")
    print("  " + "-" * 53)
    for _, r in table.iterrows():
        print(f"  {int(r['tournament_year']):<6}{int(r['n_matches']):>4}  "
              f"{r['system']:<13}{r['accuracy']*100:>9.1f}%"
              f"{r['log_loss']:>11.4f}{r['brier']:>9.4f}")


def _reconcile(table: pd.DataFrame) -> None:
    """Match-weighted average of per-tournament metrics must match the combined
    headline (each tournament is 64 matches, so it is an exact pooled mean)."""
    print("\nReconciliation vs combined 192-match summary (match-weighted):")
    for system, _ in SYSTEMS:
        sub = table[table["system"] == system]
        w = sub["n_matches"].values
        acc = np.average(sub["accuracy"].values, weights=w)
        ll = np.average(sub["log_loss"].values, weights=w)
        br = np.average(sub["brier"].values, weights=w)
        print(f"  {system:<13} acc {acc*100:5.1f}%  log-loss {ll:.4f}  Brier {br:.4f}")


def _verdict(table: pd.DataFrame) -> None:
    piv = table.pivot(index="tournament_year", columns="system", values="log_loss")
    diff = (piv["Blend"] - piv["Elo baseline"])   # negative => blend better
    parts = ", ".join(f"{y}: {d:+.4f}" for y, d in diff.items())
    print(f"\n  Blend - Elo log-loss by year (negative = blend better): {parts}")
    better = [str(y) for y, d in diff.items() if d < 0]
    worse = [str(y) for y, d in diff.items() if d > 0]
    print(f"  Blend lower log-loss in: {', '.join(better) or 'none'}; "
          f"Elo lower in: {', '.join(worse) or 'none'}.")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table = per_tournament_table()
    write_csv(table)
    figure5(table)
    _print_table(table)
    _reconcile(table)
    _verdict(table)


if __name__ == "__main__":
    main()
