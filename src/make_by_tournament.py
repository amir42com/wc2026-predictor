"""
Per-tournament breakdown of the WC backtest (supplementary to Figures 3-5).

Splits the SAME 192 leakage-free match predictions (produced via the shared
train.predict_neutral_proba inference and written to
reports/backtest_predictions.csv by src/export_backtest.py) by tournament year,
and scores all three systems with the IDENTICAL metric definitions used for the
combined summary (export_backtest._metrics → accuracy, log-loss with labels
[0,1,2], multiclass Brier). No predictions are recomputed here.

Reuses make_figures' dark theme + SYSTEM_COLORS palette (no duplicated style).

Writes  reports/backtest_by_tournament.csv          (9 rows: 3 years x 3 systems)
        reports/figures/figure9_by_tournament.png   (grouped log-loss + Brier)

Usage:
    python src/make_by_tournament.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Shared style + palette (single source of truth — never re-declared here).
from make_figures import (REPORTS_DIR, FIG_DIR, SYSTEM_COLORS, _dark_axes,
                          DARK_BG, DARK_TEXT, DARK_TEXT2, DARK_GRID, DARK_SPINE)
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


def figure9(table: pd.DataFrame) -> None:
    """Grouped bars: each tournament's three systems side by side, log-loss
    (left) and Brier (right). Lower is better; the panels expose that the
    blend's probabilistic-scoring edge is a 2014 effect that reverses in 2018."""
    years = sorted(table["tournament_year"].unique())
    x = np.arange(len(years))
    width = 0.26
    offsets = {sysname: (i - 1) * width for i, (sysname, _) in enumerate(SYSTEMS)}

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    panels = [("log_loss", "Log-loss", axes[0]), ("brier", "Brier", axes[1])]

    for metric, title, ax in panels:
        _dark_axes(fig, ax)
        for sysname, _ in SYSTEMS:
            vals = [float(table[(table["tournament_year"] == y)
                                & (table["system"] == sysname)][metric].iloc[0])
                    for y in years]
            xs = x + offsets[sysname]
            ax.bar(xs, vals, width=width, color=SYSTEM_COLORS[sysname],
                   edgecolor=DARK_BG, linewidth=0.8, zorder=2,
                   label=sysname if metric == "log_loss" else None)
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7.5, color=DARK_TEXT2, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([f"WC {y}" for y in years], fontsize=11, color=DARK_TEXT)
        ax.set_ylabel(f"{title}  (lower is better)", fontsize=10.5)
        ax.set_title(title, fontsize=11.5)
        lo = min(table[metric]) - 0.04
        hi = max(table[metric]) + 0.06
        ax.set_ylim(max(0.0, lo), hi)
        ax.grid(axis="y", color=DARK_GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9.5,
                     framealpha=0.92, facecolor="#151d3b", edgecolor=DARK_SPINE,
                     bbox_to_anchor=(0.5, 1.0))
    for txt in leg.get_texts():
        txt.set_color(DARK_TEXT)

    fig.suptitle("WC backtest by tournament — probabilistic scoring per system\n"
                 "The blend's edge over Elo is a 2014 effect and reverses in 2018",
                 fontsize=11.5, color=DARK_TEXT, y=1.13)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = FIG_DIR / "figure9_by_tournament.png"
    fig.savefig(out, dpi=220, facecolor=DARK_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


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
    print("  VERDICT: " + (
        "the blend's pooled probabilistic edge is NOT consistent -- it is driven "
        "almost entirely by WC 2014 and reverses (Elo wins) in WC 2018."
        if diff.get(2014, 0) < 0 and diff.get(2018, 0) > 0 else
        "the blend's probabilistic edge is broadly consistent across tournaments."))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table = per_tournament_table()
    write_csv(table)
    figure9(table)
    _print_table(table)
    _reconcile(table)
    _verdict(table)


if __name__ == "__main__":
    main()
