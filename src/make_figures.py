"""
Article figures 3, 4 and 5 from the backtest export.

Reads  reports/backtest_predictions.csv     (written by src/export_backtest.py)
       reports/backtest_metrics_summary.csv  (combined row → Figure 3 table)
Writes reports/figures/figure3_backtest_table.png
       reports/figures/figure4_accuracy_wilson.png
       reports/figures/figure5_blend_reliability.png

matplotlib only (no seaborn).

Usage:
    python src/make_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless / file output only
import matplotlib.pyplot as plt

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIG_DIR = REPORTS_DIR / "figures"

# Outcome palette — RESERVED for match outcomes (home/draw/away). Used by
# Figure 5 (reliability) only; mirrors the app's blue/grey/amber convention.
CLASS_COLORS = {"home": "#3b82f6", "draw": "#93a1c8", "away": "#f59e0b"}

# System palette — for figures that compare the three SYSTEMS (Figures 3 & 4).
# Deliberately NOT the outcome blue/grey/amber: violet family = "my models",
# teal = the Elo benchmark that beat them, set visually apart.
SYSTEM_COLORS = {
    "Raw XGBoost":  "#9C8AC9",   # light violet
    "Blend":        "#6C4AA6",   # deep violet (production model)
    "Elo baseline": "#2A9D8F",   # teal (the benchmark)
}


# --------------------------------------------------------------------------- #
# Reusable stats helpers
# --------------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion k/n.

    Returns (low, high) as proportions. Unlike the Wald interval it stays
    inside [0, 1] and is well-behaved for the small n (192) here.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (centre - half, centre + half)


def multiclass_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10,
                   strategy: str = "quantile") -> tuple[float, dict]:
    """
    One-vs-rest Expected Calibration Error for a multiclass classifier.

    For each class c, treat p_c vs the binary indicator (y == c), bin the
    predicted probabilities (10 equal-frequency / quantile bins by default),
    and accumulate |confidence - accuracy| weighted by bin count. The overall
    ECE is the sample-weighted mean of the per-class ECEs (each class
    contributes n samples), matching the reliability curves in figure 5.

    Returns (overall_ece, {class_index: per_class_ece}).
    """
    n = len(y_true)
    per_class = {}
    for c in range(proba.shape[1]):
        p = proba[:, c]
        hit = (y_true == c).astype(float)
        if strategy == "quantile":
            edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        else:
            edges = np.linspace(0, 1, n_bins + 1)
        # assign each sample to a bin; clip the right edge into the last bin
        idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
        ece = 0.0
        for b in range(len(edges) - 1):
            m = idx == b
            if not m.any():
                continue
            conf = p[m].mean()
            acc = hit[m].mean()
            ece += (m.sum() / n) * abs(conf - acc)
        per_class[c] = ece
    overall = float(np.mean(list(per_class.values())))  # equal n per class
    return overall, per_class


def _reliability_points(hit: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """
    Equal-frequency bin points for a reliability curve.

    `hit` is the BINARY one-vs-rest target (1 where the class occurred, else 0)
    and `p` the predicted probability for that class. Returns
    (mean predicted, observed frequency, count) per quantile bin.
    """
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    xs, ys, ns = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        xs.append(p[m].mean())
        ys.append(hit[m].mean())
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


# --------------------------------------------------------------------------- #
# Figure 4 — accuracy with Wilson intervals
# --------------------------------------------------------------------------- #
def figure4(df: pd.DataFrame) -> None:
    n = len(df)
    y = df["y_true"].values

    specs = [
        ("Raw XGBoost", ["xgb_home", "xgb_draw", "xgb_away"], SYSTEM_COLORS["Raw XGBoost"]),
        ("Blend",       ["blend_home", "blend_draw", "blend_away"], SYSTEM_COLORS["Blend"]),
        ("Elo",         ["elo_home", "elo_draw", "elo_away"], SYSTEM_COLORS["Elo baseline"]),
    ]

    labels, ks, accs, los, his, colors = [], [], [], [], [], []
    for label, cols, color in specs:
        pred = df[cols].values.argmax(axis=1)
        k = int((pred == y).sum())
        lo, hi = wilson_interval(k, n)
        labels.append(label)
        ks.append(k)
        accs.append(k / n)
        los.append(lo)
        his.append(hi)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(labels))
    accs = np.array(accs)
    yerr = np.vstack([accs - np.array(los), np.array(his) - accs])

    bars = ax.bar(x, accs, width=0.58, color=colors, edgecolor="#1f2937",
                  linewidth=0.8, zorder=2)
    ax.errorbar(x, accs, yerr=yerr, fmt="none", ecolor="#1f2937",
                elinewidth=1.5, capsize=8, capthick=1.5, zorder=3)

    for xi, (acc, k, lo, hi) in enumerate(zip(accs, ks, los, his)):
        ax.text(xi, acc + 0.002, f"{acc*100:.1f}%\n({k}/{n})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(xi, hi + 0.006, f"95% CI\n[{lo*100:.1f}, {hi*100:.1f}]",
                ha="center", va="bottom", fontsize=8, color="#475569")

    ax.set_ylim(0.40, 0.70)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("WC backtest accuracy (192 matches, 2014/2018/2022)\n"
                 "Point estimates differ by ~5 matches; Wilson 95% intervals "
                 "overlap heavily",
                 fontsize=11.5)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / "figure4_accuracy_wilson.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    print("  Accuracy (k/n) with 95% Wilson intervals:")
    for label, k, acc, lo, hi in zip(labels, ks, accs, los, his):
        print(f"    {label:<12} {k:>3}/{n}  {acc*100:5.1f}%  "
              f"[{lo*100:.1f}%, {hi*100:.1f}%]")


# --------------------------------------------------------------------------- #
# Figure 5 — Blend reliability (one-vs-rest per class)
# --------------------------------------------------------------------------- #
def figure5(df: pd.DataFrame) -> None:
    y = df["y_true"].values
    proba = df[["blend_home", "blend_draw", "blend_away"]].values

    overall_ece, per_class = multiclass_ece(y, proba, n_bins=10, strategy="quantile")
    class_names = {0: "home", 1: "draw", 2: "away"}

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", linewidth=1.2,
            label="Perfect calibration", zorder=1)

    for c in range(3):
        name = class_names[c]
        hit = (y == c).astype(float)
        xs, ys, ns = _reliability_points(hit, proba[:, c], n_bins=10)
        sizes = 30 + (ns / ns.max()) * 120
        ax.plot(xs, ys, "-", color=CLASS_COLORS[name], linewidth=1.6,
                alpha=0.9, zorder=2)
        ax.scatter(xs, ys, s=sizes, color=CLASS_COLORS[name],
                   edgecolor="#1f2937", linewidth=0.6, zorder=3,
                   label=f"{name.capitalize()}  (ECE {per_class[c]*100:.1f}%)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Observed frequency", fontsize=11)
    ax.set_title("Blend reliability — one-vs-rest, 10 equal-frequency bins\n"
                 f"Overall multiclass ECE = {overall_ece*100:.1f}%  "
                 "(point size ∝ bin count)",
                 fontsize=11.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(color="#eceff3", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / "figure5_blend_reliability.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    print("  Blend reliability — Expected Calibration Error:")
    for c in range(3):
        print(f"    {class_names[c]:<6} ECE {per_class[c]*100:.2f}%")
    print(f"    overall ECE {overall_ece*100:.2f}%")


# --------------------------------------------------------------------------- #
# Figure 3 — combined backtest comparison table (rendered as an image)
# --------------------------------------------------------------------------- #
def _combined_metrics() -> pd.DataFrame:
    """
    Pull the COMBINED (192-match) row for each system from the metrics summary,
    so the table can never drift from the data. Returns rows in fixed order
    (Raw XGBoost, Blend, Elo baseline) with display names.
    """
    summary = pd.read_csv(REPORTS_DIR / "backtest_metrics_summary.csv")
    comb = summary[summary["tournament"] == "Combined"].set_index("model")
    order = [("Raw XGBoost", "Raw XGBoost"),
             ("Blend", "Blend"),
             ("Elo", "Elo baseline")]
    rows = []
    for csv_name, display in order:
        r = comb.loc[csv_name]
        rows.append({"system": display,
                     "accuracy": float(r["accuracy"]),
                     "log_loss": float(r["log_loss"]),
                     "brier": float(r["brier"])})
    return pd.DataFrame(rows)


def figure3() -> None:
    data = _combined_metrics()

    # Best per metric: accuracy higher is better; log-loss / Brier lower.
    best = {"accuracy": int(data["accuracy"].idxmax()),
            "log_loss": int(data["log_loss"].idxmin()),
            "brier":    int(data["brier"].idxmin())}

    plt.rcParams["font.family"] = "DejaVu Sans"  # clean sans-serif
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column x-anchors (right edges for the numeric columns)
    x_swatch = (0.015, 0.034)
    x_system = 0.060
    cols = {"accuracy": (0.40, 0.575),   # (cell left, right-align x)
            "log_loss": (0.60, 0.775),
            "brier":    (0.80, 0.975)}
    headers = {"accuracy": "Accuracy", "log_loss": "Log-loss", "brier": "Brier"}

    # Row geometry
    header_y0, header_y1 = 0.80, 0.96
    row_h = 0.175
    top = header_y0
    row_bands = [(top - (i + 1) * row_h, top - i * row_h) for i in range(len(data))]

    INK = "#1f2937"
    GREY = "#5b6473"
    SEP = "#d9dce1"
    HEADER_BG = "#eef0f3"
    HIGHLIGHT = "#e7f3f0"  # very light teal — subtle "best" emphasis

    # Header band
    ax.add_patch(plt.Rectangle((0, header_y0), 1, header_y1 - header_y0,
                               facecolor=HEADER_BG, edgecolor="none", zorder=1))
    hy = (header_y0 + header_y1) / 2
    ax.text(x_system, hy, "System", ha="left", va="center",
            fontsize=11.5, fontweight="bold", color=INK, zorder=3)
    for key, (_, xr) in cols.items():
        ax.text(xr, hy, headers[key], ha="right", va="center",
                fontsize=11.5, fontweight="bold", color=INK, zorder=3)

    # Top + bottom rules of the table
    ax.plot([0, 1], [header_y1, header_y1], color=INK, linewidth=1.3, zorder=2)
    ax.plot([0, 1], [header_y0, header_y0], color=INK, linewidth=1.0, zorder=2)
    ax.plot([0, 1], [row_bands[-1][0], row_bands[-1][0]], color=INK,
            linewidth=1.3, zorder=2)

    for i, (y0, y1) in enumerate(row_bands):
        yc = (y0 + y1) / 2
        row = data.iloc[i]
        sys_color = SYSTEM_COLORS[row["system"]]

        # Left colour swatch tying the row to its Figure 4 bar
        ax.add_patch(plt.Rectangle((x_swatch[0], y0 + 0.018),
                                   x_swatch[1] - x_swatch[0], (y1 - y0) - 0.036,
                                   facecolor=sys_color, edgecolor="none", zorder=3))

        # System name
        ax.text(x_system, yc, row["system"], ha="left", va="center",
                fontsize=11, color=INK, zorder=3)

        # Numeric cells (right-aligned), best value highlighted + bold
        values = {"accuracy": f"{row['accuracy']*100:.1f}%",
                  "log_loss": f"{row['log_loss']:.4f}",
                  "brier":    f"{row['brier']:.4f}"}
        for key, (xl, xr) in cols.items():
            is_best = best[key] == i
            if is_best:
                ax.add_patch(plt.Rectangle((xl, y0 + 0.012), xr - xl + 0.02,
                                           (y1 - y0) - 0.024, facecolor=HIGHLIGHT,
                                           edgecolor="none", zorder=2))
            ax.text(xr, yc, values[key], ha="right", va="center",
                    fontsize=11, color=INK,
                    fontweight="bold" if is_best else "normal", zorder=4)

        # Light separator between data rows
        if i < len(row_bands) - 1:
            ax.plot([0, 1], [y0, y0], color=SEP, linewidth=0.8, zorder=2)

    # Footnote inside the image
    ax.text(0, row_bands[-1][0] - 0.10,
            "Accuracy higher is better; log-loss and Brier lower is better.  "
            "192 matches, WC 2014/2018/2022, leakage-free.",
            ha="left", va="top", fontsize=8, color=GREY, zorder=3)

    out = FIG_DIR / "figure3_backtest_table.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.rcParams["font.family"] = plt.rcParamsDefault["font.family"]
    print(f"Wrote {out}")
    print("  Combined-row values pulled from the metrics summary:")
    for _, r in data.iterrows():
        print(f"    {r['system']:<13} acc {r['accuracy']*100:5.1f}%   "
              f"log-loss {r['log_loss']:.4f}   Brier {r['brier']:.4f}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REPORTS_DIR / "backtest_predictions.csv")
    print(f"Loaded {len(df)} match rows from backtest_predictions.csv\n")
    figure3()
    print()
    figure4(df)
    print()
    figure5(df)


if __name__ == "__main__":
    main()
