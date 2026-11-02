"""
Article figures 3, 4, 6 and 10 in the website figure contract (DEC-121).

Reads  reports/backtest_predictions.csv     (written by src/export_backtest.py)
       reports/backtest_metrics_summary.csv  (combined rows -> Figure 3 table)
       reports/live_evaluation.csv           (written by live_evaluation --final)
Writes reports/figures/<name>-light.png / <name>-dark.png (+ .svg) for
       figure3_backtest_table, figure4_accuracy_wilson,
       figure6_blend_reliability, figure10_live_evaluation

Website contract: each figure is rendered once per theme, 1600 px wide, on a
transparent background, with the site's text, soft and rule inks and the
site's three series colours (amir42-web scripts/editorial-art/figures.ts,
THEME_INK). Labels are about 48 px and notes about 40 px at that width.
Colour is never the only carrier of meaning: every series also has a hatch
or marker shape, a legend and direct labels. Titles and findings belong to
the article caption, not the image. SVGs keep text as text.

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
from matplotlib import font_manager

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIG_DIR = REPORTS_DIR / "figures"

# ── website figure contract ─────────────────────────────────────────────────
FIGURE_WIDTH_PX = 1600
DPI = 200                                   # 8 in x 200 dpi = 1600 px
PX = 72.0 / DPI                             # one figure pixel in points
LABEL_PT, NOTE_PT, FIGURE_PT = 48 * PX, 40 * PX, 44 * PX

THEMES = {
    "light": {"text": "#211d1a", "soft": "#5f5a54", "rule": "#ccc6bb",
              "series": ("#3b6fb0", "#c76d3a", "#4f8f6a")},
    "dark":  {"text": "#e7e3dd", "soft": "#bab5ad", "rule": "#4b4843",
              "series": ("#6f9bd8", "#e0905f", "#7ab894")},
}

# Fixed system order -> series slot, the same meaning in every figure.
SYSTEMS = ("Raw XGBoost", "Blend", "Elo baseline")
SYSTEM_SLOT = {name: i for i, name in enumerate(SYSTEMS)}
# Secondary encoding so identity never rests on colour alone.
SYSTEM_HATCH = {"Raw XGBoost": "//", "Blend": "", "Elo baseline": ".."}
SYSTEM_MARKER = {"Raw XGBoost": "o", "Blend": "D", "Elo baseline": "s"}
OUTCOMES = (("home", "Home win", "o"), ("draw", "Draw", "s"), ("away", "Away win", "^"))


def _font_family() -> str:
    """Inter when installed (the website's face), else DejaVu Sans."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Inter" if "Inter" in names else "DejaVu Sans"


FONT = _font_family()


def system_color(theme: dict, system: str) -> str:
    return theme["series"][SYSTEM_SLOT[system]]


def _style(theme: dict) -> None:
    plt.rcParams.update({
        "font.family": FONT, "font.size": NOTE_PT,
        "text.color": theme["text"], "axes.labelcolor": theme["text"],
        "axes.edgecolor": theme["rule"], "axes.labelsize": LABEL_PT,
        "xtick.color": theme["soft"], "ytick.color": theme["soft"],
        "xtick.labelcolor": theme["text"], "ytick.labelcolor": theme["soft"],
        "xtick.labelsize": LABEL_PT, "ytick.labelsize": NOTE_PT,
        "axes.grid": False, "grid.color": theme["rule"], "grid.linewidth": 1.0,
        "hatch.color": theme["text"], "hatch.linewidth": 1.2,
        "legend.frameon": False, "legend.fontsize": NOTE_PT,
        "svg.fonttype": "none", "svg.hashsalt": "wc2026-figures",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _clean_axes(ax, theme: dict, grid_axis: str = "y") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color=theme["rule"], linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["rule"])
    ax.tick_params(length=0, pad=8)


def render_pair(name: str, draw, height_px: int) -> list[Path]:
    """Render `draw(fig, theme)` once per theme at 1600 px wide on a
    transparent background; write <name>-<theme>.png and .svg."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for theme_name, theme in THEMES.items():
        with plt.rc_context():
            _style(theme)
            fig = plt.figure(figsize=(FIGURE_WIDTH_PX / DPI, height_px / DPI),
                             dpi=DPI, layout="constrained")
            fig.get_layout_engine().set(w_pad=24 / DPI, h_pad=24 / DPI)
            draw(fig, theme)
            for ext, meta in (("png", {"Software": None}), ("svg", {"Date": None})):
                out = FIG_DIR / f"{name}-{theme_name}.{ext}"
                fig.savefig(out, dpi=DPI, transparent=True, metadata=meta)
                written.append(out)
            plt.close(fig)
    print("Wrote " + ", ".join(p.name for p in written))
    return written


# --------------------------------------------------------------------------- #
# Reusable stats helpers
# --------------------------------------------------------------------------- #
def _fmt_pct(p: float) -> str:
    """
    Percentage with one decimal — except when the value sits exactly on the
    rounding boundary (108/192 = 56.25%), where Python's round-half-even
    would render a misleading 56.2%; show both decimals instead.
    """
    pct = p * 100.0
    if abs((pct * 10) % 1 - 0.5) < 1e-9:
        return f"{pct:.2f}%"
    return f"{pct:.1f}%"


def _years_label(tournaments) -> str:
    """Derive '2014/2018/2022' from 'WC <year>' values — never hard-coded."""
    years = sorted({int(str(t).split()[-1]) for t in tournaments})
    return "/".join(str(y) for y in years)


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
    contributes n samples), matching the reliability curves in figure 6.

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
# Shared bar and legend helpers
# --------------------------------------------------------------------------- #
def _bar(ax, theme: dict, x, height, system: str, width: float, **kw):
    """One system's bars: its series colour and its hatch."""
    return ax.bar(x, height, width=width, color=system_color(theme, system),
                  hatch=SYSTEM_HATCH[system], edgecolor=theme["text"],
                  linewidth=0, zorder=2, label=system, **kw)


def _legend(ax, theme: dict, systems) -> None:
    """Legend above the axes, swatches carrying the same hatch as the bars."""
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=system_color(theme, s),
                             hatch=SYSTEM_HATCH[s], edgecolor=theme["text"],
                             linewidth=0) for s in systems]
    ax.legend(handles, systems, loc="lower left", bbox_to_anchor=(0, 1.02),
              ncol=len(systems), fontsize=LABEL_PT, handlelength=1.6,
              handleheight=1.0, columnspacing=1.6, borderaxespad=0)


# --------------------------------------------------------------------------- #
# Figure 3: combined backtest comparison table
# --------------------------------------------------------------------------- #
def _combined_metrics() -> tuple[pd.DataFrame, int, str]:
    """
    Pull the COMBINED (192-match) row for each system from the metrics summary,
    so the table can never drift from the data. Returns (rows in fixed order
    (Raw XGBoost, Blend, Elo baseline) with display names, match count,
    'YYYY/YYYY/YYYY' tournament label), all derived from the CSV.
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
    n = int(comb["n"].iloc[0])
    years = _years_label(t for t in summary["tournament"]
                         if str(t).startswith("WC "))
    return pd.DataFrame(rows), n, years


def figure3() -> None:
    data, n_matches, years = _combined_metrics()
    best = {"accuracy": int(data["accuracy"].idxmax()),
            "log_loss": int(data["log_loss"].idxmin()),
            "brier":    int(data["brier"].idxmin())}
    cols = {"accuracy": ("Accuracy", 1010), "log_loss": ("Log-loss", 1300),
            "brier": ("Brier", 1560)}            # header, right edge (px)
    header_h, row_h, top = 120, 130, 30
    height = top + header_h + row_h * len(data) + 250

    def draw(fig, theme):
        fig.set_layout_engine("none")             # one manual full-bleed axes
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, FIGURE_WIDTH_PX)
        ax.set_ylim(height, 0)                    # pixel coordinates, y down
        ax.axis("off")
        ax.plot([40, 1560], [top, top], color=theme["text"], linewidth=1.6)
        hy = top + header_h / 2
        ax.text(150, hy, "System", ha="left", va="center", fontsize=LABEL_PT,
                fontweight="bold")
        for label, xr in cols.values():
            ax.text(xr, hy, label, ha="right", va="center", fontsize=LABEL_PT,
                    fontweight="bold")
        y = top + header_h
        ax.plot([40, 1560], [y, y], color=theme["text"], linewidth=1.0)
        for i, row in data.iterrows():
            yc = y + row_h / 2
            ax.add_patch(plt.Rectangle((40, yc - 34), 80, 68,
                                       facecolor=system_color(theme, row["system"]),
                                       hatch=SYSTEM_HATCH[row["system"]],
                                       edgecolor=theme["text"], linewidth=0))
            ax.text(150, yc, row["system"], ha="left", va="center",
                    fontsize=LABEL_PT)
            values = {"accuracy": _fmt_pct(row["accuracy"]),
                      "log_loss": f"{row['log_loss']:.4f}",
                      "brier": f"{row['brier']:.4f}"}
            for key, (_, xr) in cols.items():
                ax.text(xr, yc, values[key], ha="right", va="center",
                        fontsize=FIGURE_PT,
                        fontweight="bold" if best[key] == i else "normal")
            y += row_h
            last = i == len(data) - 1
            ax.plot([40, 1560], [y, y], color=theme["text"] if last else theme["rule"],
                    linewidth=1.6 if last else 1.0)
        for k, note in enumerate((
                "Bold: best in each column. Higher accuracy is better;",
                "lower log-loss and Brier are better.",
                f"{n_matches} matches, WC {years}, predicted before each tournament.")):
            ax.text(40, y + 40 + 62 * k, note, ha="left", va="top",
                    fontsize=NOTE_PT, color=theme["soft"])

    render_pair("figure3_backtest_table", draw, height)
    print("  Combined-row values pulled from the metrics summary:")
    for _, r in data.iterrows():
        print(f"    {r['system']:<13} acc {_fmt_pct(r['accuracy']):>7}   "
              f"log-loss {r['log_loss']:.4f}   Brier {r['brier']:.4f}")


# --------------------------------------------------------------------------- #
# Figure 4: accuracy with Wilson intervals
# --------------------------------------------------------------------------- #
def figure4(df: pd.DataFrame) -> None:
    n = len(df)
    y = df["y_true"].values
    specs = [("Raw XGBoost", ["xgb_home", "xgb_draw", "xgb_away"]),
             ("Blend", ["blend_home", "blend_draw", "blend_away"]),
             ("Elo baseline", ["elo_home", "elo_draw", "elo_away"])]
    rows = []
    for system, cols in specs:
        k = int((df[cols].values.argmax(axis=1) == y).sum())
        lo, hi = wilson_interval(k, n)
        rows.append((system, k, k / n, lo, hi))

    def draw(fig, theme):
        ax = fig.add_subplot()
        _clean_axes(ax, theme)
        x = np.arange(len(rows))
        for xi, (system, k, acc, lo, hi) in zip(x, rows):
            _bar(ax, theme, xi, acc, system, width=0.56)
            ax.errorbar(xi, acc, yerr=[[acc - lo], [hi - acc]], fmt="none",
                        ecolor=theme["text"], elinewidth=2.4, capsize=18,
                        capthick=2.4, zorder=3)
            ax.text(xi, hi + 0.048, _fmt_pct(acc), ha="center", va="bottom",
                    fontsize=FIGURE_PT, fontweight="bold")
            ax.text(xi, hi + 0.012, f"{k}/{n}  [{lo*100:.1f}, {hi*100:.1f}]",
                    ha="center", va="bottom", fontsize=NOTE_PT, color=theme["soft"])
        ax.set_ylim(0.30, 0.76)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.set_ylabel("Accuracy")
        ax.set_xlabel(f"Error bars and brackets: 95% Wilson intervals.\n"
                      f"{n} matches, WC {_years_label(df['tournament'])}.",
                      fontsize=NOTE_PT, color=theme["soft"], labelpad=24)

    render_pair("figure4_accuracy_wilson", draw, 1150)
    print("  Accuracy (k/n) with 95% Wilson intervals:")
    for system, k, acc, lo, hi in rows:
        print(f"    {system:<12} {k:>3}/{n}  {_fmt_pct(acc):>7}  "
              f"[{lo*100:.1f}%, {hi*100:.1f}%]")


# --------------------------------------------------------------------------- #
# Figure 6: Blend reliability (one-vs-rest per class)
# --------------------------------------------------------------------------- #
def figure6(df: pd.DataFrame) -> None:
    y = df["y_true"].values
    proba = df[["blend_home", "blend_draw", "blend_away"]].values
    overall_ece, per_class = multiclass_ece(y, proba, n_bins=10, strategy="quantile")

    def draw(fig, theme):
        ax = fig.add_subplot()
        _clean_axes(ax, theme, grid_axis="both")
        ax.plot([0, 1], [0, 1], linestyle=(0, (6, 6)), color=theme["soft"],
                linewidth=2.0, zorder=1, label="Perfect calibration")
        for c, (_, label, marker) in enumerate(OUTCOMES):
            hit = (y == c).astype(float)
            xs, ys, ns = _reliability_points(hit, proba[:, c], n_bins=10)
            color = theme["series"][c]
            ax.plot(xs, ys, "-", color=color, linewidth=2.4, zorder=2)
            ax.scatter(xs, ys, s=120 + (ns / ns.max()) * 380, marker=marker,
                       color=color, linewidth=0, zorder=3,
                       label=f"{label} (ECE {per_class[c]*100:.1f}%)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.legend(loc="upper left", fontsize=NOTE_PT, markerscale=0.8,
                  handletextpad=0.4, labelspacing=0.5)
        ax.text(0.98, 0.02, f"Blend, {len(y)} matches. 10 equal-frequency\n"
                f"bins per outcome; marker area shows\nbin size. "
                f"Overall ECE {overall_ece*100:.1f}%.",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=NOTE_PT, color=theme["soft"])

    render_pair("figure6_blend_reliability", draw, 1600)
    print("  Blend reliability, Expected Calibration Error:")
    for c, (key, _, _) in enumerate(OUTCOMES):
        print(f"    {key:<6} ECE {per_class[c]*100:.2f}%")
    print(f"    overall ECE {overall_ece*100:.2f}%")


# --------------------------------------------------------------------------- #
# Figure 10: final live evaluation (WC 2026, 104 matches)
# --------------------------------------------------------------------------- #
LIVE_SCOPES = (("pooled", "All matches"),
               ("prospective", "Logged before kickoff"),
               ("retrospective", "Recomputed after"))
# Two-line tick labels so the three groups never collide at 1600 px.
LIVE_TICKS = {"pooled": "All\nmatches", "prospective": "Logged before\nkickoff",
              "retrospective": "Recomputed\nafter"}


def _live_metrics() -> dict:
    """{scope: {"n", "blend", "elo", "diff"}} from reports/live_evaluation.csv;
    diff is (mean, ci_low, ci_high) of Blend minus Elo log-loss, or None where
    the paired tests are not applicable."""
    import re

    df = pd.read_csv(REPORTS_DIR / "live_evaluation.csv")
    out = {}
    for scope, _ in LIVE_SCOPES:
        sub = df[df["scope"] == scope].set_index("system")
        boot = sub.loc["Bootstrap logloss diff (Blend-Elo)"]
        diff = None
        if pd.notna(boot["mean_diff"]):
            lo, hi = map(float, re.search(r"\[([-\d.]+), ([-\d.]+)\]",
                                          boot["detail"]).groups())
            diff = (float(boot["mean_diff"]), lo, hi)
        out[scope] = {"n": int(sub.loc["Blend", "n"]), "blend": sub.loc["Blend"],
                      "elo": sub.loc["Elo baseline"], "diff": diff}
    return out


def figure10() -> None:
    m = _live_metrics()
    systems = ("Blend", "Elo baseline")
    keys = {"Blend": "blend", "Elo baseline": "elo"}
    labels = [f"{LIVE_TICKS[s]}\nn = {m[s]['n']}" for s, _ in LIVE_SCOPES]
    x = np.arange(len(LIVE_SCOPES))
    w = 0.36

    def draw(fig, theme):
        ax_acc, ax_ll, ax_d = fig.subplots(3, 1, height_ratios=(1.1, 1.0, 0.85))
        for ax in (ax_acc, ax_ll):
            _clean_axes(ax, theme)
        for i, system in enumerate(systems):
            off = (i - 0.5) * (w + 0.02)
            for xi, (scope, _) in zip(x, LIVE_SCOPES):
                r = m[scope][keys[system]]
                acc, ll = float(r["accuracy"]), float(r["log_loss"])
                _bar(ax_acc, theme, xi + off, acc, system, width=w)
                ax_acc.text(xi + off, acc + 0.02, f"{int(r['correct'])}",
                            ha="center", va="bottom", fontsize=NOTE_PT)
                _bar(ax_ll, theme, xi + off, ll, system, width=w)
                ax_ll.text(xi + off, ll + 0.015, f"{ll:.3f}", ha="center",
                           va="bottom", fontsize=NOTE_PT)
        _legend(ax_acc, theme, systems)
        ax_acc.set_ylim(0, 1.0)
        ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax_acc.set_ylabel("Accuracy (correct picks)")
        ax_ll.set_ylim(0, 1.15)
        ax_ll.set_ylabel("Log-loss")
        for ax in (ax_acc, ax_ll):
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=NOTE_PT)

        # Panel 3: paired bootstrap interval of Blend minus Elo log-loss.
        _clean_axes(ax_d, theme, grid_axis="x")
        ax_d.axvline(0, color=theme["soft"], linewidth=2.0, zorder=1)
        ys = np.arange(len(LIVE_SCOPES))[::-1]
        for yi, (scope, _) in zip(ys, LIVE_SCOPES):
            d = m[scope]["diff"]
            if d is None:
                ax_d.text(0.006, yi, f"not tested: {m[scope]['n']} matches",
                          ha="left", va="center", fontsize=NOTE_PT,
                          color=theme["soft"])
                continue
            mean, lo, hi = d
            for xs_, ys_ in (([lo, hi], [yi, yi]), ([lo, lo], [yi - 0.2, yi + 0.2]),
                             ([hi, hi], [yi - 0.2, yi + 0.2])):
                ax_d.plot(xs_, ys_, color=theme["text"], linewidth=3.0, zorder=2)
            ax_d.plot(mean, yi, "D", markersize=16, zorder=3,
                      color=system_color(theme, "Blend"))
            ax_d.text(0.006, yi, f"{mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
                      ha="left", va="center", fontsize=NOTE_PT)
        ax_d.set_yticks(ys)
        ax_d.set_yticklabels([name for _, name in LIVE_SCOPES], fontsize=NOTE_PT,
                             color=theme["text"])
        ax_d.set_ylim(-0.6, len(LIVE_SCOPES) - 0.4)
        ax_d.set_xlim(-0.075, 0.125)              # room right of zero for values
        ax_d.set_xticks([-0.06, -0.03, 0.0])
        ax_d.tick_params(axis="x", labelsize=NOTE_PT)
        ax_d.spines["left"].set_visible(False)
        ax_d.set_xlabel("Blend minus Elo log-loss; below zero: Blend better.\n"
                        "95% paired bootstrap interval, 10,000 resamples.",
                        fontsize=NOTE_PT, color=theme["soft"], labelpad=16)

    render_pair("figure10_live_evaluation", draw, 2300)
    for scope, name in LIVE_SCOPES:
        b, e, d = m[scope]["blend"], m[scope]["elo"], m[scope]["diff"]
        print(f"    {name:<22} n={m[scope]['n']:<3} Blend {int(b['correct'])} "
              f"ll {float(b['log_loss']):.4f} | Elo {int(e['correct'])} "
              f"ll {float(e['log_loss']):.4f} | diff {d}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REPORTS_DIR / "backtest_predictions.csv")
    print(f"Loaded {len(df)} match rows from backtest_predictions.csv\n")
    figure3()
    print()
    figure4(df)
    print()
    figure6(df)
    print()
    figure10()


if __name__ == "__main__":
    main()
