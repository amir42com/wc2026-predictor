"""
Experiments to beat the naive Elo baseline on World Cup matches.

Evaluation harness mirrors src/backtest.py: walk-forward over WC 2014/2018/2022,
every model (and any ensemble weight / calibrator) fitted only on data strictly
before the tournament being evaluated.  Ensemble weights are tuned on WC
2006 + 2010 only (pre-2014, as required).

Usage:
    python src/experiments.py e1 e2 e3 e4 e5     # run selected experiments
    python src/experiments.py all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss

from train import NUMERIC_COLS, make_X, train_model
from backtest import brier_multiclass, elo_baseline_proba
from features import load_canonical_results

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
CACHE_DIR     = PROCESSED_DIR / "exp_cache"

TUNE_YEARS = [2006, 2010]   # for ensemble-weight tuning only (pre-2014)
EVAL_YEARS = [2014, 2018, 2022]

# Extra feature columns added by the extended features.py (experiment 3).
EXTRA_COLS = [
    "home_days_since_last", "away_days_since_last",
    "home_comp_wr_10", "away_comp_wr_10",
    "home_qual_wr", "away_qual_wr",
    "home_tourn_match_n", "away_tourn_match_n",
]


# ── data / split helpers ───────────────────────────────────────────────────

def load_df() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    # Scores come from the ONE canonical load path (features.py owns the
    # duplicate policy). The kept 1974 Tahiti double-header makes the
    # (date, home, away) key non-unique, so align same-key rows by their
    # per-key rank in canonical sort order instead of dropping one.
    raw = load_canonical_results()
    key = ["date", "home_team", "away_team"]
    raw = raw.assign(_k=raw.groupby(key).cumcount())
    df = df.assign(_k=df.groupby(key).cumcount()).merge(
        raw[key + ["_k", "home_score", "away_score"]],
        on=key + ["_k"], how="left",
    ).drop(columns="_k")
    return df


def wc_split(df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(df_pre, df_wc): everything strictly before the tournament's first match."""
    wc_mask = (df["is_world_cup"] == 1) & (df["date"].dt.year == year)
    df_wc = df[wc_mask].reset_index(drop=True)
    cutoff = df_wc["date"].min()
    df_pre = df[df["date"] < cutoff].reset_index(drop=True)
    return df_pre, df_wc


def metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    proba = proba / proba.sum(axis=1, keepdims=True)
    return {
        "acc":   accuracy_score(y, proba.argmax(axis=1)),
        "ll":    log_loss(y, proba, labels=[0, 1, 2]),
        "brier": brier_multiclass(y, proba),
    }


# ── cached XGB probabilities ───────────────────────────────────────────────

def xgb_proba(df: pd.DataFrame, year: int, tag: str,
              train_mask_fn=None, extra_cols: list[str] | None = None) -> np.ndarray:
    """
    Leakage-free XGB probabilities for `year`'s WC matches, cached on disk.
    train_mask_fn(df_pre) -> bool mask restricts training rows.
    extra_cols extends NUMERIC_COLS (monkey-patched around train_model/make_X).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{tag}_{year}.npy"
    if cache.exists():
        return np.load(cache)

    df_pre, df_wc = wc_split(df, year)
    if train_mask_fn is not None:
        df_pre = df_pre[train_mask_fn(df_pre)].reset_index(drop=True)

    import train as _train
    saved = _train.NUMERIC_COLS
    if extra_cols:
        _train.NUMERIC_COLS = saved + extra_cols
    try:
        print(f"    [{tag}] WC {year}: training on {len(df_pre):,} matches ...")
        model, fc = train_model(df_pre)
        X_wc, _ = make_X(df_wc, fc)
    finally:
        _train.NUMERIC_COLS = saved

    proba = model.predict_proba(X_wc)
    np.save(cache, proba)
    return proba


def baseline_proba(df: pd.DataFrame, year: int) -> np.ndarray:
    df_pre, df_wc = wc_split(df, year)
    draw_rate = float((df_pre["outcome"] == 1).mean())
    return elo_baseline_proba(df_wc, draw_rate)


# ── reporting ──────────────────────────────────────────────────────────────

def report(df: pd.DataFrame, name: str, proba_fn) -> dict:
    """proba_fn(year) -> proba for that year's WC. Prints table, returns combined."""
    print(f"\n  {name}")
    print(f"  {'Tournament':<12}{'N':>4} | {'Acc':>7} {'LogLoss':>8} {'Brier':>7} |"
          f" {'BaseAcc':>8} {'BaseLL':>7}")
    print("  " + "-" * 62)
    ys, ps, bs = [], [], []
    for year in EVAL_YEARS:
        _, df_wc = wc_split(df, year)
        y = df_wc["outcome"].values
        p = proba_fn(year)
        b = baseline_proba(df, year)
        m, mb = metrics(y, p), metrics(y, b)
        ys.append(y); ps.append(p); bs.append(b)
        print(f"  WC {year:<9}{len(y):>4} | {m['acc']*100:>6.1f}% {m['ll']:>8.4f}"
              f" {m['brier']:>7.4f} | {mb['acc']*100:>7.1f}% {mb['ll']:>7.4f}")
    y_all = np.concatenate(ys)
    m  = metrics(y_all, np.vstack(ps))
    mb = metrics(y_all, np.vstack(bs))
    print("  " + "-" * 62)
    print(f"  {'Combined':<12}{len(y_all):>4} | {m['acc']*100:>6.1f}% {m['ll']:>8.4f}"
          f" {m['brier']:>7.4f} | {mb['acc']*100:>7.1f}% {mb['ll']:>7.4f}")
    return m


# ── experiment 1: specialized model ────────────────────────────────────────

def e1(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("E1: specialized training set")
    print("=" * 70)

    report(df, "E1a: trained on neutral OR non-friendly matches",
           lambda yr: xgb_proba(df, yr, "e1a",
               train_mask_fn=lambda d: (d["neutral"] == 1) | (d["tournament"] != "Friendly")))

    report(df, "E1b: trained on neutral-venue matches only",
           lambda yr: xgb_proba(df, yr, "e1b",
               train_mask_fn=lambda d: d["neutral"] == 1))


# ── experiment 2: XGB + Elo ensemble ───────────────────────────────────────

def tune_weight(df: pd.DataFrame, tag: str, **xgb_kwargs) -> float:
    """Grid-search blend weight w (XGB share) minimizing log-loss on WC 2006+2010."""
    ys, xs, bs = [], [], []
    for year in TUNE_YEARS:
        _, df_wc = wc_split(df, year)
        ys.append(df_wc["outcome"].values)
        xs.append(xgb_proba(df, year, tag, **xgb_kwargs))
        bs.append(baseline_proba(df, year))
    y, x, b = np.concatenate(ys), np.vstack(xs), np.vstack(bs)

    grid = np.linspace(0, 1, 21)
    losses = [log_loss(y, w * x + (1 - w) * b, labels=[0, 1, 2]) for w in grid]
    w = float(grid[int(np.argmin(losses))])
    print(f"    tuned w on WC {TUNE_YEARS}: XGB weight = {w:.2f} "
          f"(log-loss {min(losses):.4f})")
    return w


def e2(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("E2: ensemble XGB + Elo baseline (weight tuned on WC 2006/2010)")
    print("=" * 70)
    w = tune_weight(df, "base")
    report(df, f"E2: {w:.2f}*XGB + {1-w:.2f}*Elo",
           lambda yr: w * xgb_proba(df, yr, "base") + (1 - w) * baseline_proba(df, yr))


# ── experiment 3: extended features ────────────────────────────────────────

def e3(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("E3: extended features (rest days, competitive form, qual record, stage)")
    print("=" * 70)
    missing = [c for c in EXTRA_COLS if c not in df.columns]
    if missing:
        print(f"  features.csv is missing {missing} - rebuild with: python src/features.py")
        return
    report(df, "E3: base + extended features",
           lambda yr: xgb_proba(df, yr, "e3", extra_cols=EXTRA_COLS))


# ── experiment 4: Poisson goal model blend ─────────────────────────────────

def poisson_proba(df: pd.DataFrame, year: int) -> np.ndarray:
    """
    Independent-Poisson goal model.  One PoissonRegressor on team-perspective
    rows: goals_for ~ elo_diff + home_advantage, fitted on pre-tournament data.
    W/D/L probabilities from the 0-10 scoreline grid.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"poisson_{year}.npy"
    if cache.exists():
        return np.load(cache)

    df_pre, df_wc = wc_split(df, year)
    df_pre = df_pre.dropna(subset=["home_score", "away_score"])

    # long format: two rows per match (home and away perspective)
    home_adv = (1 - df_pre["neutral"].values).astype(float)
    X_h = np.column_stack([df_pre["elo_diff"].values / 400.0,  home_adv])
    X_a = np.column_stack([-df_pre["elo_diff"].values / 400.0, np.zeros(len(df_pre))])
    X = np.vstack([X_h, X_a])
    g = np.concatenate([df_pre["home_score"].values, df_pre["away_score"].values])
    g = np.clip(g, 0, 8)  # tame 31-0 freak results

    pr = PoissonRegressor(alpha=1e-4, max_iter=300)
    pr.fit(X, g)

    wc_home_adv = (1 - df_wc["neutral"].values).astype(float)
    lam_h = pr.predict(np.column_stack([df_wc["elo_diff"].values / 400.0, wc_home_adv]))
    lam_a = pr.predict(np.column_stack([-df_wc["elo_diff"].values / 400.0,
                                        np.zeros(len(df_wc))]))

    from scipy.stats import poisson
    goals = np.arange(0, 11)
    out = np.zeros((len(df_wc), 3))
    for i, (lh, la) in enumerate(zip(lam_h, lam_a)):
        ph = poisson.pmf(goals, lh)
        pa = poisson.pmf(goals, la)
        grid = np.outer(ph, pa)
        grid /= grid.sum()
        out[i] = [np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()]

    np.save(cache, out)
    return out


def e4(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("E4: Poisson goal model + XGB blend (weight tuned on WC 2006/2010)")
    print("=" * 70)

    report(df, "E4a: Poisson model alone", lambda yr: poisson_proba(df, yr))

    # tune XGB-vs-Poisson blend on pre-2014 WCs
    ys, xs, ps = [], [], []
    for year in TUNE_YEARS:
        _, df_wc = wc_split(df, year)
        ys.append(df_wc["outcome"].values)
        xs.append(xgb_proba(df, year, "base"))
        ps.append(poisson_proba(df, year))
    y, x, p = np.concatenate(ys), np.vstack(xs), np.vstack(ps)
    grid = np.linspace(0, 1, 21)
    losses = [log_loss(y, w * x + (1 - w) * p, labels=[0, 1, 2]) for w in grid]
    w = float(grid[int(np.argmin(losses))])
    print(f"    tuned w on WC {TUNE_YEARS}: XGB weight = {w:.2f}")

    report(df, f"E4b: {w:.2f}*XGB + {1-w:.2f}*Poisson",
           lambda yr: w * xgb_proba(df, yr, "base") + (1 - w) * poisson_proba(df, yr))


# ── experiment 5: isotonic calibration ─────────────────────────────────────

def calibrated_proba(df: pd.DataFrame, year: int, cal_n: int = 4000) -> np.ndarray:
    """
    Train on df_pre[:-cal_n], fit per-class isotonic regression on the last
    cal_n pre-tournament matches, apply to WC matches, renormalize.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"e5_{year}.npy"
    if cache.exists():
        return np.load(cache)

    df_pre, df_wc = wc_split(df, year)
    df_fit = df_pre.iloc[:-cal_n].reset_index(drop=True)
    df_cal = df_pre.iloc[-cal_n:].reset_index(drop=True)

    print(f"    [e5] WC {year}: training on {len(df_fit):,}, "
          f"calibrating on {len(df_cal):,} ...")
    model, fc = train_model(df_fit)

    X_cal, _ = make_X(df_cal, fc)
    p_cal = model.predict_proba(X_cal)
    y_cal = df_cal["outcome"].values

    X_wc, _ = make_X(df_wc, fc)
    p_wc = model.predict_proba(X_wc)

    out = np.zeros_like(p_wc)
    for c in range(3):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
        iso.fit(p_cal[:, c], (y_cal == c).astype(float))
        out[:, c] = iso.predict(p_wc[:, c])
    out /= out.sum(axis=1, keepdims=True)

    np.save(cache, out)
    return out


def e5(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("E5: isotonic calibration (fitted on last 4,000 pre-tournament matches)")
    print("=" * 70)
    report(df, "E5: calibrated XGB", lambda yr: calibrated_proba(df, yr))


# ── experiment 6: 3-way blends of the E1-E5 winners ────────────────────────

def _gather(df: pd.DataFrame, years: list[int], fns: list) -> tuple[np.ndarray, list[np.ndarray]]:
    ys, stacks = [], [[] for _ in fns]
    for year in years:
        _, df_wc = wc_split(df, year)
        ys.append(df_wc["outcome"].values)
        for i, fn in enumerate(fns):
            stacks[i].append(fn(year))
    return np.concatenate(ys), [np.vstack(s) for s in stacks]


def tune_3way(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, pc: np.ndarray,
              step: float = 0.05) -> tuple[float, float]:
    """Grid-search simplex weights (wa, wb, 1-wa-wb) minimizing log-loss."""
    best, best_w = np.inf, (1.0, 0.0)
    for wa in np.arange(0, 1 + 1e-9, step):
        for wb in np.arange(0, 1 - wa + 1e-9, step):
            p = wa * pa + wb * pb + (1 - wa - wb) * pc
            ll = log_loss(y, p / p.sum(axis=1, keepdims=True), labels=[0, 1, 2])
            if ll < best:
                best, best_w = ll, (float(wa), float(wb))
    return best_w


def e6(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("E6: 3-way blends (weights tuned on WC 2006/2010)")
    print("=" * 70)

    e1b_fn  = lambda yr: xgb_proba(df, yr, "e1b",
                                   train_mask_fn=lambda d: d["neutral"] == 1)
    base_fn = lambda yr: xgb_proba(df, yr, "base")
    pois_fn = lambda yr: poisson_proba(df, yr)
    elo_fn  = lambda yr: baseline_proba(df, yr)

    for label, xgb_fn in [("base XGB", base_fn), ("neutral-only XGB", e1b_fn)]:
        y, (px, pp, pe) = _gather(df, TUNE_YEARS, [xgb_fn, pois_fn, elo_fn])
        wx, wp = tune_3way(y, px, pp, pe)
        we = 1 - wx - wp
        print(f"\n    [{label}] tuned weights: XGB={wx:.2f} Poisson={wp:.2f} Elo={we:.2f}")
        report(df, f"E6: {wx:.2f}*({label}) + {wp:.2f}*Poisson + {we:.2f}*Elo",
               lambda yr, _wx=wx, _wp=wp, _we=we, _fn=xgb_fn:
                   _wx * _fn(yr) + _wp * pois_fn(yr) + _we * elo_fn(yr))


# ── experiment 7: 4-way blend + draw boost ─────────────────────────────────

def _boost_draw(proba: np.ndarray, k: float) -> np.ndarray:
    p = proba.copy()
    p[:, 1] *= k
    return p / p.sum(axis=1, keepdims=True)


def e7(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("E7: 4-way blend + draw boost (all tuned on WC 2006/2010)")
    print("=" * 70)

    fns = [
        lambda yr: xgb_proba(df, yr, "e1b", train_mask_fn=lambda d: d["neutral"] == 1),
        lambda yr: xgb_proba(df, yr, "base"),
        lambda yr: poisson_proba(df, yr),
        lambda yr: baseline_proba(df, yr),
    ]
    names = ["neutralXGB", "baseXGB", "Poisson", "Elo"]

    y, ps = _gather(df, TUNE_YEARS, fns)

    # coarse simplex grid over 4 weights (step .1), tuned by log-loss
    best, best_w = np.inf, None
    step = 0.1
    for w0 in np.arange(0, 1 + 1e-9, step):
        for w1 in np.arange(0, 1 - w0 + 1e-9, step):
            for w2 in np.arange(0, 1 - w0 - w1 + 1e-9, step):
                w3 = 1 - w0 - w1 - w2
                p = w0 * ps[0] + w1 * ps[1] + w2 * ps[2] + w3 * ps[3]
                ll = log_loss(y, p / p.sum(axis=1, keepdims=True), labels=[0, 1, 2])
                if ll < best:
                    best, best_w = ll, (w0, w1, w2, w3)
    w = best_w
    print(f"\n    tuned 4-way weights: "
          + " ".join(f"{n}={v:.1f}" for n, v in zip(names, w)))

    blend_fn = lambda yr: sum(wi * fn(yr) for wi, fn in zip(w, fns))
    report(df, "E7a: 4-way blend", blend_fn)

    # draw boost k tuned by accuracy (tie-break: log-loss) on 2006/2010
    p_tune = sum(wi * pi for wi, pi in zip(w, ps))
    best_k, best_key = 1.0, (-np.inf, np.inf)
    for k in np.arange(1.0, 3.01, 0.1):
        pb = _boost_draw(p_tune, k)
        key = (accuracy_score(y, pb.argmax(axis=1)),
               -log_loss(y, pb, labels=[0, 1, 2]))
        if key > best_key:
            best_key, best_k = key, float(k)
    print(f"\n    tuned draw boost k = {best_k:.1f} "
          f"(tune acc {best_key[0]*100:.1f}%, ll {-best_key[1]:.4f})")

    report(df, f"E7b: 4-way blend + draw boost k={best_k:.1f}",
           lambda yr: _boost_draw(blend_fn(yr), best_k))


# ── experiment 8: Dixon-Coles Poisson (time decay + rho correction) ────────

def _dc_tau_grid(grid: np.ndarray, lam: float, mu: float, rho: float) -> np.ndarray:
    """Apply the Dixon-Coles low-score correction to a scoreline grid."""
    g = grid.copy()
    g[0, 0] *= 1 - lam * mu * rho
    g[0, 1] *= 1 + lam * rho
    g[1, 0] *= 1 + mu * rho
    g[1, 1] *= 1 - rho
    return g


def dc_poisson_proba(df: pd.DataFrame, year: int,
                     half_life_years: float = 8.0) -> np.ndarray:
    """
    Dixon-Coles flavoured Poisson: time-decayed fit + rho correction for
    low-scoring dependence (rho fitted by max likelihood, pre-tournament only).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"dc_{year}.npy"
    if cache.exists():
        return np.load(cache)

    from scipy.stats import poisson

    df_pre, df_wc = wc_split(df, year)
    df_pre = df_pre.dropna(subset=["home_score", "away_score"])
    cutoff = df_wc["date"].min()

    age_years = (cutoff - df_pre["date"]).dt.days / 365.25
    w = 0.5 ** (age_years / half_life_years)

    home_adv = (1 - df_pre["neutral"].values).astype(float)
    X_h = np.column_stack([df_pre["elo_diff"].values / 400.0,  home_adv])
    X_a = np.column_stack([-df_pre["elo_diff"].values / 400.0, np.zeros(len(df_pre))])
    X = np.vstack([X_h, X_a])
    g = np.clip(np.concatenate([df_pre["home_score"], df_pre["away_score"]]), 0, 8)

    pr = PoissonRegressor(alpha=1e-4, max_iter=300)
    pr.fit(X, g, sample_weight=np.concatenate([w, w]))

    # fit rho by max weighted likelihood on recent (last 20y) pre-tournament data
    recent = df_pre[age_years < 20].reset_index(drop=True)
    r_adv = (1 - recent["neutral"].values).astype(float)
    lam = pr.predict(np.column_stack([recent["elo_diff"].values / 400.0, r_adv]))
    mu  = pr.predict(np.column_stack([-recent["elo_diff"].values / 400.0,
                                      np.zeros(len(recent))]))
    hs = np.clip(recent["home_score"].values, 0, 8).astype(int)
    as_ = np.clip(recent["away_score"].values, 0, 8).astype(int)
    base_ll_terms = poisson.logpmf(hs, lam) + poisson.logpmf(as_, mu)

    def tau_term(rho: float) -> float:
        t = np.ones(len(recent))
        m00 = (hs == 0) & (as_ == 0)
        m01 = (hs == 0) & (as_ == 1)
        m10 = (hs == 1) & (as_ == 0)
        m11 = (hs == 1) & (as_ == 1)
        t[m00] = 1 - lam[m00] * mu[m00] * rho
        t[m01] = 1 + lam[m01] * rho
        t[m10] = 1 + mu[m10] * rho
        t[m11] = 1 - rho
        return float(np.sum(np.log(np.clip(t, 1e-10, None)) + base_ll_terms))

    rhos = np.arange(-0.3, 0.11, 0.01)
    rho = float(rhos[int(np.argmax([tau_term(r) for r in rhos]))])
    print(f"    [dc] WC {year}: fitted rho = {rho:+.2f}")

    wc_adv = (1 - df_wc["neutral"].values).astype(float)
    lam_wc = pr.predict(np.column_stack([df_wc["elo_diff"].values / 400.0, wc_adv]))
    mu_wc  = pr.predict(np.column_stack([-df_wc["elo_diff"].values / 400.0,
                                         np.zeros(len(df_wc))]))

    goals = np.arange(0, 11)
    out = np.zeros((len(df_wc), 3))
    for i, (lh, la) in enumerate(zip(lam_wc, mu_wc)):
        grid = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))
        grid = _dc_tau_grid(grid, lh, la, rho)
        grid = np.clip(grid, 0, None)
        grid /= grid.sum()
        out[i] = [np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()]

    np.save(cache, out)
    return out


def e8(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("E8: Dixon-Coles Poisson (time decay + rho), alone and blended")
    print("=" * 70)

    dc_fn  = lambda yr: dc_poisson_proba(df, yr)
    e1b_fn = lambda yr: xgb_proba(df, yr, "e1b",
                                  train_mask_fn=lambda d: d["neutral"] == 1)
    elo_fn = lambda yr: baseline_proba(df, yr)

    report(df, "E8a: DC-Poisson alone", dc_fn)

    y, (px, pp, pe) = _gather(df, TUNE_YEARS, [e1b_fn, dc_fn, elo_fn])
    wx, wp = tune_3way(y, px, pp, pe)
    we = 1 - wx - wp
    print(f"\n    tuned weights: neutralXGB={wx:.2f} DC-Poisson={wp:.2f} Elo={we:.2f}")
    report(df, f"E8b: {wx:.2f}*neutralXGB + {wp:.2f}*DC-Poisson + {we:.2f}*Elo",
           lambda yr: wx * e1b_fn(yr) + wp * dc_fn(yr) + we * elo_fn(yr))


# ── experiment 9: accuracy-targeted selection on the tune set ──────────────

def e9(df: pd.DataFrame) -> None:
    """
    Select the final configuration by TUNE-SET ACCURACY (log-loss tiebreak),
    since accuracy is the stated goal.  Search space: simplex blends of
    (neutral-only XGB, DC-Poisson, Elo baseline) x draw-boost k.
    All selection happens on WC 2006/2010 — strictly pre-2014.
    """
    print("\n" + "=" * 70)
    print("E9: accuracy-targeted blend selection (tuned on WC 2006/2010)")
    print("=" * 70)

    e1b_fn = lambda yr: xgb_proba(df, yr, "e1b",
                                  train_mask_fn=lambda d: d["neutral"] == 1)
    dc_fn  = lambda yr: dc_poisson_proba(df, yr)
    elo_fn = lambda yr: baseline_proba(df, yr)
    fns = [e1b_fn, dc_fn, elo_fn]

    y, (px, pd_, pe) = _gather(df, TUNE_YEARS, fns)

    best_key, best_cfg = (-np.inf, np.inf), None
    for wx in np.arange(0, 1 + 1e-9, 0.05):
        for wd in np.arange(0, 1 - wx + 1e-9, 0.05):
            we = 1 - wx - wd
            p0 = wx * px + wd * pd_ + we * pe
            for k in np.arange(1.0, 2.01, 0.1):
                p = _boost_draw(p0, float(k))
                key = (accuracy_score(y, p.argmax(axis=1)),
                       -log_loss(y, p, labels=[0, 1, 2]))
                if key > best_key:
                    best_key = key
                    best_cfg = (float(wx), float(wd), float(we), float(k))

    wx, wd, we, k = best_cfg
    print(f"\n    selected: neutralXGB={wx:.2f} DC-Poisson={wd:.2f} "
          f"Elo={we:.2f} drawboost={k:.1f}")
    print(f"    tune-set: acc {best_key[0]*100:.1f}%, log-loss {-best_key[1]:.4f}")

    report(df, f"E9: acc-selected blend (w=({wx:.2f},{wd:.2f},{we:.2f}), k={k:.1f})",
           lambda yr: _boost_draw(
               wx * e1b_fn(yr) + wd * dc_fn(yr) + we * elo_fn(yr), k))


# ── experiment 10: DC-Poisson + Elo (Elo-monotone family) ──────────────────

def e10(df: pd.DataFrame) -> None:
    """
    Both DC-Poisson and the Elo baseline pick the higher-Elo team, so any
    blend ties the baseline's accuracy by construction; the blend weight
    (tuned on WC 2006/2010 log-loss) only sharpens the probabilities.
    """
    print("\n" + "=" * 70)
    print("E10: DC-Poisson + Elo blend (accuracy-neutral, calibration-only)")
    print("=" * 70)

    dc_fn  = lambda yr: dc_poisson_proba(df, yr)
    elo_fn = lambda yr: baseline_proba(df, yr)

    y, (pd_, pe) = _gather(df, TUNE_YEARS, [dc_fn, elo_fn])
    grid = np.linspace(0, 1, 21)
    losses = [log_loss(y, w * pd_ + (1 - w) * pe, labels=[0, 1, 2]) for w in grid]
    w = float(grid[int(np.argmin(losses))])
    print(f"\n    tuned w on WC {TUNE_YEARS}: DC-Poisson weight = {w:.2f}")

    report(df, f"E10: {w:.2f}*DC-Poisson + {1-w:.2f}*Elo",
           lambda yr: w * dc_fn(yr) + (1 - w) * elo_fn(yr))


EXPERIMENTS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5,
               "e6": e6, "e7": e7, "e8": e8, "e9": e9, "e10": e10}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="+", choices=list(EXPERIMENTS) + ["all"])
    args = parser.parse_args()

    which = list(EXPERIMENTS) if "all" in args.which else args.which

    print("Loading data ...")
    df = load_df()
    print(f"  {len(df):,} matches")

    report(df, "Reference: base XGB (current model)",
           lambda yr: xgb_proba(df, yr, "base"))

    for name in which:
        EXPERIMENTS[name](df)


if __name__ == "__main__":
    main()
