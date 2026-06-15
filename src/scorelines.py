"""
Scoreline (exact-score) probability layer for WC 2026 predictions.

The base model outputs only Win / Draw / Loss probabilities (XGBoost + Elo).
This module adds an ADDITIVE scoreline layer on top WITHOUT changing those
numbers. For a fixture it solves for the expected-goals pair
(lambda_home, lambda_away) of a Dixon-Coles-corrected double-Poisson whose
score grid rolls up to the SAME W/D/L probabilities the card already shows,
then reads the most-likely scorelines off that grid. Consistency with the
headline W/D/L therefore holds *by construction* — there is no independent
scoreline model that could disagree with the displayed numbers.

Orientation: scorelines are home-away (grid row = home goals, col = away
goals), matching the Match Predictor card. Any home advantage is inherited
from the input W/D/L (which for neutral World Cup matches already carries no
home advantage), so nothing extra is added here.

The Dixon-Coles rho (low-score dependence) is a fixed constant fitted once by
max-likelihood on historical international scorelines — see fit_rho_from_data().
Its exact value only shapes the 0-0/1-0/0-1/1-1 cells; it does NOT affect the
W/D/L consistency, which is enforced by the lambda solve for any rho.

Usage:
    from scorelines import top_scorelines
    tops = top_scorelines(p_home, p_draw, p_away, top_n=5)
    # -> [((home_goals, away_goals), prob), ...] sorted high->low
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

# Fitted once by MLE on historical scorelines (see fit_rho_from_data()): ~-0.05,
# a mild low-score dependence typical of football. Only shapes the four lowest
# cells; does NOT affect W/D/L consistency (enforced by the lambda solve).
DIXON_COLES_RHO = -0.05
MAX_GOALS = 10          # grid covers 0..10 goals per side; tail beyond is tiny


# ── core distribution ───────────────────────────────────────────────────────

def dc_grid(lam_home: float, lam_away: float,
            rho: float = DIXON_COLES_RHO, max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Dixon-Coles-corrected double-Poisson scoreline grid, normalised to sum 1.
    grid[i, j] = P(home scores i, away scores j).
    """
    goals = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))
    # Dixon-Coles low-score correction (tau) on the four lowest cells
    grid[0, 0] *= 1.0 - lam_home * lam_away * rho
    grid[0, 1] *= 1.0 + lam_home * rho
    grid[1, 0] *= 1.0 + lam_away * rho
    grid[1, 1] *= 1.0 - rho
    grid = np.clip(grid, 0.0, None)
    total = grid.sum()
    return grid / total if total > 0 else grid


def grid_wdl(grid: np.ndarray) -> tuple[float, float, float]:
    """Roll a scoreline grid up to (home_win, draw, away_win) probabilities."""
    return (float(np.tril(grid, -1).sum()),   # home goals > away goals
            float(np.trace(grid)),            # equal
            float(np.triu(grid, 1).sum()))    # away goals > home goals


# ── consistency solve: find lambdas reproducing the model's W/D/L ───────────

def solve_lambdas(p_home: float, p_draw: float, p_away: float,
                  rho: float = DIXON_COLES_RHO) -> tuple[float, float]:
    """
    Find (lambda_home, lambda_away) whose Dixon-Coles grid rolls up to the given
    W/D/L probabilities. Optimised over log-lambdas (kept positive); robust via a
    small multistart. Returns the expected-goals pair.
    """
    target = np.array([p_home, p_draw, p_away], dtype=float)
    target = target / target.sum()

    def loss(x: np.ndarray) -> float:
        lam_h, lam_a = np.exp(x)
        wdl = np.array(grid_wdl(dc_grid(lam_h, lam_a, rho)))
        return float(np.sum((wdl - target) ** 2))

    # Heuristic starting total goals: fewer goals when draws are more likely.
    total0 = float(np.clip(3.3 - 3.0 * target[1], 1.4, 3.6))
    share_h = target[0] / max(target[0] + target[2], 1e-9)

    starts = [
        (total0 * (0.3 + 0.4 * share_h), total0 * (0.3 + 0.4 * (1 - share_h))),
        (total0 / 2.0, total0 / 2.0),
        (1.6, 1.0), (1.0, 1.6), (1.3, 1.3),
    ]
    best_x, best_loss = None, np.inf
    for lh, la in starts:
        x0 = np.log([max(lh, 0.05), max(la, 0.05)])
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 800})
        if res.fun < best_loss:
            best_loss, best_x = res.fun, res.x
        if best_loss < 1e-9:
            break
    lam_h, lam_a = np.exp(best_x)
    return float(lam_h), float(lam_a)


def scoreline_grid(p_home: float, p_draw: float, p_away: float,
                   rho: float = DIXON_COLES_RHO) -> tuple[np.ndarray, tuple[float, float]]:
    """Grid consistent with the given W/D/L, plus the solved (lambda_home, lambda_away)."""
    lam_h, lam_a = solve_lambdas(p_home, p_draw, p_away, rho)
    return dc_grid(lam_h, lam_a, rho), (lam_h, lam_a)


def top_scorelines(p_home: float, p_draw: float, p_away: float,
                   top_n: int = 5, rho: float = DIXON_COLES_RHO) -> list[tuple[tuple[int, int], float]]:
    """
    Top-N most likely scorelines for a fixture, oriented home-away.
    Returns [((home_goals, away_goals), prob), ...] highest first.
    """
    grid, _ = scoreline_grid(p_home, p_draw, p_away, rho)
    flat_order = np.argsort(grid.ravel())[::-1][:top_n]
    rows, cols = np.unravel_index(flat_order, grid.shape)
    return [((int(r), int(c)), float(grid[r, c])) for r, c in zip(rows, cols)]


# ── offline calibration of rho (not called at runtime) ──────────────────────

def fit_rho_from_data(results_csv: str = "data/raw/results.csv",
                      features_csv: str = "data/processed/features.csv") -> float:
    """
    Estimate the Dixon-Coles rho by max-likelihood on historical scorelines.

    Per-match expected goals are taken from two simple Poisson regressions of
    goals on the Elo gap (and a home-advantage term for non-neutral matches),
    then rho is chosen to maximise the DC tau likelihood on the four low-score
    cells. Run once; the result is hard-coded as DIXON_COLES_RHO above.
    """
    import pandas as pd
    from sklearn.linear_model import PoissonRegressor
    from scipy.optimize import minimize_scalar

    res = pd.read_csv(results_csv).dropna(subset=["home_score", "away_score"])
    feat = pd.read_csv(features_csv)[["date", "home_team", "away_team", "elo_diff"]]
    df = res.merge(feat, on=["date", "home_team", "away_team"], how="inner")
    hs = np.clip(df["home_score"].to_numpy(float), 0, 8)
    as_ = np.clip(df["away_score"].to_numpy(float), 0, 8)
    home_adv = (1 - df["neutral"].astype(int)).to_numpy(float)
    ed = df["elo_diff"].to_numpy(float) / 400.0

    reg_h = PoissonRegressor(alpha=1e-4, max_iter=400).fit(
        np.column_stack([ed, home_adv]), hs)
    reg_a = PoissonRegressor(alpha=1e-4, max_iter=400).fit(
        np.column_stack([-ed, np.zeros_like(home_adv)]), as_)
    lam_h = reg_h.predict(np.column_stack([ed, home_adv]))
    lam_a = reg_a.predict(np.column_stack([-ed, np.zeros_like(home_adv)]))

    # rho only enters the four low-score cells; maximise their tau log-likelihood.
    m00 = (hs == 0) & (as_ == 0)
    m01 = (hs == 0) & (as_ == 1)
    m10 = (hs == 1) & (as_ == 0)
    m11 = (hs == 1) & (as_ == 1)

    def neg_ll(rho: float) -> float:
        eps = 1e-9
        ll = 0.0
        ll += np.log(np.clip(1 - lam_h[m00] * lam_a[m00] * rho, eps, None)).sum()
        ll += np.log(np.clip(1 + lam_h[m01] * rho, eps, None)).sum()
        ll += np.log(np.clip(1 + lam_a[m10] * rho, eps, None)).sum()
        ll += np.log(np.clip(1 - rho, eps, None)) * int(m11.sum())
        return -ll

    out = minimize_scalar(neg_ll, bounds=(-0.3, 0.1), method="bounded")
    return float(out.x)


if __name__ == "__main__":
    rho = fit_rho_from_data()
    print(f"Fitted Dixon-Coles rho = {rho:.4f}  (hard-coded value: {DIXON_COLES_RHO})")
