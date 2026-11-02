"""
Blend-weight search on the TUNE folds (WC 2006, WC 2010) under the corrected
protocol (remediation Phase 3) — the artifact the article's selection story
refers to.

Protocol per fold (this is the Phase 4 target protocol, replicated here on the
tune folds only):
  * fresh model trained on canonical matches strictly before the tournament's
    first fixture (train.train_model: chronological 10% tail for early
    stopping, production hyperparameters);
  * ONE frozen pre-tournament team-state/H2H snapshot, built by re-running
    features.build_features on the canonical data truncated at the fold cutoff
    (post-last-match serving semantics, identical to simulate.Predictor's
    team_state.csv) — recorded in-tournament feature rows are NOT used, so no
    within-tournament state updates leak into any prediction;
  * venue contract per Phase 2: MatchContext from each match's recorded
    neutral flag — host rows (Germany 2006, South Africa 2010) run
    single-orientation with the HFA-shifted prior (train.HFA_ELO);
  * production Elo-logistic prior (train.ELO_DRAW_RATE, HFA on host rows).

The model output p_model and the prior are computed once per match; the sweep
w in [0, 1] step 0.01 blends them analytically (exactly what
predict_match_proba does for any fixed w).

Selection metric: POOLED log-loss over both tune folds (the original selection
story). Accuracy and Brier are recorded as secondary columns.

Output: reports/blend_weight_search.csv with header metadata (code SHA,
canonical-data hash, HFA, draw rate, date).

BOUNDARY: the 2014/2018/2022 benchmark folds are out of bounds here (Phase 4
holdout) — enforced by TUNE_YEARS and an assertion. Tune-fold metrics are
selection machinery, not quotable numbers.

Usage:
    python src/blend_weight_search.py
"""

import hashlib
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features, load_canonical_results  # noqa: E402
from train import (ELO_DRAW_RATE, HFA_ELO, MatchContext, elo_prior_proba,  # noqa: E402
                   make_X, predict_match_proba, train_model)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

TUNE_YEARS = [2006, 2010]          # NEVER the 2014/2018/2022 benchmark
WEIGHTS = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)
BOOT_N, BOOT_SEED = 10_000, 12345  # matches paired_tests.py conventions


def frozen_snapshot(canonical: pd.DataFrame, cutoff: pd.Timestamp):
    """ONE pre-tournament snapshot: per-team post-last-match state + H2H,
    exactly the serving semantics (simulate.Predictor's team_state.csv)."""
    df_cut = canonical[canonical["date"] < cutoff].reset_index(drop=True)
    feats, _elo, team_state_df = build_features(df_cut)

    state = {
        r["team"]: {"elo": float(r["elo"]),
                    "win_rate_5": float(r["win_rate_5"]),
                    "gd_5": float(r["gd_5"]),
                    "win_rate_10": float(r["win_rate_10"]),
                    "gd_10": float(r["gd_10"]),
                    "confederation": str(r["confederation"]),
                    "conf_elo": float(r["conf_elo"])}
        for _, r in team_state_df.iterrows()
    }
    h2h: dict[frozenset, list] = {}
    for _, row in feats.iterrows():
        key = frozenset({row["home_team"], row["away_team"]})
        h2h.setdefault(key, []).append((row["home_team"], int(row["outcome"])))
    h2h = {k: v[-10:] for k, v in h2h.items()}
    return state, h2h


def _h2h_stats(h2h: dict, home: str, away: str) -> tuple[int, float]:
    hist = h2h.get(frozenset({home, away}), [])
    n = len(hist)
    if n == 0:
        return 0, 0.5
    hw = sum(1 for h, o in hist if (h == home and o == 0) or (h == away and o == 2))
    return n, hw / n


def fold_frozen_predictions(canonical: pd.DataFrame, features_all: pd.DataFrame,
                            year: int):
    """
    Frozen-state fold evaluation (THE canonical fold protocol — the Phase 4
    benchmark imports and runs this very function).

    Returns (fold_df, y, p_model, prior): fold_df is the fold's canonical
    match rows plus `frozen_home_elo`/`frozen_away_elo` (the pre-tournament
    snapshot ratings every prediction uses — needed by the Elo baseline so it
    runs on the SAME frozen state), y the outcomes, p_model the venue-aware
    model output before the blend, prior the production Elo-logistic prior.
    """
    fold = canonical[(canonical["tournament"] == "FIFA World Cup")
                     & (canonical["date"].dt.year == year)].reset_index(drop=True)
    if fold.empty:
        raise RuntimeError(f"no WC {year} matches in canonical data")
    cutoff = fold["date"].min()

    train_df = features_all[features_all["date"] < cutoff].reset_index(drop=True)
    print(f"  WC {year}: {len(fold)} matches | training on {len(train_df):,} "
          f"matches before {cutoff.date()} ...")
    model, feature_cols = train_model(train_df)

    state, h2h = frozen_snapshot(canonical, cutoff)

    def build_x(home, away, neutral):
        hs, as_ = state[home], state[away]
        n_h2h, hwr = _h2h_stats(h2h, home, away)
        row = {
            "home_elo": hs["elo"], "away_elo": as_["elo"],
            "elo_diff": hs["elo"] - as_["elo"],
            "home_win_rate_5": hs["win_rate_5"], "away_win_rate_5": as_["win_rate_5"],
            "home_gd_5": hs["gd_5"], "away_gd_5": as_["gd_5"],
            "home_win_rate_10": hs["win_rate_10"], "away_win_rate_10": as_["win_rate_10"],
            "home_gd_10": hs["gd_10"], "away_gd_10": as_["gd_10"],
            "h2h_n": n_h2h, "h2h_home_wr": hwr,
            "home_conf_elo": hs["conf_elo"], "away_conf_elo": as_["conf_elo"],
            "neutral": int(neutral), "is_world_cup": 1,
            "home_confederation": hs["confederation"],
            "away_confederation": as_["confederation"],
        }
        X, _ = make_X(pd.DataFrame([row]), feature_cols)
        return X

    y, p_model, prior = [], [], []
    frozen_h_elo, frozen_a_elo = [], []
    for _, m in fold.iterrows():
        home, away = m["home_team"], m["away_team"]
        neutral = bool(m["neutral"])
        ctx = MatchContext(home, away, neutral=neutral)
        elo_h, elo_a = state[home]["elo"], state[away]["elo"]
        frozen_h_elo.append(elo_h)
        frozen_a_elo.append(elo_a)

        # Model output before the blend (venue contract applied inside).
        p_model.append(predict_match_proba(model, build_x, ctx, elo_h, elo_a,
                                           blend_weight=1.0))
        # Production prior, exactly as predict_match_proba's blend sees it.
        h_off = 0.0 if neutral else HFA_ELO
        prior.append(elo_prior_proba(elo_h + h_off, elo_a, ELO_DRAW_RATE))

        hs_, as_ = float(m["home_score"]), float(m["away_score"])
        y.append(0 if hs_ > as_ else (1 if hs_ == as_ else 2))

    fold = fold.assign(frozen_home_elo=frozen_h_elo,
                       frozen_away_elo=frozen_a_elo)
    return fold, np.array(y), np.vstack(p_model), np.vstack(prior)


def brier(y: np.ndarray, proba: np.ndarray) -> float:
    yb = np.zeros_like(proba)
    yb[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((proba - yb) ** 2, axis=1)))


def _blend(w: float, p_model: np.ndarray, prior: np.ndarray) -> np.ndarray:
    b = w * p_model + (1.0 - w) * prior
    return b / b.sum(axis=1, keepdims=True)


def main() -> None:
    canonical = load_canonical_results()
    features_all = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])

    folds = {}
    for year in TUNE_YEARS:
        assert year < 2014, "benchmark folds (2014/2018/2022) are Phase 4 holdout"
        fold_df, yy, pm, pr = fold_frozen_predictions(canonical, features_all, year)
        folds[year] = (yy, pm, pr)
        n_host = int((fold_df["neutral"] == 0).sum())
        print(f"    host (non-neutral) rows: {n_host} of {len(fold_df)}")

    y_all = np.concatenate([folds[y][0] for y in TUNE_YEARS])
    pm_all = np.vstack([folds[y][1] for y in TUNE_YEARS])
    pr_all = np.vstack([folds[y][2] for y in TUNE_YEARS])

    rows = []
    for w in WEIGHTS:
        rec = {"w": w}
        for year in TUNE_YEARS:
            yy, pm, pr = folds[year]
            b = _blend(w, pm, pr)
            rec[f"ll_{year}"] = log_loss(yy, b, labels=[0, 1, 2])
            rec[f"acc_{year}"] = accuracy_score(yy, b.argmax(axis=1))
            rec[f"brier_{year}"] = brier(yy, b)
        b = _blend(w, pm_all, pr_all)
        rec["ll_pooled"] = log_loss(y_all, b, labels=[0, 1, 2])
        rec["acc_pooled"] = accuracy_score(y_all, b.argmax(axis=1))
        rec["brier_pooled"] = brier(y_all, b)
        rows.append(rec)
    table = pd.DataFrame(rows)

    argmin_w = float(table.loc[table["ll_pooled"].idxmin(), "w"])

    # Bootstrap (paired, pooled matches): SE of LL at the argmin, and SE of the
    # LL difference between w=0.75 and the argmin — the pre-registered band.
    rng = np.random.default_rng(BOOT_SEED)
    b_min = _blend(argmin_w, pm_all, pr_all)
    b_075 = _blend(0.75, pm_all, pr_all)
    n = len(y_all)
    ll_min_bs, ll_diff_bs = [], []
    for _ in range(BOOT_N):
        idx = rng.integers(0, n, n)
        lm = log_loss(y_all[idx], b_min[idx], labels=[0, 1, 2])
        l7 = log_loss(y_all[idx], b_075[idx], labels=[0, 1, 2])
        ll_min_bs.append(lm)
        ll_diff_bs.append(l7 - lm)
    se_min = float(np.std(ll_min_bs, ddof=1))
    se_diff = float(np.std(ll_diff_bs, ddof=1))

    # Header metadata + artifact.
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = "unknown"
    feat_hash = hashlib.sha256(
        (PROCESSED_DIR / "features.csv").read_bytes()).hexdigest()

    out = REPORTS_DIR / "blend_weight_search.csv"
    header = (
        f"# Blend-weight search on TUNE folds {TUNE_YEARS} under the corrected\n"
        f"# frozen-state protocol (remediation Phase 3). Selection metric:\n"
        f"# pooled log-loss. Tune-fold metrics are selection machinery, NOT\n"
        f"# quotable performance numbers (the 2014/2018/2022 benchmark is the\n"
        f"# untouched Phase 4 holdout).\n"
        f"# date={date.today().isoformat()}  code_sha={sha}\n"
        f"# canonical_features_sha256={feat_hash}\n"
        f"# HFA_ELO={HFA_ELO}  ELO_DRAW_RATE={ELO_DRAW_RATE}\n"
        f"# argmin_w={argmin_w}  ll_argmin={table['ll_pooled'].min():.6f}\n"
        f"# ll_at_0.75={float(table.loc[table['w'] == 0.75, 'll_pooled'].iloc[0]):.6f}\n"
        f"# bootstrap({BOOT_N}, seed {BOOT_SEED}): se_ll_argmin={se_min:.6f}  "
        f"se_ll_diff_075_vs_argmin={se_diff:.6f}\n"
    )
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write(header)
        table.to_csv(fh, index=False, float_format="%.6f")
    print(f"\nwrote {out}")

    ll75 = float(table.loc[table["w"] == 0.75, "ll_pooled"].iloc[0])
    print(f"argmin w = {argmin_w}  (pooled LL {table['ll_pooled'].min():.6f})")
    print(f"w=0.75 pooled LL {ll75:.6f}  delta {ll75 - table['ll_pooled'].min():.6f}")
    print(f"bootstrap SE at argmin {se_min:.6f}; SE of paired diff {se_diff:.6f}")


if __name__ == "__main__":
    main()
