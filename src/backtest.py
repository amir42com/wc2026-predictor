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

from train import (ELO_BLEND_W, ELO_DRAW_RATE, HFA_ELO, MatchContext,
                   elo_prior_proba, make_X, predict_match_proba, train_model)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

BACKTEST_YEARS = [2014, 2018, 2022]

# ── Protocol gate (remediation) ─────────────────────────────────────────────
# Phase 2 landed the venue contract in this module (host rows get real home
# advantage; the Elo baseline gets the same HFA information). The frozen-state
# backtest protocol does not land until Phase 4, so ANY number produced by the
# fold machinery below would mix protocols and must not look quotable — not in
# a terminal, not in a CSV. Phase 4 flips this flag when the full protocol is
# in place; until then every inference entry point refuses to run.
PROTOCOL_COMPLETE = False


def _require_complete_protocol() -> None:
    if not PROTOCOL_COMPLETE:
        raise RuntimeError(
            "PROTOCOL-INCOMPLETE: the venue contract (Phase 2) is applied but "
            "the frozen-state backtest protocol (Phase 4) has not landed. "
            "Refusing to produce backtest numbers — any output now would mix "
            "protocols. This gate is lifted by backtest.PROTOCOL_COMPLETE in "
            "Phase 4.")


def _swap_orientation(d: dict) -> dict:
    """
    A recorded feature row with home/away roles swapped, for symmetry averaging.
    Mirrors what deployment's feature_X produces for the reversed ordering: swap
    the paired home_/away_ features, negate elo_diff, and set h2h_home_wr to the
    new home's (old away's) head-to-head win share.
    """
    s = dict(d)
    for x, y in [("home_elo", "away_elo"),
                 ("home_win_rate_5", "away_win_rate_5"),
                 ("home_gd_5", "away_gd_5"),
                 ("home_win_rate_10", "away_win_rate_10"),
                 ("home_gd_10", "away_gd_10"),
                 ("home_conf_elo", "away_conf_elo"),
                 ("home_confederation", "away_confederation")]:
        s[x], s[y] = d[y], d[x]
    s["elo_diff"] = -d["elo_diff"]
    n = d.get("h2h_n", 0) or 0
    if n:
        s["h2h_home_wr"] = (d.get("h2h_away_wins", 0) or 0) / n
    return s


def deployed_model_and_blend(model, df_wc: pd.DataFrame,
                             feature_cols: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the EXACT deployed inference (train.predict_match_proba) on each
    recorded WC match, under the venue contract: each row's MatchContext comes
    from its recorded neutral flag, so host matches (15 of the 192: Brazil
    2014, Russia 2018, Qatar 2022) run single-orientation with the HFA-shifted
    prior, and neutral matches symmetry-average exactly as deployed. Returns
    (model_avg, blend) — model_avg is the model output BEFORE the blend
    (blend_weight=1.0).
    """
    _require_complete_protocol()
    model_avg, blend = [], []
    for i in range(len(df_wc)):
        row = df_wc.iloc[i].to_dict()
        rev = _swap_orientation(row)
        home0 = row["home_team"]
        ctx = MatchContext(row["home_team"], row["away_team"],
                           neutral=bool(row["neutral"]))

        def build_x(home, away, neutral, _row=row, _rev=rev, _home0=home0):
            d = dict(_row if home == _home0 else _rev)
            d["neutral"] = int(neutral)   # context governs the flag the model sees
            X, _ = make_X(pd.DataFrame([d]), feature_cols)
            return X

        ea, eb = row["home_elo"], row["away_elo"]
        model_avg.append(predict_match_proba(
            model, build_x, ctx, ea, eb, blend_weight=1.0))
        blend.append(predict_match_proba(model, build_x, ctx, ea, eb))
    return np.vstack(model_avg), np.vstack(blend)


def production_blend_prior(df_wc: pd.DataFrame) -> np.ndarray:
    """
    The Elo-logistic prior the DEPLOYED app blends XGB with.

    Identical formula to elo_baseline_proba, but with the FIXED production
    draw share (train.ELO_DRAW_RATE = 0.227) rather than a fold-specific draw
    rate.  This is exactly train.elo_prior_proba / simulate.Predictor's prior,
    so blending against it makes the backtested blend the *real* production
    system.  Venue contract: host rows shift the home rating by the fixed
    production HFA_ELO, exactly as predict_match_proba's non-neutral path does.
    """
    _require_complete_protocol()
    return np.vstack([elo_prior_proba(h + HFA_ELO * (1 - n), a, ELO_DRAW_RATE)
                      for h, a, n in zip(df_wc["home_elo"].values,
                                         df_wc["away_elo"].values,
                                         df_wc["neutral"].values)])


def brier_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean multi-class Brier score: mean over matches of sum_c (p_c - y_c)^2."""
    y_bin = np.zeros_like(proba)
    y_bin[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - y_bin) ** 2, axis=1)))


def elo_baseline_proba(df_wc: pd.DataFrame, draw_rate: float,
                       hfa_elo: float = HFA_ELO) -> np.ndarray:
    """
    Naive Elo baseline probabilities.

    Win probability from the standard Elo expectation
        E_home = 1 / (1 + 10^((away_elo - home_elo) / 400))
    then reserve `draw_rate` (estimated from pre-tournament data) for the
    draw class and split the rest proportionally.

    Venue contract (Checkpoint 2 decision): the baseline receives the SAME
    venue information as the blend — host rows (neutral == 0) shift the home
    rating by the same fixed HFA offset — so the article's headline comparison
    never rests on an information asymmetry between the systems.
    """
    _require_complete_protocol()
    eff_home = df_wc["home_elo"] + hfa_elo * (1 - df_wc["neutral"])
    e_home = 1.0 / (1.0 + 10.0 ** ((df_wc["away_elo"] - eff_home) / 400.0))
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
    _require_complete_protocol()
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

        y_wc = df_wc["outcome"].values

        # Naive Elo baseline — draw rate from the same pre-tournament window
        # (no leakage). This stays fold-specific.
        draw_rate = float((df_pre["outcome"] == 1).mean())
        base_proba = elo_baseline_proba(df_wc, draw_rate)

        # Production model = the EXACT deployed inference under the venue
        # contract (train.predict_match_proba): neutral rows symmetry-average
        # both orderings, host rows run single-orientation with the HFA prior,
        # then blend with the fixed-0.227 Elo prior.
        _, model_proba = deployed_model_and_blend(model, df_wc, feature_cols)

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
