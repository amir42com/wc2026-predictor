"""
Train an XGBoost classifier to predict match outcome (home win / draw / away win).

Reads:  data/processed/features.csv
Writes: models/xgb_wc2026.joblib  — bundle: {model, feature_cols, label_map}

Two distinct models are built here, and they must not be confused:
  * An EVALUATION model trained on pre-TEST_YEAR data and scored on TEST_YEAR+,
    purely to print a held-out accuracy/log-loss read-out.  No shuffling —
    chronological order is preserved to prevent leakage.
  * The SHIPPED model (the saved bundle), retrained on ALL available
    pre-tournament data via production_mask() so the deployed predictor has
    seen every match up to the eve of WC 2026 (1872 → June 2026), never just
    the pre-2018 slice.  It is held out only from the WC 2026 fixtures it
    predicts.

(The leakage-free per-fold article backtest lives in src/backtest.py and is
entirely separate from both of the above.)

Usage:
    python src/train.py              # train + evaluate
    python src/train.py --backtest   # also run WC backtesting (trains 4 extra models)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

TEST_YEAR = 2018   # eval-only split boundary (held-out read-out, NOT the shipped model)


def production_mask(df: pd.DataFrame) -> pd.Series:
    """
    Rows the SHIPPED model trains on: all pre-tournament data.

    Includes every match up to the eve of WC 2026 (friendlies, qualifiers,
    continental cups — all legitimate pre-tournament form) and excludes ONLY
    the WC 2026 tournament fixtures we are predicting.  is_world_cup flags
    World Cup matches and 2026 is the only future World Cup, so dropping WC
    matches dated 2026+ holds out exactly that tournament while keeping the
    full build-up.
    """
    return ~((df["is_world_cup"] == 1) & (df["date"].dt.year >= 2026))

# Columns fed to the model — order is fixed for inference alignment
NUMERIC_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_win_rate_5",  "away_win_rate_5",  "home_gd_5",  "away_gd_5",
    "home_win_rate_10", "away_win_rate_10", "home_gd_10", "away_gd_10",
    "h2h_n", "h2h_home_wr",
    "home_conf_elo", "away_conf_elo",
    "neutral", "is_world_cup",
]

LABEL_MAP = {0: "Home win", 1: "Draw", 2: "Away win"}

# Matches the canonical article/eval window (src/backtest.py, experiments.py).
# This is the quick `--backtest` sanity table only; the 2010 fold was dropped
# so it no longer disagrees with the 192-match (2014/2018/2022) backtest.
WC_BACKTEST_YEARS = [2014, 2018, 2022]

# Production ensemble: blend XGB probabilities with an Elo-logistic prior.
# Honest finding from the canonical 192-match backtest (see
# reports/backtest_metrics_summary.csv): the blend IMPROVES on raw XGB
# (log-loss/Brier) and posts numerically lower pooled log-loss (0.9720 vs
# 0.9769) and Brier (0.5735 vs 0.5770) than the naive Elo baseline, but Elo
# is more accurate (57.8% vs 56.25%) and the gap is within noise (McNemar
# p≈0.58, bootstrap log-loss CI straddles 0). The blend ships for its better
# probabilistic scores over raw XGB, not as a claimed win over Elo. Weight
# tuned on WC 2006/2010 only.
ELO_BLEND_W   = 0.75   # XGB share of the blend
ELO_DRAW_RATE = 0.227  # historical draw share used by the prior

# Venue contract (remediation Phase 2): the pipeline's Elo is venue-blind, so
# when a match is NOT neutral the Elo-logistic prior receives a fixed
# home-advantage offset. Fitted ONCE (2026-07-13) by maximum likelihood under
# the production link on the FULL non-neutral canonical pre-cutoff history
# (fit_hfa_elo below: 36,346 rows; the prior's fixed draw share makes draws
# uninformative for the offset, so 28,038 decided matches enter). A sensitivity
# study (table in reports/remediation_phase2.md) put every era window
# (full/>=1990/>=2000/>=2010), match type (all/competitive/friendly) and a
# draw-inclusive Davidson link in the 107-138 range; the decision-rule window
# (post-1990 competitive, 128.32) corroborates the full-history fit, and the
# rejected deployment-domain alternative — post-2010 competitive, 113.33 —
# would move blended probabilities by only ~0.35pp. HFA declines monotonically
# across eras (competitive: 138 -> 113), so read Phase 4's mandatory per-fold
# refit sensitivity line in that context. Like ELO_DRAW_RATE this is a frozen
# production constant with recorded provenance, NOT tuned against any
# published metric; tests/test_match_context.py re-runs the fit recipe.
HFA_ELO = 125.58  # full-history MLE, production link (canonical features.csv)


@dataclass(frozen=True)
class MatchContext:
    """
    Explicit venue/inference context for one fixture (remediation Phase 2).

    Invariant (upstream martj42 semantics, verified on all non-neutral WC
    rows): when `neutral` is False, `home_team` IS the side receiving home
    advantage. `venue_country` is evidence for deriving `neutral` (and for
    audit); it is never fed to the model.
    """
    home_team: str
    away_team: str
    neutral: bool
    venue_country: "str | None" = None
    is_world_cup: bool = True


def elo_prior_proba(elo_home: float, elo_away: float,
                    draw_rate: float = ELO_DRAW_RATE) -> np.ndarray:
    """Elo-logistic (win, draw, loss) prior with a fixed draw share."""
    e = 1.0 / (1.0 + 10.0 ** ((elo_away - elo_home) / 400.0))
    p = np.array([(1 - draw_rate) * e, draw_rate, (1 - draw_rate) * (1 - e)])
    return p / p.sum()


def fit_hfa_elo(features_df: pd.DataFrame) -> float:
    """
    Maximum-likelihood home-advantage offset (in Elo points) for the prior.

    Recipe (deterministic, canonical data only): take every non-neutral row of
    the canonical features.csv (pre-cutoff by construction), model the home
    side's win probability among decided matches as the Elo expectation with
    the home rating shifted by `h`, and maximize the log-likelihood over h in
    [0, 400]. Under the elo_prior_proba form the draw share is a constant
    factor independent of h, so drawn matches contribute nothing to the
    gradient and only home wins / away wins enter.
    """
    from scipy.optimize import minimize_scalar

    nn = features_df[features_df["neutral"] == 0]
    diff = nn["elo_diff"].to_numpy(dtype=float)
    home_win = (nn["outcome"] == 0).to_numpy()
    away_win = (nn["outcome"] == 2).to_numpy()

    def neg_ll(h: float) -> float:
        e = 1.0 / (1.0 + 10.0 ** (-(diff + h) / 400.0))
        return -(np.log(e[home_win]).sum() + np.log(1.0 - e[away_win]).sum())

    res = minimize_scalar(neg_ll, bounds=(0.0, 400.0), method="bounded",
                          options={"xatol": 1e-6})
    return float(res.x)


def predict_match_proba(model, build_x, ctx: MatchContext,
                        elo_home: float, elo_away: float,
                        blend_weight: float = ELO_BLEND_W,
                        draw_rate: float = ELO_DRAW_RATE,
                        hfa_elo: "float | None" = None) -> np.ndarray:
    """
    THE single deployed inference, shared by the Streamlit predictor, the
    tracker, the simulator and the backtest so all evaluate the exact same
    procedure. The venue treatment is governed entirely by `ctx` (MatchContext).

    `build_x(home, away, neutral)` is supplied by the caller and returns a
    single-row feature matrix aligned to the model's feature columns (the
    caller owns X construction — team-state reconstruction for serving,
    recorded-row re-orientation for the backtest).

    Neutral fixture (ctx.neutral=True — orientation is bookkeeping):
      1. predict_proba for home-vs-away and away-vs-home, both with neutral=1,
      2. reverse the second vector ([2,1,0]) back to home's perspective,
      3. average the two (cancels the model's residual home/away asymmetry),
      4. shrink toward the symmetric Elo-logistic prior at `blend_weight`.

    Non-neutral fixture (ctx.home_team has real home advantage):
      1. a SINGLE orientation — ctx.home_team in the home slot, neutral=0
         (symmetry-averaging would erase exactly the advantage being modelled),
      2. shrink toward the Elo-logistic prior with the home rating shifted by
         `hfa_elo` (default: the frozen production constant HFA_ELO).

    `elo_home`/`elo_away` follow ctx.home_team/ctx.away_team. Returns
    P(home win, draw, away win), summing to 1.
    """
    if ctx.neutral:
        p_ab = model.predict_proba(build_x(ctx.home_team, ctx.away_team, 1))[0]
        p_ba = model.predict_proba(build_x(ctx.away_team, ctx.home_team, 1))[0]
        p_model = (p_ab + p_ba[[2, 1, 0]]) / 2.0
        prior = elo_prior_proba(elo_home, elo_away, draw_rate)
    else:
        h = HFA_ELO if hfa_elo is None else hfa_elo
        p_model = model.predict_proba(build_x(ctx.home_team, ctx.away_team, 0))[0]
        prior = elo_prior_proba(elo_home + h, elo_away, draw_rate)
    blended = blend_weight * p_model + (1.0 - blend_weight) * prior
    return blended / blended.sum()


def predict_neutral_proba(model, build_x, team_a: str, team_b: str,
                          elo_a: float, elo_b: float,
                          blend_weight: float = ELO_BLEND_W,
                          draw_rate: float = ELO_DRAW_RATE) -> np.ndarray:
    """
    DEPRECATED shim — the pre-Phase-2 neutral-only inference, kept so the
    frozen serving path (simulate.Predictor, tracker, app) is byte-identical
    until the serving rebuild phase. `build_x(home, away)` is the legacy
    two-argument contract; the neutral flag it hardcodes internally is what
    the neutral path fed the model before the contract existed. New code must
    construct a MatchContext and call predict_match_proba.
    """
    ctx = MatchContext(team_a, team_b, neutral=True)
    return predict_match_proba(model, lambda h, a, _n: build_x(h, a), ctx,
                               elo_a, elo_b, blend_weight, draw_rate)


def make_X(df: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the model-ready feature matrix.
    One-hot encode confederation columns; reindex to feature_cols when provided
    (ensures column alignment between train-time and inference-time).
    """
    conf_dummies = pd.get_dummies(
        df[["home_confederation", "away_confederation"]],
        prefix=["h_conf", "a_conf"],
    )
    X = pd.concat(
        [df[NUMERIC_COLS].reset_index(drop=True), conf_dummies.reset_index(drop=True)],
        axis=1,
    )
    if feature_cols is not None:
        X = X.reindex(columns=feature_cols, fill_value=0)
    else:
        feature_cols = list(X.columns)
    return X.astype(float), feature_cols


def train_model(df_train: pd.DataFrame) -> tuple[XGBClassifier, list[str]]:
    X_train, feature_cols = make_X(df_train)
    y_train = df_train["outcome"].values

    # Hold out the last 10 % of training data (chronologically) for early stopping
    split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best iteration: {model.best_iteration}  (of 1000 max)")
    return model, feature_cols


def evaluate(model: XGBClassifier, df_test: pd.DataFrame, feature_cols: list[str]) -> None:
    X_test, _ = make_X(df_test, feature_cols)
    y_test = df_test["outcome"].values

    proba = model.predict_proba(X_test)
    y_pred = proba.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, proba)
    names = [LABEL_MAP[i] for i in range(3)]

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Log-loss : {ll:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=names, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {n}" for n in names],
        columns=[f"Pred: {n}" for n in names],
    )
    print(cm_df.to_string())

    # WC-only breakdown
    wc_mask = df_test["is_world_cup"] == 1
    if wc_mask.sum():
        wc_proba = model.predict_proba(X_test[wc_mask.values])
        wc_pred = wc_proba.argmax(axis=1)
        wc_acc = accuracy_score(y_test[wc_mask.values], wc_pred)
        print(f"\n  World Cup matches only ({wc_mask.sum()}):  accuracy = {wc_acc:.4f}")


def _brier_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean multi-class Brier score: mean over matches of sum_c (p_c - y_c)^2."""
    y_bin = np.zeros_like(proba)
    y_bin[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - y_bin) ** 2, axis=1)))


def backtest_wc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward backtest over WC_BACKTEST_YEARS.

    For each tournament year:
      - Train a fresh model on all matches BEFORE that year (no leakage).
      - Evaluate on that year's World Cup matches only.

    Returns a DataFrame with one row per tournament plus a Combined row.
    """
    rows: list[dict] = []

    for year in WC_BACKTEST_YEARS:
        wc_mask = (df["is_world_cup"] == 1) & (df["date"].dt.year == year)
        df_wc = df[wc_mask].reset_index(drop=True)

        if len(df_wc) == 0:
            print(f"  WC {year}: no matches found in features.csv — skipping.")
            continue

        df_pre = df[df["date"].dt.year < year].reset_index(drop=True)
        print(f"  WC {year}: {len(df_wc):>2} matches | "
              f"training on {len(df_pre):,} pre-{year} matches ...")

        model_y, fc = train_model(df_pre)

        X_wc, _ = make_X(df_wc, fc)
        y_wc     = df_wc["outcome"].values
        proba    = model_y.predict_proba(X_wc)
        y_pred   = proba.argmax(axis=1)

        rows.append({
            "Tournament": f"WC {year}",
            "Matches":    len(df_wc),
            "Accuracy":   accuracy_score(y_wc, y_pred),
            "Log-loss":   log_loss(y_wc, proba),
            "Brier":      _brier_multiclass(y_wc, proba),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Combined row — weighted by match count so per-sample metrics stay correct
    n = result["Matches"].sum()
    combined = {
        "Tournament": "Combined",
        "Matches":    n,
        "Accuracy":   (result["Accuracy"] * result["Matches"]).sum() / n,
        "Log-loss":   (result["Log-loss"] * result["Matches"]).sum() / n,
        "Brier":      (result["Brier"]    * result["Matches"]).sum() / n,
    }
    return pd.concat([result, pd.DataFrame([combined])], ignore_index=True)


def _print_backtest_table(df: pd.DataFrame) -> None:
    sep   = "=" * 62
    inner = "-" * 62
    header = f"{'Tournament':<14} {'Matches':>7}  {'Accuracy':>9}  {'Log-loss':>9}  {'Brier':>9}"

    print(f"\n{'WC Backtesting Results':^62}")
    print(f"{'(each model trained on pre-tournament data only)':^62}")
    print(sep)
    print(header)
    print(inner)

    for _, row in df.iterrows():
        is_combined = row["Tournament"] == "Combined"
        if is_combined:
            print(inner)
        print(
            f"{row['Tournament']:<14} {int(row['Matches']):>7}  "
            f"{row['Accuracy']*100:>8.1f}%  "
            f"{row['Log-loss']:>9.4f}  "
            f"{row['Brier']:>9.4f}"
        )

    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true",
                        help="Run WC walk-forward backtest (trains 4 extra models, ~2 min)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features.csv ...")
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    print(f"  {len(df):,} rows total")

    mask = df["date"].dt.year < TEST_YEAR
    df_train, df_test = df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)
    print(f"  Train: {len(df_train):,} matches (before {TEST_YEAR})")
    print(f"  Test:  {len(df_test):,} matches  ({TEST_YEAR}+)\n")

    print("Training evaluation model (pre-2018 only) ...")
    eval_model, feature_cols = train_model(df_train)
    print(f"  {len(feature_cols)} input features\n")

    print(f"Evaluation on held-out test set ({TEST_YEAR}+):")
    evaluate(eval_model, df_test, feature_cols)

    # Shipped model: retrain on ALL pre-tournament data so the deployed
    # predictor has seen 2018–2026, not just the pre-2018 eval slice.
    df_prod = df[production_mask(df)].reset_index(drop=True)
    n_excluded = len(df) - len(df_prod)
    print(f"\nTraining FINAL production model on all pre-tournament data ...")
    print(f"  {len(df_prod):,} matches (excludes {n_excluded} WC 2026 fixture(s))")
    model, feature_cols = train_model(df_prod)

    bundle = {"model": model, "feature_cols": feature_cols, "label_map": LABEL_MAP}
    out = MODELS_DIR / "xgb_wc2026.joblib"
    joblib.dump(bundle, out)

    size_mb = out.stat().st_size / 1e6
    print(f"\nModel saved -> {out}  ({size_mb:.1f} MB)")

    print("\nTop 15 features by importance (shipped model):")
    imp = pd.Series(model.feature_importances_, index=feature_cols).nlargest(15)
    for feat, score in imp.items():
        bar = "#" * int(score * 400)
        print(f"  {feat:<30s} {score:.4f}  {bar}")

    if args.backtest:
        print(f"\n\nRunning WC backtest for {WC_BACKTEST_YEARS} ...")
        print("(trains one model per tournament on pre-tournament data)\n")
        bt = backtest_wc(df)
        if not bt.empty:
            _print_backtest_table(bt)


if __name__ == "__main__":
    main()
