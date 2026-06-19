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
# Honest finding from the clean walk-forward rerun (see
# notebooks/04_model_improvement.md): the blend IMPROVES on raw XGB
# (log-loss/Brier) but does NOT beat the naive Elo baseline — Elo is in fact
# better on all three metrics, and the blend-vs-Elo gap is within noise
# (McNemar p≈0.30, bootstrap log-loss CI straddles 0). The blend ships for its
# better calibration over raw XGB, not as a claimed win over Elo. Weight tuned
# on WC 2006/2010 only.
ELO_BLEND_W   = 0.75   # XGB share of the blend
ELO_DRAW_RATE = 0.227  # historical draw share used by the prior


def elo_prior_proba(elo_home: float, elo_away: float,
                    draw_rate: float = ELO_DRAW_RATE) -> np.ndarray:
    """Elo-logistic (win, draw, loss) prior with a fixed draw share."""
    e = 1.0 / (1.0 + 10.0 ** ((elo_away - elo_home) / 400.0))
    p = np.array([(1 - draw_rate) * e, draw_rate, (1 - draw_rate) * (1 - e)])
    return p / p.sum()


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
