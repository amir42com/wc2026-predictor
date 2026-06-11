"""
Train an XGBoost classifier to predict match outcome (home win / draw / away win).

Reads:  data/processed/features.csv
Writes: models/xgb_wc2026.joblib  — bundle: {model, feature_cols, label_map}

Temporal split: all matches before TEST_YEAR are training data; the rest are
held out for evaluation.  No shuffling — preserves chronological ordering to
prevent leakage.

Usage:
    python src/train.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

TEST_YEAR = 2018

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


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features.csv ...")
    df = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])
    print(f"  {len(df):,} rows total")

    mask = df["date"].dt.year < TEST_YEAR
    df_train, df_test = df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)
    print(f"  Train: {len(df_train):,} matches (before {TEST_YEAR})")
    print(f"  Test:  {len(df_test):,} matches  ({TEST_YEAR}+)\n")

    print("Training ...")
    model, feature_cols = train_model(df_train)
    print(f"  {len(feature_cols)} input features\n")

    print(f"Evaluation on held-out test set ({TEST_YEAR}+):")
    evaluate(model, df_test, feature_cols)

    bundle = {"model": model, "feature_cols": feature_cols, "label_map": LABEL_MAP}
    out = MODELS_DIR / "xgb_wc2026.joblib"
    joblib.dump(bundle, out)

    size_mb = out.stat().st_size / 1e6
    print(f"\nModel saved -> {out}  ({size_mb:.1f} MB)")

    print("\nTop 15 features by importance:")
    imp = pd.Series(model.feature_importances_, index=feature_cols).nlargest(15)
    for feat, score in imp.items():
        bar = "#" * int(score * 400)
        print(f"  {feat:<30s} {score:.4f}  {bar}")


if __name__ == "__main__":
    main()
