"""Fair offline comparison of quantile boosting models on the Part 2 split."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))
from feature_contract import ALLOWED_FEATURES, assert_no_leakage


DROP = {
    "market_median_price_w", "market_q1_price", "market_q3_price",
    "has_price_quantiles", "transaction_subgroup",
}
TARGET_ENCODED = ("model", "generation")
QUANTILES = (0.25, 0.75)


df = pd.read_csv(ROOT / "dataset_prepared.csv", parse_dates=["evaluation_date"])
cutoff = df["evaluation_date"].quantile(0.8)
train = df[df["evaluation_date"] < cutoff].copy()
test = df[df["evaluation_date"] >= cutoff].copy()
feature_cols = [c for c in ALLOWED_FEATURES if c in df.columns and c not in DROP]
assert_no_leakage(feature_cols)
for frame in (train, test):
    frame["month_idx"] = (
        (frame["evaluation_date"].dt.year - 2024) * 12
        + frame["evaluation_date"].dt.month
    )

y_train = np.log(train["buy_price"])
X_train_raw = train[feature_cols + ["month_idx"]].copy()
X_test_raw = test[feature_cols + ["month_idx"]].copy()
cat_cols = X_train_raw.select_dtypes(include=["object", "bool"]).columns.tolist()


def encoded_frames():
    """Current sklearn preprocessing, reused for XGBoost for comparability."""
    x_train, x_test = X_train_raw.copy(), X_test_raw.copy()
    global_mean = y_train.mean()
    for column in TARGET_ENCODED:
        stats = y_train.groupby(x_train[column]).agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * 20) / (stats["count"] + 20)
        x_train[f"{column}_te"] = x_train[column].map(smooth).fillna(global_mean)
        x_test[f"{column}_te"] = x_test[column].map(smooth).fillna(global_mean)
        x_train.drop(columns=column, inplace=True)
        x_test.drop(columns=column, inplace=True)
    bool_columns = x_train.select_dtypes(include=["bool"]).columns
    for column in bool_columns:
        x_train[column] = x_train[column].astype("int8")
        x_test[column] = x_test[column].astype("int8")
    remaining_cats = x_train.select_dtypes(include=["object"]).columns
    for column in remaining_cats:
        train_values = x_train[column].fillna("__MISSING__").astype(str)
        test_values = x_test[column].fillna("__MISSING__").astype(str)
        levels = pd.Index(train_values.unique())
        x_train[column] = pd.Categorical(train_values, categories=levels)
        x_test[column] = pd.Categorical(test_values, categories=levels)
    return x_train, x_test


def catboost_frames():
    x_train, x_test = X_train_raw.copy(), X_test_raw.copy()
    for column in cat_cols:
        x_train[column] = x_train[column].fillna("__MISSING__").astype(str)
        x_test[column] = x_test[column].fillna("__MISSING__").astype(str)
    return x_train, x_test


def lgbm_frames():
    x_train, x_test = X_train_raw.copy(), X_test_raw.copy()
    for column in cat_cols:
        levels = pd.Index(x_train[column].dropna().unique())
        x_train[column] = pd.Categorical(x_train[column], categories=levels)
        x_test[column] = pd.Categorical(x_test[column], categories=levels)
    return x_train, x_test


def fit_pair(name):
    predictions = []
    if name == "HistGradientBoosting":
        x_train, x_test = encoded_frames()
        factories = [lambda q=q: HistGradientBoostingRegressor(
            loss="quantile", quantile=q, categorical_features="from_dtype",
            max_iter=300, learning_rate=0.05, random_state=42,
        ) for q in QUANTILES]
        fit_kwargs = {}
    elif name == "CatBoost":
        x_train, x_test = catboost_frames()
        factories = [lambda q=q: CatBoostRegressor(
            loss_function=f"Quantile:alpha={q}", iterations=300, depth=8,
            learning_rate=0.05, random_seed=42, verbose=False,
        ) for q in QUANTILES]
        fit_kwargs = {"cat_features": cat_cols}
    elif name == "LightGBM":
        x_train, x_test = lgbm_frames()
        factories = [lambda q=q: LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=300, num_leaves=31,
            learning_rate=0.05, random_state=42, verbosity=-1,
        ) for q in QUANTILES]
        fit_kwargs = {"categorical_feature": cat_cols}
    elif name == "XGBoost":
        x_train, x_test = encoded_frames()
        factories = [lambda q=q: XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=q,
            n_estimators=300, max_depth=7, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, enable_categorical=True,
            tree_method="hist", random_state=42, n_jobs=-1,
        ) for q in QUANTILES]
        fit_kwargs = {}
    else:
        raise ValueError(name)

    for factory in factories:
        model = factory()
        model.fit(x_train, y_train, **fit_kwargs)
        predictions.append(np.exp(model.predict(x_test)))
    return np.minimum(*predictions), np.maximum(*predictions)


def metrics(lower, upper):
    actual = test["buy_price"].to_numpy()
    midpoint = (lower + upper) / 2
    quarter_end = test["evaluation_date"].max()
    quarter_mask = test["evaluation_date"].between(
        quarter_end - pd.DateOffset(months=3), quarter_end
    ).to_numpy()
    gm2_sim = test["gm2"].to_numpy() + actual - midpoint
    gm2_actual_q = test.loc[quarter_mask, "gm2"].sum()
    gm2_sim_q = gm2_sim[quarter_mask].sum()
    return {
        "coverage_pct": round(100 * np.mean((actual >= lower) & (actual <= upper)), 2),
        "median_width_pct": round(100 * np.median((upper - lower) / actual), 2),
        "median_mid_ape_pct": round(100 * np.median(np.abs(midpoint - actual) / actual), 2),
        "quarter_gm2_delta_m_kzt": round((gm2_sim_q - gm2_actual_q) / 1_000_000, 1),
        "quarter_loss_deals_pct": round(100 * np.mean(gm2_sim[quarter_mask] < 0), 2),
    }


results = {}
for algorithm in ("HistGradientBoosting", "CatBoost", "LightGBM", "XGBoost"):
    lower, upper = fit_pair(algorithm)
    results[algorithm] = metrics(lower, upper)
    print(algorithm, results[algorithm], flush=True)

output = ROOT / "experiments" / "booster_results.json"
output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved {output}")
