from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from feature_contract import ALLOWED_FEATURES, assert_no_leakage


QUANTILES = {"price_from": 0.25, "price_to": 0.75}
TARGET_ENCODED = ("model", "generation")
DROP_STUBS = {
    "market_median_price_w",
    "market_q1_price",
    "market_q3_price",
    "has_price_quantiles",
    "transaction_subgroup",
}
SMOOTHING = 20


@dataclass
class PriceBandBundle:
    models: dict
    raw_feature_cols: list[str]
    model_feature_cols: list[str]
    target_encoding: dict[str, dict]
    global_log_mean: float
    categorical_levels: dict[str, list]
    trained_through: str

    def prepare(self, records: pd.DataFrame, as_of: datetime | None = None) -> pd.DataFrame:
        assert_no_leakage(records.columns)
        missing = sorted(set(self.raw_feature_cols) - set(records.columns))
        if missing:
            raise ValueError(f"Не хватает признаков модели: {missing}")

        features = records[self.raw_feature_cols].copy()
        inference_time = as_of or datetime.now()
        features["month_idx"] = (inference_time.year - 2024) * 12 + inference_time.month

        for column in TARGET_ENCODED:
            mapping = self.target_encoding[column]
            features[f"{column}_te"] = features[column].map(mapping).fillna(self.global_log_mean)
            features.drop(columns=column, inplace=True)

        for column, levels in self.categorical_levels.items():
            features[column] = pd.Categorical(features[column], categories=levels)
        return features[self.model_feature_cols]

    def predict(self, records: pd.DataFrame, as_of: datetime | None = None) -> pd.DataFrame:
        features = self.prepare(records, as_of=as_of)
        raw = {name: np.exp(model.predict(features)) for name, model in self.models.items()}
        return pd.DataFrame(
            {
                "price_from": np.minimum(raw["price_from"], raw["price_to"]),
                "price_to": np.maximum(raw["price_from"], raw["price_to"]),
            },
            index=records.index,
        )


def train_bundle(dataset_path: str | Path) -> PriceBandBundle:
    df = pd.read_csv(dataset_path, parse_dates=["evaluation_date"])
    cutoff = df["evaluation_date"].quantile(0.8)
    train = df[df["evaluation_date"] < cutoff].copy()
    train["month_idx"] = (
        (train["evaluation_date"].dt.year - 2024) * 12
        + train["evaluation_date"].dt.month
    )

    raw_feature_cols = [
        column for column in ALLOWED_FEATURES
        if column in df.columns and column not in DROP_STUBS
    ]
    assert_no_leakage(raw_feature_cols)
    features = train[raw_feature_cols + ["month_idx"]].copy()
    target = np.log(train["buy_price"])
    global_log_mean = float(target.mean())
    encodings = {}

    for column in TARGET_ENCODED:
        stats = target.groupby(features[column]).agg(["mean", "count"])
        smoothed = (
            stats["mean"] * stats["count"] + global_log_mean * SMOOTHING
        ) / (stats["count"] + SMOOTHING)
        encodings[column] = smoothed.to_dict()
        features[f"{column}_te"] = features[column].map(smoothed).fillna(global_log_mean)
        features.drop(columns=column, inplace=True)

    categorical_columns = features.select_dtypes(include=["object", "bool"]).columns
    categorical_levels = {}
    for column in categorical_columns:
        features[column] = features[column].astype("category")
        categorical_levels[column] = features[column].cat.categories.tolist()

    models = {}
    for name, quantile in QUANTILES.items():
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            categorical_features="from_dtype",
            max_iter=300,
            learning_rate=0.05,
            random_state=42,
        )
        model.fit(features, target)
        models[name] = model

    return PriceBandBundle(
        models=models,
        raw_feature_cols=raw_feature_cols,
        model_feature_cols=features.columns.tolist(),
        target_encoding=encodings,
        global_log_mean=global_log_mean,
        categorical_levels=categorical_levels,
        trained_through=train["evaluation_date"].max().isoformat(),
    )


def save_bundle(bundle: PriceBandBundle, artifact_path: str | Path) -> None:
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)


def load_bundle(artifact_path: str | Path) -> PriceBandBundle:
    return joblib.load(artifact_path)
