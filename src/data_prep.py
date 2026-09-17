import argparse
import os
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
from feature_contract import ALLOWED_FEATURES, CAR_CARD_FEATURES, CONTEXT_FEATURES, assert_no_leakage

SCOPE_TRANSACTION_GROUPS = ["purchase", "trade_in", "trade_up"]
TAIL_CUTOFF_DAYS = 15

MARKET_JOIN_MAP = {"mark": "brand", "model": "model", "car_year": "car_year"}

STATE_MAP_5_TO_3 = {
    "Ужасное": 1, "Плохое": 1, "Среднее": 2, "Хорошее": 3, "Отличное": 3,
}


def load_raw(data_dir: str):
    ev = pd.read_csv(os.path.join(data_dir, "evaluations.csv.gz"), low_memory=False)
    co = pd.read_csv(os.path.join(data_dir, "car_outcomes.csv"))
    mk = pd.read_csv(os.path.join(data_dir, "market_kolesa.csv.gz"), low_memory=False)
    return ev, co, mk


def build_market_monthly(mk: pd.DataFrame) -> pd.DataFrame:
    mk = mk.copy()
    mk["_w_price"] = mk["avg_price"] * mk["ads_cnt"]
    mk["_w_mileage"] = mk["avg_mileage"] * mk["ads_cnt"]
    grp = mk.groupby(["month_start", "brand", "model", "car_year"], as_index=False).agg(
        market_ads_cnt=("ads_cnt", "sum"),
        _w_price_sum=("_w_price", "sum"),
        _w_mileage_sum=("_w_mileage", "sum"),
    )
    grp["market_avg_price"] = grp["_w_price_sum"] / grp["market_ads_cnt"]
    grp["market_avg_mileage"] = grp["_w_mileage_sum"] / grp["market_ads_cnt"]
    grp = grp.drop(columns=["_w_price_sum", "_w_mileage_sum"])
    grp["month_start"] = pd.to_datetime(grp["month_start"])
    return grp


def attach_market_features(evals: pd.DataFrame, market_monthly: pd.DataFrame) -> pd.DataFrame:
    evals = evals.copy()
    eval_month = evals["evaluation_date"].dt.to_period("M").dt.to_timestamp()
    evals["_market_month"] = eval_month - pd.DateOffset(months=1)

    merged = evals.merge(
        market_monthly,
        left_on=["_market_month", "mark", "model", "car_year"],
        right_on=["month_start", "brand", "model", "car_year"],
        how="left",
        suffixes=("", "_mk"),
    )
    merged["has_price_quantiles"] = False 
    merged["market_median_price_w"] = np.nan
    merged["market_q1_price"] = np.nan
    merged["market_q3_price"] = np.nan
    return merged.drop(columns=["_market_month", "month_start", "brand"], errors="ignore")


def build_dataset(data_dir: str) -> pd.DataFrame:
    ev, co, mk = load_raw(data_dir)

    ev["evaluation_date"] = pd.to_datetime(ev["evaluation_date"])
    ev["purchase_date"] = pd.to_datetime(ev["purchase_date"])
    max_date = ev["evaluation_date"].max()
    cutoff = max_date - pd.Timedelta(days=TAIL_CUTOFF_DAYS)

    scoped = ev[
        ev["purchase_date"].notna()
        & (ev["purchase_date"] <= cutoff)
        & ev["transaction_group"].isin(SCOPE_TRANSACTION_GROUPS)
    ].copy()

    co_small = co[["deal_key", "buy_price", "gm2", "status", "date_buy"]].copy()
    merged = scoped.merge(co_small, on="deal_key", how="inner")

    print(f"[data_prep] evaluations всего: {len(ev)}")
    print(f"[data_prep] в скоупе (purchase/trade_in/trade_up, старше {TAIL_CUTOFF_DAYS} дн, purchased): {len(scoped)}")
    print(f"[data_prep] нашли исход в car_outcomes (deal_key join): {len(merged)} "
          f"({len(merged) / max(len(scoped), 1):.1%} покрытие)")

    merged = merged[merged["total_car_state"] != "0"]  # брак записи "0", data_diary п.5
    merged["total_car_state"] = merged["total_car_state"].map(STATE_MAP_5_TO_3)

    market_monthly = build_market_monthly(mk)
    merged = attach_market_features(merged, market_monthly)

    feature_cols = list(CAR_CARD_FEATURES) + list(CONTEXT_FEATURES)
    feature_cols += ["market_avg_price", "market_ads_cnt", "market_avg_mileage",
                      "has_price_quantiles", "market_median_price_w",
                      "market_q1_price", "market_q3_price"]
    assert_no_leakage(feature_cols)
    feature_cols.remove("transaction_subgroup")

    out = merged[["deal_key", "evaluation_date", "buy_price", "gm2"] + feature_cols].copy()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset_task")
    parser.add_argument("--out", default="dataset_prepared.csv")
    args = parser.parse_args()

    df = build_dataset(args.data_dir)
    print(df.shape)
    print(df.isna().mean().sort_values(ascending=False).head(10))
    df.to_csv(args.out, index=False)
    print(f"[data_prep] сохранено: {args.out}")
