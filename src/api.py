"""HTTP API for purchase-price band inference."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from pricing_model import PriceBandBundle, load_bundle


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "price_band_model.joblib"


class PriceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mark: str = Field(min_length=1)
    model: str = Field(min_length=1)
    car_year: int = Field(ge=1950, le=2100)
    mileage: float = Field(ge=0)
    generation: Optional[str] = None
    engine_cc: Optional[float] = Field(default=None, gt=0)
    body_type: Optional[str] = None
    engine_type: Optional[str] = None
    transmission: Optional[str] = None
    drive_type: Optional[str] = None
    color: Optional[str] = None
    steering: Optional[int] = None
    door_num: Optional[float] = Field(default=None, ge=0)
    keys_count: Optional[float] = Field(default=None, ge=0)
    on_guarantee: Optional[str] = None
    car_category: Optional[str] = None
    total_car_state: Optional[int] = Field(default=None, ge=1, le=3)
    transaction_group: Optional[str] = None
    transaction: Optional[str] = None
    is_mobile_purch: Optional[bool] = None
    is_comtrade: Optional[bool] = None
    branch_key: Optional[str] = None
    market_avg_price: Optional[float] = Field(default=None, gt=0)
    market_ads_cnt: Optional[float] = Field(default=None, ge=0)
    market_avg_mileage: Optional[float] = Field(default=None, ge=0)


class PriceResponse(BaseModel):
    price_from: float
    price_to: float
    currency: str = "KZT"
    model_trained_through: str


@lru_cache(maxsize=1)
def get_model() -> PriceBandBundle:
    if not ARTIFACT.exists():
        raise RuntimeError(
            f"Model artifact not found: {ARTIFACT}. Run `python src/train_price_model.py`."
        )
    return load_bundle(ARTIFACT)


app = FastAPI(title="Pricing Assistant", version="1.0.0")


@app.get("/health")
def health() -> dict:
    bundle = get_model()
    return {"status": "ok", "model_trained_through": bundle.trained_through}


@app.post("/price", response_model=PriceResponse)
def price(request: PriceRequest) -> PriceResponse:
    bundle = get_model()
    payload = request.model_dump()
    frame = pd.DataFrame([{column: payload.get(column) for column in bundle.raw_feature_cols}])
    prediction = bundle.predict(frame).iloc[0]
    return PriceResponse(
        price_from=round(float(prediction["price_from"]), 2),
        price_to=round(float(prediction["price_to"]), 2),
        model_trained_through=bundle.trained_through,
    )
