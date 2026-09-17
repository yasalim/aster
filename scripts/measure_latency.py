"""Measure warm single-request latency against the in-process HTTP app."""

import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))
from api import app


def json_value(value):
    return None if pd.isna(value) else value.item() if hasattr(value, "item") else value


data = pd.read_csv(ROOT / "dataset_prepared.csv").tail(200)
allowed = set(app.openapi()["components"]["schemas"]["PriceRequest"]["properties"])
records = [
    {column: json_value(value) for column, value in row.items() if column in allowed}
    for row in data.to_dict(orient="records")
]

client = TestClient(app)
for record in records[:10]:
    assert client.post("/price", json=record).status_code == 200

latencies_ms = []
for record in records:
    started = perf_counter()
    response = client.post("/price", json=record)
    latencies_ms.append((perf_counter() - started) * 1000)
    assert response.status_code == 200, response.text

summary = {
    "requests": len(latencies_ms),
    "p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
    "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
    "max_ms": round(float(max(latencies_ms)), 2),
    "method": "FastAPI TestClient, sequential warm requests, local machine",
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
