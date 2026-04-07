"""
serving/main.py

FastAPI inference service — loads the Production model from
MLflow Model Registry on startup and serves predictions.
"""

import os
import logging
import warnings
from contextlib import asynccontextmanager
from typing import Any

# Suppress MLflow environment mismatch warnings
os.environ["MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING"] = "TRUE"
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5005")
MODEL_NAME          = os.getenv("MODEL_NAME", "house_price_prediction")
MODEL_STAGE         = os.getenv("MODEL_STAGE", "Production")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

model_cache: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"Loading model '{MODEL_NAME}' @ stage='{MODEL_STAGE}'")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_cache["model"] = mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}/{MODEL_STAGE}",
        suppress_warnings=True,
    )
    log.info("Model loaded successfully")
    yield
    model_cache.clear()


app = FastAPI(
    title="House Price Prediction API",
    description="MLOps Intensive — serving layer",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    features: list[dict[str, Any]] = Field(
        ...,
        example=[{
            "area": 120.5, "bedrooms": 3, "bathrooms": 2,
            "floor": 5, "age": 10, "distance_bts": 0.8,
            "distance_center": 5.2, "parking": 1,
            "quality": "good", "direction": "north",
        }],
    )


class PredictResponse(BaseModel):
    predictions: list[float]
    model_name: str
    model_stage: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_stage: str


@app.get("/health", response_model=HealthResponse)
async def health():
    loaded = "model" in model_cache
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if "model" not in model_cache:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        df    = pd.DataFrame(req.features)
        preds = model_cache["model"].predict(df)
        return PredictResponse(
            predictions=[round(float(p), 2) for p in preds],
            model_name=MODEL_NAME,
            model_stage=MODEL_STAGE,
        )
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "House Price API — see /docs for Swagger UI"}
