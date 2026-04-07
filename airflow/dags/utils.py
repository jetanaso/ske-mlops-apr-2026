import io
import os
import pandas as pd
from dotenv import load_dotenv

from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    DropCorrelatedFeatures,
)
from feature_engine.imputation import MeanMedianImputer, RandomSampleImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.encoding import OneHotEncoder as Ohe

from minio import Minio

load_dotenv("/opt/.env")
MINIO_USER = os.getenv("AWS_ACCESS_KEY_ID", "admin")
MINIO_PASS = os.getenv("AWS_SECRET_ACCESS_KEY", "1qaz2wsx")
MINIO_HOST = "s3:9000"


class QualityTransformer(TransformerMixin):
    """Ordinal-encode the 'quality' column using a fixed mapping."""

    MAPPING = {"bad": 1, "fair": 2, "average": 3, "good": 4, "excellent": 5}

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["quality"] = X_["quality"].map(self.MAPPING)
        return X_


def _minio_client() -> Minio:
    return Minio(MINIO_HOST, access_key=MINIO_USER, secret_key=MINIO_PASS, secure=False)


def download_from_s3(bucket_name: str, filename: str) -> pd.DataFrame:
    """Download a CSV file from MinIO and return it as a DataFrame."""
    client = _minio_client()
    response = client.get_object(bucket_name, filename)
    try:
        df = pd.read_csv(response)
    finally:
        response.close()
        response.release_conn()
    return df


def upload_to_s3(df: pd.DataFrame, bucket_name: str, filename: str) -> None:
    """Upload a DataFrame as CSV to MinIO."""
    client = _minio_client()
    csv_data = df.to_csv(index=False).encode("utf-8")
    client.put_object(
        bucket_name, filename,
        data=io.BytesIO(csv_data),
        length=len(csv_data),
        content_type="application/csv",
    )


def pipeline_prep(n_estimators: int, max_depth: int) -> Pipeline:
    """
    Build a full sklearn Pipeline:
    feature selection -> imputation -> encoding -> scaling -> RandomForest
    """
    return Pipeline([
        ("drop_constant",    DropConstantFeatures(tol=1, missing_values="ignore")),
        ("drop_duplicates",  DropDuplicateFeatures()),
        ("drop_correlated",  DropCorrelatedFeatures(method="pearson", threshold=0.85)),
        ("quality_encode",   QualityTransformer()),
        ("impute_numeric",   MeanMedianImputer(imputation_method="mean")),
        ("impute_categoric",  RandomSampleImputer()),
        ("robust_scale",     SklearnTransformerWrapper(RobustScaler())),
        ("one_hot_encode",   Ohe()),
        ("model",            RandomForestRegressor(
                                 n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 n_jobs=-1,
                                 random_state=42,
                             )),
    ])
