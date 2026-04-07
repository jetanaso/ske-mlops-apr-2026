"""
generate_initial_data.py

Run this script once before the session to create and upload
initial_data.csv and data.csv (reference) to MinIO.

Usage:
    pip install minio pandas numpy
    python generate_initial_data.py
"""

import io
import random
import numpy as np
import pandas as pd
from minio import Minio

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS   = "admin"
MINIO_SECRET   = "1qaz2wsx"
BUCKET         = "data"
N_ROWS         = 20_000
RANDOM_SEED    = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def generate_house_data(n: int) -> pd.DataFrame:
    """Generate synthetic house price data with realistic correlations."""
    area         = np.random.normal(120, 40, n).clip(30, 400)
    bedrooms     = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.25, 0.45, 0.20, 0.05])
    bathrooms    = (bedrooms * np.random.uniform(0.5, 1.0, n)).astype(int).clip(1, 5)
    floor        = np.random.choice(range(1, 31), n)
    age          = np.random.randint(0, 30, n)
    dist_bts     = np.random.exponential(1.5, n).clip(0.1, 15)
    dist_center  = np.random.exponential(8, n).clip(0.5, 50)
    parking      = np.random.choice([0, 1, 2], n, p=[0.1, 0.6, 0.3])
    quality      = np.random.choice(
        ["bad", "fair", "average", "good", "excellent"], n,
        p=[0.05, 0.10, 0.35, 0.35, 0.15],
    )
    direction    = np.random.choice(["north", "south", "east", "west"], n)

    quality_map  = {"bad": 1, "fair": 2, "average": 3, "good": 4, "excellent": 5}
    q_num        = np.array([quality_map[q] for q in quality])

    base_price = (
        area          * 25_000
        + bedrooms    * 150_000
        + bathrooms   * 80_000
        + floor       * 5_000
        - age         * 20_000
        - dist_bts    * 200_000
        - dist_center * 15_000
        + parking     * 100_000
        + q_num       * 300_000
    )
    noise = np.random.normal(0, base_price * 0.08)
    price = (base_price + noise).clip(500_000, 50_000_000)

    df = pd.DataFrame({
        "area":            area.round(1),
        "bedrooms":        bedrooms,
        "bathrooms":       bathrooms,
        "floor":           floor,
        "age":             age,
        "distance_bts":    dist_bts.round(2),
        "distance_center": dist_center.round(2),
        "parking":         parking,
        "quality":         quality,
        "direction":       direction,
        "target":          price.round(0).astype(int),
    })

    # Introduce ~3% missing values in selected columns
    for col in ["area", "floor", "age", "distance_bts", "quality"]:
        df.loc[np.random.rand(n) < 0.03, col] = np.nan

    return df


def upload_to_minio(df: pd.DataFrame) -> None:
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS,
                   secret_key=MINIO_SECRET, secure=False)

    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)
        print(f"Created bucket: {BUCKET}")

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    for filename in ["initial_data.csv", "data.csv"]:
        client.put_object(
            BUCKET, filename,
            data=io.BytesIO(csv_bytes),
            length=len(csv_bytes),
            content_type="application/csv",
        )
        print(f"Uploaded {filename} ({len(df):,} rows) -> minio/{BUCKET}/")


if __name__ == "__main__":
    print(f"Generating {N_ROWS:,} rows...")
    df = generate_house_data(N_ROWS)
    print(df.describe())
    print(f"\nMissing values:\n{df.isnull().sum()}")
    upload_to_minio(df)
    print("\nDone. Ready to trigger pretrain DAG.")
