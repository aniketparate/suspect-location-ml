"""Diagnostics for clustered GPS labels and DBSCAN behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug clustered GPS labels and DBSCAN output.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/gps_log.csv",
        help="Path to clustered GPS CSV.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Sample size for fresh DBSCAN run.",
    )
    return parser.parse_args()


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def print_hour_sample(df: pd.DataFrame, hour: int, label: str, weekday_only: bool = False) -> None:
    sample_df = df[df["hour"] == hour].copy()
    if weekday_only:
        sample_df = sample_df[sample_df["day_of_week"] < 5]

    n = min(10, len(sample_df))
    print_section(f"Sample rows for {label} (hour={hour}, n={n})")
    if n == 0:
        print("No rows available for this filter.")
        return

    out = sample_df.sample(n=n, random_state=42)[["latitude", "longitude", "place_label"]]
    print(out.to_string(index=False))


def run_fresh_dbscan(df: pd.DataFrame, sample_size: int) -> None:
    sample_n = min(sample_size, len(df))
    work = df.sample(n=sample_n, random_state=42).copy()
    coords_deg = work[["latitude", "longitude"]].to_numpy(dtype=float)
    coords_rad = np.radians(coords_deg)

    model = DBSCAN(eps=0.0005, min_samples=20, metric="haversine")
    labels = model.fit_predict(coords_rad)
    work["dbscan_cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    print_section(f"Fresh DBSCAN on sample of {sample_n} rows")
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise}")

    non_noise = work[work["dbscan_cluster"] != -1]
    if non_noise.empty:
        print("No non-noise clusters to compute centroids.")
        return

    centroids = (
        non_noise.groupby("dbscan_cluster", as_index=False)[["latitude", "longitude"]]
        .mean()
        .rename(columns={"latitude": "centroid_lat", "longitude": "centroid_lon"})
        .sort_values("dbscan_cluster")
    )

    print("\nCluster centroids (fresh DBSCAN):")
    for row in centroids.itertuples(index=False):
        print(f"  cluster {int(row.dbscan_cluster)}: ({row.centroid_lat:.6f}, {row.centroid_lon:.6f})")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"latitude", "longitude", "place_id", "place_label", "hour", "day_of_week"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print_section("place_label value_counts()")
    print(df["place_label"].value_counts(dropna=False).to_string())

    print_section("place_id value_counts()")
    print(df["place_id"].value_counts(dropna=False).sort_index().to_string())

    print_section("Centroid per place_label")
    centroids = (
        df.groupby("place_label", dropna=False)[["latitude", "longitude"]]
        .mean()
        .rename(columns={"latitude": "mean_lat", "longitude": "mean_lon"})
    )
    print(centroids.to_string())

    print_section("Noise points with place_id == -1")
    print(int((df["place_id"] == -1).sum()))

    print_hour_sample(df, hour=2, label="night-home expectation", weekday_only=False)
    print_hour_sample(df, hour=10, label="weekday-office expectation", weekday_only=True)
    print_hour_sample(df, hour=7, label="transit expectation", weekday_only=False)

    run_fresh_dbscan(df, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
