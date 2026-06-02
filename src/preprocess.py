"""Feature engineering pipeline for location prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build location prediction features.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/gps_log.csv",
        help="Path to input clustered GPS CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/features.csv",
        help="Path to output engineered feature CSV.",
    )
    parser.add_argument(
        "--encoder-out",
        type=str,
        default="models/label_encoder.pkl",
        help="Path to save sklearn LabelEncoder via joblib.",
    )
    return parser.parse_args()


def add_time_block(hour: pd.Series) -> pd.Series:
    bins = [0, 6, 12, 18, 24]
    labels = ["night", "morning", "afternoon", "evening"]
    return pd.cut(hour, bins=bins, labels=labels, right=False, include_lowest=True).astype(str)


def print_class_distribution(df: pd.DataFrame) -> None:
    print("\nClass distribution of place_label:")
    counts = df["place_label"].value_counts(dropna=False)
    for label, count in counts.items():
        print(f"  {label}: {int(count)}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    encoder_path = Path(args.encoder_out)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")
    if "place_label" not in df.columns:
        raise ValueError("Missing required column: place_label")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.dropna(subset=["place_label"]).copy()

    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["minute"] = df["timestamp"].dt.minute.astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5)
    df["month"] = df["timestamp"].dt.month.astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["time_block"] = add_time_block(df["hour"])

    encoder = LabelEncoder()
    df["place_id_encoded"] = encoder.fit_transform(df["place_label"])

    print_class_distribution(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    joblib.dump(encoder, encoder_path)

    print(f"\nSaved features CSV to: {output_path}")
    print(f"Saved label encoder to: {encoder_path}")


if __name__ == "__main__":
    main()
