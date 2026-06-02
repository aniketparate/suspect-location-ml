"""Cluster GPS points into place IDs/labels using Cartesian-meter DBSCAN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances


EARTH_RADIUS_M = 6_371_000
EARTH_RADIUS_KM = 6371.0088

KNOWN_LOCATIONS: dict[str, tuple[float, float]] = {
    "home": (19.1136, 72.8697),
    "office": (19.0176, 72.8562),
    "gym": (19.1298, 72.8364),
    "local_market": (19.1050, 72.8756),
    "mall": (19.0883, 72.8276),
    "friend_area": (19.0595, 72.8307),
    "cafe": (19.0210, 72.8580),
    "dhaba": (19.1243, 72.8598),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster and label GPS points.")
    parser.add_argument("--input", type=str, default="data/raw/gps_log.csv", help="Input CSV path.")
    parser.add_argument("--output", type=str, default="data/raw/gps_log.csv", help="Output CSV path.")
    parser.add_argument(
        "--centroids-out",
        type=str,
        default="data/processed/centroids.json",
        help="Output centroids JSON path.",
    )
    return parser.parse_args()


def to_cartesian_meters(df: pd.DataFrame) -> np.ndarray:
    lat_rad = np.radians(df["latitude"].to_numpy(dtype=float))
    lon_rad = np.radians(df["longitude"].to_numpy(dtype=float))
    lat_center = float(np.radians(df["latitude"].mean()))
    lon_center = float(np.radians(df["longitude"].mean()))

    x = EARTH_RADIUS_M * np.cos(lat_center) * (lon_rad - lon_center)
    y = EARTH_RADIUS_M * (lat_rad - float(np.radians(df["latitude"].mean())))
    return np.column_stack([x, y])


def run_dbscan_with_retries(df: pd.DataFrame) -> tuple[np.ndarray, float, int]:
    coords_meters = to_cartesian_meters(df)
    last_labels: np.ndarray | None = None
    last_eps = 120.0

    for min_samples in (50, 30):
        for eps in (80, 100, 120):
            db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
            labels = db.fit_predict(coords_meters).astype(int)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"DBSCAN attempt eps={eps}, min_samples={min_samples}: clusters_found={n_clusters}")
            last_labels = labels
            last_eps = float(eps)
            if n_clusters >= 6:
                return labels, float(eps), int(min_samples)
        if min_samples == 50:
            n_clusters_last = len(set(last_labels)) - (1 if -1 in last_labels else 0) if last_labels is not None else 0
            if n_clusters_last < 6:
                print("Clusters < 6 with min_samples=50. Retrying with min_samples=30.")

    assert last_labels is not None
    return last_labels, last_eps, 30


def compute_cluster_centroids(df: pd.DataFrame) -> pd.DataFrame:
    non_noise = df[df["place_id"] != -1]
    if non_noise.empty:
        return pd.DataFrame(columns=["place_id", "centroid_lat", "centroid_lon", "cluster_size"])

    return (
        non_noise.groupby("place_id", as_index=False)
        .agg(
            centroid_lat=("latitude", "mean"),
            centroid_lon=("longitude", "mean"),
            cluster_size=("place_id", "size"),
        )
        .sort_values("place_id")
        .reset_index(drop=True)
    )


def auto_label_centroids(centroids: pd.DataFrame) -> pd.DataFrame:
    if centroids.empty:
        return centroids.assign(place_label=pd.Series(dtype=str), distance_km=pd.Series(dtype=float))

    known_names = list(KNOWN_LOCATIONS.keys())
    known_coords_rad = np.radians(np.array([KNOWN_LOCATIONS[n] for n in known_names], dtype=float))
    centroid_coords_rad = np.radians(centroids[["centroid_lat", "centroid_lon"]].to_numpy(dtype=float))

    dists_rad = haversine_distances(centroid_coords_rad, known_coords_rad)
    nearest_idx = dists_rad.argmin(axis=1)
    nearest_km = dists_rad[np.arange(len(centroids)), nearest_idx] * EARTH_RADIUS_KM

    out = centroids.copy()
    out["place_label"] = [known_names[i] for i in nearest_idx]
    out["distance_km"] = nearest_km
    return out


def smooth_labels(df: pd.DataFrame, min_stay_minutes: int = 20) -> tuple[pd.DataFrame, int]:
    rows_per_stay = max(1, min_stay_minutes // 5)
    labels = df["place_label"].to_numpy(dtype=object)
    if len(labels) == 0:
        return df, 0

    result = labels.copy()
    changes = 0
    i = 0
    n = len(result)

    while i < n:
        j = i + 1
        while j < n and result[j] == result[i]:
            j += 1
        run_length = j - i
        if run_length < rows_per_stay:
            prev_exists = i > 0
            next_exists = j < n
            if prev_exists or next_exists:
                if prev_exists and next_exists:
                    # Pick label from longer adjacent segment.
                    p_start = i - 1
                    while p_start > 0 and result[p_start - 1] == result[i - 1]:
                        p_start -= 1
                    prev_len = i - p_start

                    n_end = j + 1
                    while n_end < n and result[n_end] == result[j]:
                        n_end += 1
                    next_len = n_end - j
                    replacement = result[i - 1] if prev_len >= next_len else result[j]
                elif prev_exists:
                    replacement = result[i - 1]
                else:
                    replacement = result[j]

                changed = int(np.sum(result[i:j] != replacement))
                if changed > 0:
                    result[i:j] = replacement
                    changes += changed
        i = j

    out = df.copy()
    out["place_label"] = result
    return out, changes


def assign_labels_and_fill_noise(
    df: pd.DataFrame, labeled_centroids: pd.DataFrame
) -> tuple[pd.DataFrame, int, int]:
    mapping = {int(r.place_id): str(r.place_label) for r in labeled_centroids.itertuples(index=False)}
    out = df.copy()
    out["place_label"] = out["place_id"].map(mapping)

    noise_count = int((out["place_id"] == -1).sum())
    out.loc[out["place_id"] == -1, "place_label"] = np.nan

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["place_label"] = out["place_label"].ffill().bfill()
    out, smoothed_count = smooth_labels(out, min_stay_minutes=20)
    return out, noise_count, smoothed_count


def save_known_centroids_json(path: Path) -> None:
    payload = {name: [coords[0], coords[1]] for name, coords in KNOWN_LOCATIONS.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_validation_summary(
    df: pd.DataFrame,
    centroids_labeled: pd.DataFrame,
    noise_count: int,
    smoothed_count: int,
) -> None:
    n_clusters = int(df.loc[df["place_id"] != -1, "place_id"].nunique())
    print(f"\nNumber of clusters found (excluding -1): {n_clusters}")
    print(f"Label flickers smoothed out (rows reassigned): {smoothed_count}")

    print("\nplace_label value_counts():")
    counts = df["place_label"].value_counts(dropna=False)
    print(counts.to_string())

    distinct_labels = int(df["place_label"].nunique(dropna=True))
    if distinct_labels < 6:
        print(f"WARNING: Only {distinct_labels} distinct labels found; expected 6+.")

    print("\nCentroid of each labeled cluster:")
    if centroids_labeled.empty:
        print("  No labeled clusters found.")
    else:
        for row in centroids_labeled.itertuples(index=False):
            print(
                f"  cluster={int(row.place_id):>3} label={row.place_label:<12} "
                f"centroid=({row.centroid_lat:.6f}, {row.centroid_lon:.6f}) "
                f"size={int(row.cluster_size):>5} nearest={row.distance_km:.3f} km"
            )

    print(f"\nNumber of noise points that were ffill'd: {noise_count}")

    present_labels = set(df["place_label"].dropna().astype(str).unique().tolist())
    missing = [name for name in KNOWN_LOCATIONS if name not in present_labels]
    if missing:
        print(f"WARNING: Known locations missing from place_label: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    centroids_path = Path(args.centroids_out)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    # Drop stale clustering columns if they exist from a previous run
    df = df.drop(columns=["place_id", "place_label"], errors="ignore")
    required = {"timestamp", "latitude", "longitude"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    labels, chosen_eps, chosen_min_samples = run_dbscan_with_retries(df)
    print(f"Using clustering result from eps={chosen_eps:.0f}, min_samples={chosen_min_samples}")
    df["place_id"] = labels

    centroids = compute_cluster_centroids(df)
    centroids_labeled = auto_label_centroids(centroids)
    out_df, noise_count, smoothed_count = assign_labels_and_fill_noise(df, centroids_labeled)

    print_validation_summary(out_df, centroids_labeled, noise_count, smoothed_count)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved updated CSV to: {out_path}")

    save_known_centroids_json(centroids_path)
    print(f"Saved stable centroids JSON to: {centroids_path}")


if __name__ == "__main__":
    main()
