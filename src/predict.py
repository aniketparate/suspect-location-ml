"""Inference script with LightGBM, HMM, and baseline table outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict location probabilities.")
    parser.add_argument("--date", type=str, required=True, help="Date in YYYY-MM-DD format.")
    parser.add_argument("--hour", type=int, default=None, help="Optional single hour 0-23.")
    parser.add_argument("--lgbm-model", type=str, default="models/lgbm_model.pkl", help="LightGBM model path.")
    parser.add_argument("--hmm-weekday", type=str, default="models/hmm_weekday.pkl", help="Weekday HMM bundle path.")
    parser.add_argument("--hmm-weekend", type=str, default="models/hmm_weekend.pkl", help="Weekend HMM bundle path.")
    parser.add_argument("--prob-table", type=str, default="models/prob_table.pkl", help="Baseline table model path.")
    parser.add_argument("--centroids", type=str, default="data/processed/centroids.json", help="Centroids JSON path.")
    parser.add_argument("--gps-data", type=str, default="data/raw/gps_log.csv", help="GPS CSV for uncertainty.")
    return parser.parse_args()


def normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    if not probs:
        return {}
    total = float(sum(probs.values()))
    if total <= 0:
        return probs
    return {k: float(v / total) for k, v in probs.items()}


def top_label_and_prob(probs: dict[str, float]) -> tuple[str, float]:
    if not probs:
        return "n/a", 0.0
    label, prob = max(probs.items(), key=lambda kv: kv[1])
    return label, float(prob)


def format_pct(prob: float) -> str:
    return f"{prob * 100:.0f}%"


def build_feature_row(day_of_week: int, hour: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
                "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
                "dow_sin": float(np.sin(2 * np.pi * day_of_week / 7.0)),
                "dow_cos": float(np.cos(2 * np.pi * day_of_week / 7.0)),
                "is_weekend": int(day_of_week >= 5),
            }
        ]
    )


def load_centroids(path: str | Path) -> dict[str, list[float]]:
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, list[float]] = {}
    if isinstance(payload, dict):
        for k, v in payload.items():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                out[str(k)] = [float(v[0]), float(v[1])]
    return out


def resolve_coordinates(place_probs: dict[str, float], centroids: dict[str, list[float]]) -> tuple[float, float, str, float]:
    probs = normalize_probs(place_probs)
    if not probs:
        return float("nan"), float("nan"), "n/a", 0.0
    top_place, top_prob = top_label_and_prob(probs)
    expected_lat = float(sum(prob * centroids[p][0] for p, prob in probs.items() if p in centroids))
    expected_lon = float(sum(prob * centroids[p][1] for p, prob in probs.items() if p in centroids))
    return expected_lat, expected_lon, top_place, top_prob


def compute_place_uncertainty(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if "place_label" not in df.columns:
        return out
    work = df.dropna(subset=["place_label", "latitude", "longitude"]).copy()
    for place, group in work.groupby("place_label"):
        lat_std = float(group["latitude"].std(ddof=0))
        lon_std = float(group["longitude"].std(ddof=0))
        out[str(place)] = float(np.sqrt((lat_std * 111000) ** 2 + (lon_std * 111000) ** 2))
    return out


def resolve_uncertainty(place_probs: dict[str, float], place_uncertainty: dict[str, float]) -> float:
    probs = normalize_probs(place_probs)
    if not probs:
        return float("nan")
    return float(sum(prob * float(place_uncertainty.get(place, 50.0)) for place, prob in probs.items()))


def make_hmm_observations(day_of_week: int) -> np.ndarray:
    is_weekend = int(day_of_week >= 5)
    obs = np.array([(hour // 2) * 2 + is_weekend for hour in range(24)], dtype=int)
    return obs


def hmm_posteriors(bundle: dict[str, object], day_of_week: int) -> list[dict[str, float]]:
    model = bundle["model"]
    state_to_label = {int(k): str(v) for k, v in bundle["state_to_label"].items()}
    classes = [str(c) for c in bundle["classes"]]

    obs = make_hmm_observations(day_of_week)
    X = obs.reshape(-1, 1)

    try:
        post = model.predict_proba(X)
    except Exception:
        n_obs = int(obs.max()) + 1
        X_onehot = np.zeros((len(obs), n_obs), dtype=int)
        X_onehot[np.arange(len(obs)), obs] = 1
        _, post = model.score_samples(X_onehot)

    hour_probs: list[dict[str, float]] = []
    n_states = post.shape[1]
    for h in range(post.shape[0]):
        label_probs = {c: 0.0 for c in classes}
        for state in range(n_states):
            label = state_to_label.get(state)
            if label is None:
                continue
            label_probs[label] = label_probs.get(label, 0.0) + float(post[h, state])
        hour_probs.append(normalize_probs(label_probs))
    return hour_probs


def main() -> None:
    args = parse_args()
    try:
        date_obj = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("--date must be in YYYY-MM-DD format.") from exc
    if args.hour is not None and not (0 <= args.hour <= 23):
        raise ValueError("--hour must be between 0 and 23.")

    lgbm = joblib.load(args.lgbm_model)
    prob_table = joblib.load(args.prob_table)
    hmm_bundle = joblib.load(args.hmm_weekend if date_obj.weekday() >= 5 else args.hmm_weekday)
    centroids = load_centroids(args.centroids)
    gps_df = pd.read_csv(args.gps_data)
    place_uncertainty = compute_place_uncertainty(gps_df)

    day_of_week = date_obj.weekday()
    day_type = date_obj.strftime("%A")
    labels = sorted(set(list(lgbm.classes_) + list(hmm_bundle["classes"]) + list(centroids.keys())))
    hours = [args.hour] if args.hour is not None else list(range(24))

    hmm_probs_by_hour = hmm_posteriors(hmm_bundle, day_of_week)
    hourly_payload: dict[str, dict[str, object]] = {}
    rows: list[tuple[str, str, float, str, float, str, float]] = []

    for hour in hours:
        X = build_feature_row(day_of_week, hour)
        lgbm_arr = lgbm.predict_proba(X)[0]
        lgbm_probs = {str(c): float(p) for c, p in zip(lgbm.classes_, lgbm_arr, strict=False)}
        lgbm_probs = {label: float(lgbm_probs.get(label, 0.0)) for label in labels}
        lgbm_probs = normalize_probs(lgbm_probs)

        hmm_probs = {label: float(hmm_probs_by_hour[hour].get(label, 0.0)) for label in labels}
        hmm_probs = normalize_probs(hmm_probs)

        table_probs = {
            label: float(prob_table.get(day_of_week, {}).get(hour, {}).get(label, 0.0))
            for label in labels
        }
        table_probs = normalize_probs(table_probs)

        exp_lat, exp_lon, top_place, top_prob = resolve_coordinates(lgbm_probs, centroids)
        uncertainty_m = resolve_uncertainty(lgbm_probs, place_uncertainty)

        lgbm_top, lgbm_p = top_label_and_prob(lgbm_probs)
        hmm_top, hmm_p = top_label_and_prob(hmm_probs)
        table_top, table_p = top_label_and_prob(table_probs)

        hour_key = f"{hour:02d}:00"
        hourly_payload[hour_key] = {
            "lgbm": lgbm_probs,
            "hmm": hmm_probs,
            "table": table_probs,
            "top_place": top_place,
            "top_place_probability": float(top_prob),
            "expected_coordinates": {
                "lat": round(float(exp_lat), 6) if np.isfinite(exp_lat) else None,
                "lon": round(float(exp_lon), 6) if np.isfinite(exp_lon) else None,
            },
            "coordinate_uncertainty_meters": round(float(uncertainty_m), 2) if np.isfinite(uncertainty_m) else None,
        }
        rows.append((hour_key, lgbm_top, lgbm_p, hmm_top, hmm_p, table_top, table_p))

    print(f"Prediction for: {args.date} ({day_type})")
    print("Hour | LightGBM | Prob | HMM | Prob | Table | Prob")
    print("-----------------------------------------------------")
    for hour_key, lgbm_top, lgbm_p, hmm_top, hmm_p, table_top, table_p in rows:
        print(
            f"{hour_key} | {lgbm_top:<9} | {format_pct(lgbm_p):>4} | "
            f"{hmm_top:<9} | {format_pct(hmm_p):>4} | {table_top:<9} | {format_pct(table_p):>4}"
        )

    out = {"date": args.date, "day_type": day_type, "hourly": hourly_payload}
    out_path = Path("outputs") / "predictions" / f"{args.date}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved prediction JSON to: {out_path}")


if __name__ == "__main__":
    main()
