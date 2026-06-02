"""Visualization script for location prediction results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


KNOWN_LOCATIONS: dict[str, tuple[float, float]] = {
    "home": (19.1136, 72.8697),
    "office": (19.0176, 72.8562),
    "gym": (19.1121, 72.8711),
    "local_market": (19.1150, 72.8670),
    "mall": (19.0883, 72.8276),
    "friend_area": (19.0595, 72.8307),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate prediction visualizations.")
    parser.add_argument("--date", type=str, required=True, help="Date in YYYY-MM-DD format.")
    return parser.parse_args()


def load_prediction_json(date: str) -> dict:
    path = Path("outputs") / "predictions" / f"{date}.json"
    if not path.exists():
        raise FileNotFoundError(f"Prediction JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_hour_probs(hour_entry: dict) -> dict[str, float]:
    # Backward compatible with both schemas:
    # old: {"rf": {...}, "table": {...}}
    # mid: {"place_distribution": {...}, ...}
    # current: {"lgbm": {...}, "hmm": {...}, "table": {...}, ...}
    if not isinstance(hour_entry, dict):
        return {}
    for key in ("rf", "place_distribution", "lgbm", "table"):
        probs = hour_entry.get(key)
        if isinstance(probs, dict):
            out: dict[str, float] = {}
            for k, v in probs.items():
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    out[str(k)] = 0.0
            return out
    return {}


def build_rf_matrix(prediction: dict) -> tuple[pd.DataFrame, list[str]]:
    hourly = prediction["hourly"]
    hours_sorted = sorted(hourly.keys(), key=lambda x: int(x.split(":")[0]))
    labels = sorted({label for hour_key in hours_sorted for label in get_hour_probs(hourly[hour_key]).keys()})
    if not labels:
        raise ValueError(
            "No probability distribution found in prediction JSON. "
            "Expected one of: hourly[*].rf, hourly[*].place_distribution, hourly[*].lgbm, hourly[*].table."
        )

    matrix = []
    for label in labels:
        row = [float(get_hour_probs(hourly[hour_key]).get(label, 0.0)) for hour_key in hours_sorted]
        matrix.append(row)

    values = np.asarray(matrix, dtype=float)
    df = pd.DataFrame(values, index=labels, columns=[int(h.split(":")[0]) for h in hours_sorted])
    df = df.fillna(0.0).astype(float)
    return df, labels


def make_heatmap(date: str, day_type: str, rf_hourly_df: pd.DataFrame, out_path: Path) -> None:
    weekend_day = day_type in {"Saturday", "Sunday"}
    titles = [
        f"Location Probability Heatmap — {date} (Weekday)",
        f"Location Probability Heatmap — {date} (Weekend)",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    cmap = "YlOrRd"

    # Input JSON contains a single date; reuse the same hourly RF distribution
    # for both panels and annotate which panel corresponds to the requested date type.
    for idx, ax in enumerate(axes):
        sns.heatmap(
            rf_hourly_df,
            ax=ax,
            cmap=cmap,
            cbar=True,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.2,
            linecolor="white",
        )
        suffix = " [Input day type]" if (idx == 1 and weekend_day) or (idx == 0 and not weekend_day) else ""
        ax.set_title(titles[idx] + suffix)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Place Label")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_stacked_barplot(date: str, rf_hourly_df: pd.DataFrame, out_path: Path) -> None:
    hours = rf_hourly_df.columns.to_numpy()
    labels = rf_hourly_df.index.tolist()
    bottom = np.zeros(len(hours), dtype=float)

    plt.figure(figsize=(14, 6))
    palette = sns.color_palette("tab10", n_colors=max(1, len(labels)))

    for idx, label in enumerate(labels):
        vals = rf_hourly_df.loc[label].to_numpy(dtype=float)
        plt.bar(hours, vals, bottom=bottom, label=label, color=palette[idx], width=0.85)
        bottom += vals

    plt.xticks(range(24))
    plt.ylim(0, 1.0)
    plt.xlabel("Hour of Day")
    plt.ylabel("Probability")
    plt.title(f"Hourly Location Probability (RF) — {date}")
    plt.legend(title="Place", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def place_colors(labels: list[str]) -> dict[str, str]:
    base_colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "darkred",
        "cadetblue",
        "darkgreen",
        "black",
    ]
    colors = {}
    for i, label in enumerate(labels):
        colors[label] = base_colors[i % len(base_colors)]
    return colors


def make_map(
    date: str, prediction: dict, rf_hourly_df: pd.DataFrame, labels: list[str], out_path: Path
) -> None:
    center_lat = float(np.mean([coord[0] for coord in KNOWN_LOCATIONS.values()]))
    center_lon = float(np.mean([coord[1] for coord in KNOWN_LOCATIONS.values()]))
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    colors = place_colors(labels)

    noon_key = "12:00"
    ten_pm_key = "22:00"

    # Known locations with required popup fields.
    for place_name, (lat, lon) in KNOWN_LOCATIONS.items():
        noon_prob = float(get_hour_probs(prediction["hourly"].get(noon_key, {})).get(place_name, 0.0))
        ten_pm_prob = float(get_hour_probs(prediction["hourly"].get(ten_pm_key, {})).get(place_name, 0.0))
        popup = (
            f"<b>{place_name}</b><br>"
            f"Prob @ 12:00: {noon_prob:.2%}<br>"
            f"Prob @ 22:00: {ten_pm_prob:.2%}"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=colors.get(place_name, "gray"),
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup, max_width=260),
        ).add_to(fmap)

    # Hourly top-place markers (color-coded by most probable place at each hour).
    for hour in range(24):
        hour_str = f"{hour:02d}:00"
        hour_probs = get_hour_probs(prediction["hourly"].get(hour_str, {}))
        if not hour_probs:
            continue
        top_place = max(hour_probs.items(), key=lambda kv: kv[1])[0]
        top_prob = float(hour_probs[top_place])
        if top_place not in KNOWN_LOCATIONS:
            continue
        lat, lon = KNOWN_LOCATIONS[top_place]
        folium.CircleMarker(
            location=[lat + 0.00015 * np.sin(hour), lon + 0.00015 * np.cos(hour)],
            radius=4,
            color=colors.get(top_place, "black"),
            fill=True,
            fill_opacity=0.95,
            popup=folium.Popup(f"{hour_str}: {top_place} ({top_prob:.2%})", max_width=220),
            tooltip=hour_str,
        ).add_to(fmap)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))


def main() -> None:
    args = parse_args()
    prediction = load_prediction_json(args.date)
    rf_hourly_df, labels = build_rf_matrix(prediction)
    day_type = str(prediction.get("day_type", "Unknown"))

    out_dir = Path("outputs") / "predictions"
    heatmap_path = out_dir / f"heatmap_{args.date}.png"
    barplot_path = out_dir / f"barplot_{args.date}.png"
    map_path = out_dir / f"map_{args.date}.html"

    make_heatmap(args.date, day_type, rf_hourly_df, heatmap_path)
    make_stacked_barplot(args.date, rf_hourly_df, barplot_path)
    make_map(args.date, prediction, rf_hourly_df, labels, map_path)

    print(f"Saved: {heatmap_path}")
    print(f"Saved: {barplot_path}")
    print(f"Saved: {map_path}")


if __name__ == "__main__":
    main()
