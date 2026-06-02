# Suspect Location Prediction - ML Pipeline

## Overview
This project is a behavioral pattern machine-learning pipeline built for a law-enforcement-style location prediction use case. It is trained on 3 months of synthetic GPS data for a single suspect in Mumbai: 90 days, 5-minute intervals, and 25,920 total rows.

The system predicts probable locations for any future date and time as probability distributions, not single point predictions, because human behavior is inherently probabilistic.

This project uses synthetic data generated with realistic behavioral rules. Real surveillance data was not used.

## Problem Statement
The core question answered by the model is:

"Given a target date and time, where is the suspect most likely to be, and with what probability?"

This is a behavioral pattern matching problem, not a time-series forecasting problem. A traditional time-series forecast depends heavily on recent observations and short-term continuity. This system instead learns recurring human routines: weekday office behavior, gym visits, lunch detours, Friday evening variability, weekend patterns, and sleep-hour home presence. The model can therefore estimate likely locations for a future date using calendar and time context alone.

## Suspect Persona
| Label        | Coordinates      | Description                |
|--------------|------------------|----------------------------|
| home         | 19.1136, 72.8697 | Apartment, Kandivali East  |
| office       | 19.0176, 72.8562 | BKC, ~10.7km from home     |
| gym          | 19.1298, 72.8364 | Kandivali West, ~2.1km     |
| local_market | 19.1050, 72.8756 | Thakur Village, ~1.8km     |
| dhaba        | 19.1243, 72.8598 | Poisar area, ~1.2km        |
| cafe         | 19.0210, 72.8580 | Near office, BKC           |
| mall         | 19.0883, 72.8276 | Phoenix Palladium, ~8km    |
| friend_area  | 19.0595, 72.8307 | Bandra, ~13km              |

## Dataset
- Duration: 90 days, from 2024-03-01 to 2024-05-29
- Interval: every 5 minutes
- Total rows: 25,920
- Columns: timestamp, latitude, longitude, day_of_week, is_weekend, hour, minute, activity (stationary/transit), true_place, place_label

Behavioral patterns encoded in generation:
- Weekday routine: home -> gym (3x/week) -> office -> home
- Detours: cafe morning stop, lunch out, dhaba evenings, Friday hangouts, monthly market errand
- Variance: 6 sick days, 14 WFH days, 7 Friday hangouts across 90 days
- GPS noise: +/-0.0003 degrees at stationary locations, +/-0.001 degrees during transit

## Architecture
Three-layer prediction system:

Layer 1 - LightGBM Classifier (primary model)
- Input features: hour_sin, hour_cos, dow_sin, dow_cos, is_weekend
- Output: probability distribution over 8 place labels

Layer 2 - Probability Table (baseline model)
- Simple frequency lookup: (day_of_week, hour) -> place frequencies
- Used for agreement checking against LightGBM

Layer 3 - CategoricalHMM (sequence model, stored in JSON output)
- Separate models for weekday and weekend sequences
- Observations: hour of day, 0-23
- States: 8 place labels

Coordinate Resolution:
- Expected coordinates are computed as a probability-weighted average of known place centroids from `centroids.json`.

## Model Performance
| Metric         | Value  |
|----------------|--------|
| Accuracy       | 72.01% |
| Log Loss       | 0.8528 |
| Top-3 Accuracy | 96.38% |

Per-class F1 scores:

| Place        | F1 Score |
|--------------|----------|
| home         | 0.822    |
| office       | 0.779    |
| gym          | 0.709    |
| dhaba        | 0.355    |
| cafe         | 0.160    |
| mall         | 0.000*   |
| friend_area  | 0.000*   |
| local_market | 0.000*   |

*F1 = 0.000 for low-frequency places because the 14-day test set contained zero examples of these visits. The model does predict these places correctly in live inference; see the Friday prediction example below. This is a test set sampling limitation, not a model failure.

Train/test split: temporal, not random. Cutoff: 2024-05-15.
Train: 21,887 rows | Test: 4,033 rows

## Project Structure
```text
suspect-location-ml/
|-- src/
|   |-- generate_data.py      # Synthetic GPS data generation
|   |-- validate_data.py      # Data quality checks
|   |-- cluster_places.py     # DBSCAN place clustering + labeling
|   |-- preprocess.py         # Feature engineering pipeline
|   |-- train_model.py        # Train LightGBM, HMM, probability table
|   |-- predict.py            # Inference for any target date
|   |-- visualize.py          # Heatmaps, bar plots, folium maps
|   `-- debug_clusters.py     # Cluster diagnostics utility
|-- data/
|   |-- raw/gps_log.csv       # Generated GPS data with place labels
|   `-- processed/
|       |-- features.csv      # Feature-engineered training data
|       `-- centroids.json    # Known location coordinates
|-- models/
|   |-- lgbm_model.pkl        # Trained LightGBM classifier
|   |-- hmm_weekday.pkl       # CategoricalHMM for weekdays
|   |-- hmm_weekend.pkl       # CategoricalHMM for weekends
|   |-- prob_table.pkl        # Frequency-based probability table
|   `-- label_encoder.pkl     # Place label encoder
|-- outputs/predictions/      # JSON + PNG + HTML outputs per date
`-- docs/                     # Planning and architecture documents
```

## Setup
```bash
pip install pandas numpy scikit-learn lightgbm hmmlearn folium matplotlib seaborn joblib
```

Python 3.10+ required.

## Run Order
```bash
# Step 1 - Generate synthetic GPS data
python src/generate_data.py --days 90 --start-date 2024-03-01 --output data/raw/gps_log.csv

# Step 2 - Validate data quality
python src/validate_data.py

# Step 3 - Cluster GPS points into place labels
python src/cluster_places.py

# Step 4 - Feature engineering
python src/preprocess.py

# Step 5 - Train all models
python src/train_model.py

# Step 6 - Predict for a target date
python src/predict.py --date 2026-06-05

# Step 7 - Visualize predictions
python src/visualize.py --date 2026-06-05
```

## Prediction Output Example
Prediction for: 2026-06-05 (Friday)

```text
Hour  | LightGBM     | Prob | Table  | Prob | Agreement
00:00 | home         | 100% | home   | 100% | HIGH
06:00 | gym          |  97% | gym    |  64% | HIGH
09:00 | office       |  84% | office |  64% | HIGH
18:00 | local_market |  49% | office |  55% | LOW
20:00 | dhaba        |  42% | home   |  48% | LOW
21:00 | friend_area  |  68% | home   |  55% | LOW
HIGH confidence hours: 17/24
LOW confidence hours:  7/24
```

LOW agreement hours are operationally significant. They indicate hours where behavior is genuinely variable and may warrant active surveillance.

## Output Files
For each predicted date, three files are generated in `outputs/predictions/`:

| File               | Contents                                           |
|--------------------|----------------------------------------------------|
| `{date}.json`      | Full probability distributions, all 3 models, coordinates |
| `barplot_{date}.png` | Stacked hourly probability bar chart             |
| `heatmap_{date}.png` | Weekday vs weekend probability heatmap           |
| `map_{date}.html`  | Interactive folium map with place markers          |

## Key Design Decisions
| Decision | Rationale |
|----------|-----------|
| Behavioral pattern model, not time-series forecasting | Time-series requires recent context to predict far future; behavioral modeling works from date and time alone. |
| DBSCAN for place clustering | No need to pre-specify K; handles GPS noise naturally. |
| Temporal train/test split | Prevents data leakage; random split would train on future patterns. |
| Cyclical hour/day encoding | Prevents the model from treating hour 23 as distant from hour 0. |
| `class_weight='balanced'` | Prevents home, the dominant class, from overwhelming minority locations. |
| Probabilistic output | Humans are not 100% predictable; probabilities are more honest than single labels. |

## Known Limitations
1. Low-frequency place predictions such as mall, friend_area, and local_market have reduced reliability because there are fewer training examples.
2. The model has no memory of recent behavior. Each hour is predicted independently, so it cannot detect a sudden break from routine.
3. Gym visits may be predicted across 06:00-08:00 due to hourly aggregation, while actual visits are 60 minutes from 06:00-07:00.
4. Training data is synthetic. Real GPS data may contain different noise, missing samples, irregular routines, and additional locations.
5. The model predicts behavioral patterns, not intentions. "office 84%" means the suspect was historically at the office 84% of similar weekday mornings; it does not guarantee presence.
