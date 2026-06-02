# Suspect Location Prediction — ML Project

A behavioral pattern ML system that predicts probable locations of a suspect
based on 3 months of historical GPS data.

---

## Setup

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn folium joblib
```

---

## Run Order

```bash
# Step 1: Generate synthetic GPS data
python src/generate_data.py --days 90 --start-date 2024-03-01 --output data/raw/gps_log.csv

# Step 2: Cluster GPS points into places
python src/cluster_places.py --input data/raw/gps_log.csv

# Step 3: Feature engineering
python src/preprocess.py

# Step 4: Train models
python src/train_model.py

# Step 5: Predict for a specific date
python src/predict.py --date 2024-06-15

# Step 6: Visualize
python src/visualize.py --date 2024-06-15
```

---

## Example Output

```
Prediction for: Saturday, 2024-06-15
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hour  │ Top Location    │ RF Prob │ Table Prob
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
00:00 │ home            │  97%    │  96%
06:00 │ gym             │  61%    │  58%
10:00 │ home            │  73%    │  71%
14:00 │ mall            │  54%    │  51%
20:00 │ home            │  84%    │  82%
23:00 │ home            │  98%    │  97%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| DBSCAN for place clustering | No need to pre-specify K; handles transit noise naturally |
| Temporal train/test split | Prevents data leakage from future patterns |
| Two models (table + RF) | Baseline sanity check + ML generalization |
| Cyclical hour/day encoding | Prevents model treating hour 23 as far from hour 0 |
| Probabilistic output | Correct framing; humans aren't 100% predictable |

---

## Docs Index

| File | Contents |
|------|----------|
| `docs/00_PROJECT_PLAN.md` | Architecture, phases, stack |
| `docs/01_DATA_SCHEMA.md` | CSV format, suspect persona, locations |
| `docs/02_DATA_GENERATION.md` | Behavioral rules, noise strategy |
| `docs/03_FEATURE_ENGINEERING.md` | DBSCAN, feature construction |
| `docs/04_MODEL_DESIGN.md` | Model options, inference pipeline, evaluation |
| `docs/05_CODEX_PROMPTS.md` | Copy-paste prompts for each script |
