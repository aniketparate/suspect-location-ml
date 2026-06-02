"""Train baseline table, LightGBM, and day-type HMM location models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import LabelEncoder


FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
TARGET = "place_label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train probabilistic location models.")
    parser.add_argument("--input", type=str, default="data/processed/features.csv", help="Input feature CSV path.")
    parser.add_argument("--prob-table-out", type=str, default="models/prob_table.pkl", help="Baseline table output.")
    parser.add_argument("--lgbm-out", type=str, default="models/lgbm_model.pkl", help="LightGBM model output.")
    parser.add_argument("--hmm-weekday-out", type=str, default="models/hmm_weekday.pkl", help="Weekday HMM output.")
    parser.add_argument("--hmm-weekend-out", type=str, default="models/hmm_weekend.pkl", help="Weekend HMM output.")
    parser.add_argument("--test-days", type=int, default=14, help="Last N days as temporal test split.")
    return parser.parse_args()


def make_temporal_split(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = ordered["timestamp"].max() - pd.Timedelta(days=test_days)
    train_df = ordered[ordered["timestamp"] <= cutoff].copy()
    test_df = ordered[ordered["timestamp"] > cutoff].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(f"Temporal split failed: train={len(train_df)}, test={len(test_df)}, cutoff={cutoff}")
    return train_df, test_df, cutoff


def build_probability_table(train_df: pd.DataFrame) -> dict[int, dict[int, dict[str, float]]]:
    grouped = (
        train_df.groupby(["day_of_week", "hour", TARGET], dropna=False).size().reset_index(name="count")
    )
    grouped["probability"] = grouped["count"] / grouped.groupby(["day_of_week", "hour"])["count"].transform("sum")
    out: dict[int, dict[int, dict[str, float]]] = {}
    for row in grouped.itertuples(index=False):
        d = int(row.day_of_week)
        h = int(row.hour)
        label = str(getattr(row, TARGET))
        out.setdefault(d, {}).setdefault(h, {})[label] = float(row.probability)
    return out


def top_k_accuracy_from_proba(y_true: pd.Series, proba: np.ndarray, classes: np.ndarray, k: int = 3) -> float:
    k_eff = min(k, proba.shape[1])
    top_idx = np.argsort(proba, axis=1)[:, -k_eff:]
    cls = np.asarray(classes)
    top_labels = cls[top_idx]
    matches = [t in row for t, row in zip(y_true.to_numpy(), top_labels, strict=False)]
    return float(np.mean(matches))


def print_per_class_f1(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> None:
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    print("\nPer-class F1 score:")
    for label, score in zip(labels, f1, strict=False):
        print(f"  {label}: {float(score):.6f}")


def build_observations(df: pd.DataFrame) -> np.ndarray:
    hour_bin = (df["hour"].astype(int) // 2).to_numpy()
    weekend = df["is_weekend"].astype(int).to_numpy()
    obs = hour_bin * 2 + weekend
    return obs.astype(int)


def train_hmm_bundle(day_df: pd.DataFrame, label_encoder: LabelEncoder) -> dict[str, object]:
    from hmmlearn import hmm

    if day_df.empty:
        raise ValueError("Cannot train HMM on empty dataset.")

    obs = build_observations(day_df)
    X = obs.reshape(-1, 1)
    lengths = [len(X)]
    n_states = len(label_encoder.classes_)

    model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, random_state=42)
    try:
        model.fit(X, lengths=lengths)
    except Exception:
        # Compatibility fallback for hmmlearn versions expecting one-hot multinomial input.
        n_obs = int(obs.max()) + 1
        X_onehot = np.zeros((len(obs), n_obs), dtype=int)
        X_onehot[np.arange(len(obs)), obs] = 1
        model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, random_state=42)
        model.fit(X_onehot, lengths=lengths)
        state_seq = model.predict(X_onehot, lengths=lengths)
    else:
        state_seq = model.predict(X, lengths=lengths)

    y_true = label_encoder.transform(day_df[TARGET].astype(str))
    state_to_label: dict[int, str] = {}
    fallback = str(day_df[TARGET].mode().iloc[0])
    for state in range(n_states):
        idx = np.where(state_seq == state)[0]
        if len(idx) == 0:
            state_to_label[state] = fallback
            continue
        counts = np.bincount(y_true[idx], minlength=n_states)
        state_to_label[state] = str(label_encoder.classes_[int(np.argmax(counts))])

    return {
        "model": model,
        "state_to_label": state_to_label,
        "classes": [str(c) for c in label_encoder.classes_],
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required = {"timestamp", "hour", "day_of_week", "is_weekend", TARGET, *FEATURES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", TARGET]).copy()

    train_df, test_df, cutoff = make_temporal_split(df, args.test_days)
    print(f"Temporal split cutoff: {cutoff}")
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    prob_table = build_probability_table(train_df)
    Path(args.prob_table_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(prob_table, args.prob_table_out)
    print(f"Saved probability table to: {args.prob_table_out}")

    import lightgbm as lgb

    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()
    X_train["is_weekend"] = X_train["is_weekend"].astype(int)
    X_test["is_weekend"] = X_test["is_weekend"].astype(int)
    y_train = train_df[TARGET].astype(str)
    y_test = test_df[TARGET].astype(str)

    lgbm = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    y_proba = lgbm.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba, labels=lgbm.classes_) if len(lgbm.classes_) > 1 else 0.0
    top3 = top_k_accuracy_from_proba(y_test, y_proba, lgbm.classes_, k=3)

    print("\nLightGBM evaluation:")
    print(f"Accuracy: {acc:.6f}")
    print(f"Log loss: {ll:.6f}")
    print(f"Top-3 accuracy: {top3:.6f}")
    print_per_class_f1(y_test, y_pred, labels=list(lgbm.classes_))

    Path(args.lgbm_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgbm, args.lgbm_out)
    print(f"\nSaved LightGBM model to: {args.lgbm_out}")

    label_encoder = LabelEncoder()
    label_encoder.fit(df[TARGET].astype(str))
    weekday_train = train_df[train_df["is_weekend"] == False].copy()
    weekend_train = train_df[train_df["is_weekend"] == True].copy()
    hmm_weekday = train_hmm_bundle(weekday_train, label_encoder)
    hmm_weekend = train_hmm_bundle(weekend_train, label_encoder)

    Path(args.hmm_weekday_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(hmm_weekday, args.hmm_weekday_out)
    joblib.dump(hmm_weekend, args.hmm_weekend_out)
    print(f"Saved weekday HMM to: {args.hmm_weekday_out}")
    print(f"Saved weekend HMM to: {args.hmm_weekend_out}")


if __name__ == "__main__":
    main()
