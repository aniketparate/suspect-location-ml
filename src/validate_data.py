"""Quick validation checks for synthetic GPS data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_DAYS = 90
ROWS_PER_DAY = 288
EXPECTED_TOTAL_ROWS = EXPECTED_DAYS * ROWS_PER_DAY
FREQ = "5min"

HOME_LAT = 19.1136
HOME_LON = 72.8697
OFFICE_LAT = 19.0176
OFFICE_LON = 72.8562

DET0UR_LOCATIONS: dict[str, tuple[float, float]] = {
    "cafe": (19.0210, 72.8580),
    "dhaba": (19.1100, 72.8650),
    "gym": (19.1121, 72.8711),
    "local_market": (19.1150, 72.8670),
    "mall": (19.0883, 72.8276),
    "friend_area": (19.0595, 72.8307),
}

ALL_KEY_LOCATIONS: dict[str, tuple[float, float]] = {
    "home": (HOME_LAT, HOME_LON),
    "office": (OFFICE_LAT, OFFICE_LON),
    **DET0UR_LOCATIONS,
}

PROX_TIGHT_DEG = 0.001
PROX_WFH_HOME_DEG = 0.002

MUMBAI_LAT_MIN = 18.9
MUMBAI_LAT_MAX = 19.3
MUMBAI_LON_MIN = 72.7
MUMBAI_LON_MAX = 73.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated GPS CSV.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/gps_log.csv",
        help="Path to GPS CSV file.",
    )
    return parser.parse_args()


def print_result(name: str, passed: bool, detail: str) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")


def check_total_rows(df: pd.DataFrame) -> tuple[bool, str]:
    actual = len(df)
    passed = actual == EXPECTED_TOTAL_ROWS
    return passed, f"rows={actual}, expected={EXPECTED_TOTAL_ROWS}"


def check_timestamp_continuity(df: pd.DataFrame) -> tuple[bool, str]:
    if df.empty:
        return False, "dataset is empty"

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        return False, f"invalid_timestamps={int(ts.isna().sum())}"

    ts_sorted = ts.sort_values(ignore_index=True)
    start = ts_sorted.iloc[0]
    expected = pd.date_range(start=start, periods=EXPECTED_TOTAL_ROWS, freq=FREQ)

    duplicates = int(ts_sorted.duplicated().sum())
    match_expected = ts_sorted.equals(pd.Series(expected))
    diffs_ok = ts_sorted.diff().dropna().eq(pd.Timedelta(minutes=5)).all()

    passed = match_expected and diffs_ok and duplicates == 0
    detail = (
        f"start={start}, end={ts_sorted.iloc[-1]}, expected_end={expected[-1]}, "
        f"duplicates={duplicates}, exact_5min_sequence={bool(diffs_ok)}"
    )
    return passed, detail


def check_sleep_home_proximity(df: pd.DataFrame) -> tuple[bool, str]:
    sleep = df[(df["hour"] >= 0) & (df["hour"] <= 5)]
    if sleep.empty:
        return False, "sleep_rows=0"

    in_lat = (sleep["latitude"] - HOME_LAT).abs() <= 0.005
    in_lon = (sleep["longitude"] - HOME_LON).abs() <= 0.005
    in_both = in_lat & in_lon

    valid = int(in_both.sum())
    total = len(sleep)
    ratio = valid / total
    passed = valid == total
    return passed, f"within_bounds={valid}/{total} ({ratio:.2%}), expected=100.00%"


def check_weekday_office_lat_ratio(df: pd.DataFrame) -> tuple[bool, str]:
    weekday_work = df[(df["day_of_week"] < 5) & (df["hour"] >= 9) & (df["hour"] <= 17)]
    if weekday_work.empty:
        return False, "weekday_09_00_17_55_rows=0"

    in_lat = (weekday_work["latitude"] - OFFICE_LAT).abs() <= 0.01
    match = int(in_lat.sum())
    total = len(weekday_work)
    ratio = match / total
    passed = ratio >= 0.70
    return passed, f"office_lat_match={match}/{total} ({ratio:.2%}), threshold=70.00%"


def near_location(df: pd.DataFrame, lat: float, lon: float, tol: float) -> pd.Series:
    return (df["latitude"] - lat).abs().le(tol) & (df["longitude"] - lon).abs().le(tol)


def run_check_a(df: pd.DataFrame) -> None:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).copy()
    work["date"] = work["timestamp"].dt.normalize()

    home_near_005 = near_location(work, HOME_LAT, HOME_LON, tol=0.005)
    home_near_002 = near_location(work, HOME_LAT, HOME_LON, tol=PROX_WFH_HOME_DEG)

    sick_days = 0
    wfh_days = 0
    normal_weekdays = 0
    weekend_days = 0

    for _, day_df in work.groupby("date"):
        day_of_week = int(day_df["day_of_week"].iloc[0])
        is_weekend = day_of_week >= 5
        if is_weekend:
            weekend_days += 1
            continue

        home_ratio_005 = float(home_near_005.loc[day_df.index].mean())
        home_ratio_002 = float(home_near_002.loc[day_df.index].mean())

        if home_ratio_005 >= 0.90:
            sick_days += 1
        elif home_ratio_002 >= 0.85:
            wfh_days += 1
        else:
            normal_weekdays += 1

    print("\nCHECK A - Day type counts across 90 days")
    print(f"  Sick days (weekday, 90%+ home): {sick_days}")
    print(f"  WFH days (weekday, 85%+ home within 0.002): {wfh_days}")
    print(f"  Normal weekdays: {normal_weekdays}")
    print(f"  Weekend days total (active/lazy combined): {weekend_days}")


def run_check_b(df: pd.DataFrame) -> None:
    total = len(df)
    print("\nCHECK B - Detour location hit counts (within 0.001 degrees)")
    for label, (lat, lon) in DET0UR_LOCATIONS.items():
        hits = int(near_location(df, lat, lon, tol=PROX_TIGHT_DEG).sum())
        pct = hits / total if total else 0.0
        print(f"  {label:<12} rows={hits:>5}  pct={pct:.2%}")

    cafe_hits = int(near_location(df, *DET0UR_LOCATIONS["cafe"], tol=PROX_TIGHT_DEG).sum())
    dhaba_hits = int(near_location(df, *DET0UR_LOCATIONS["dhaba"], tol=PROX_TIGHT_DEG).sum())
    if cafe_hits < 200:
        print("  WARNING: cafe hit count is below 200 rows.")
    if dhaba_hits < 100:
        print("  WARNING: dhaba hit count is below 100 rows.")


def run_check_c(df: pd.DataFrame) -> None:
    friday_evening = df[(df["day_of_week"] == 4) & (df["hour"] >= 19) & (df["hour"] <= 22)]
    total = len(friday_evening)
    if total == 0:
        print_result("CHECK C - Friday evening pattern", False, "no Friday 19:00-22:55 rows found")
        return

    away = int(((friday_evening["latitude"] - HOME_LAT).abs() > 0.01).sum())
    away_pct = away / total
    passed = away_pct > 0.40
    detail = f"away_from_home={away}/{total} ({away_pct:.2%}), expected > 40.00%"
    print_result("CHECK C - Friday evening pattern", passed, detail)


def run_check_d(df: pd.DataFrame) -> None:
    lat_min, lat_max, lat_mean = float(df["latitude"].min()), float(df["latitude"].max()), float(df["latitude"].mean())
    lon_min, lon_max, lon_mean = float(df["longitude"].min()), float(df["longitude"].max()), float(df["longitude"].mean())

    print("\nCHECK D - Coordinate spread sanity")
    print(f"  latitude:  min={lat_min:.6f}, max={lat_max:.6f}, mean={lat_mean:.6f}")
    print(f"  longitude: min={lon_min:.6f}, max={lon_max:.6f}, mean={lon_mean:.6f}")

    in_bounds = (
        (df["latitude"].between(MUMBAI_LAT_MIN, MUMBAI_LAT_MAX)).all()
        and (df["longitude"].between(MUMBAI_LON_MIN, MUMBAI_LON_MAX)).all()
    )
    print_result(
        "CHECK D bounds (Mumbai)",
        bool(in_bounds),
        f"lat in [{MUMBAI_LAT_MIN}, {MUMBAI_LAT_MAX}], lon in [{MUMBAI_LON_MIN}, {MUMBAI_LON_MAX}]",
    )


def run_check_e(df: pd.DataFrame) -> None:
    print("\nCHECK E - 0.001-degree hits with hour distribution (8 locations)")
    for label, (lat, lon) in ALL_KEY_LOCATIONS.items():
        mask = near_location(df, lat, lon, tol=PROX_TIGHT_DEG)
        hit_df = df[mask]
        total_hits = int(len(hit_df))
        if total_hits == 0:
            print(f"  {label:<12}: 0 rows - peak hours: none")
            continue

        hour_counts = hit_df["hour"].value_counts().sort_values(ascending=False)
        top = hour_counts.head(2)
        top_parts = [f"{int(h):02d}h({int(c)})" for h, c in top.items()]
        other = int(total_hits - int(top.sum()))
        if other > 0:
            top_parts.append(f"other({other})")
        summary = ", ".join(top_parts)
        print(f"  {label:<12}: {total_hits} rows - peak hours: {summary}")


def run_check_f(df: pd.DataFrame) -> None:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).copy()
    work["date"] = work["timestamp"].dt.normalize()

    wfh_days = 0
    home_near_002 = near_location(work, HOME_LAT, HOME_LON, tol=PROX_WFH_HOME_DEG)
    for _, day_df in work.groupby("date"):
        if int(day_df["day_of_week"].iloc[0]) > 4:
            continue
        home_ratio_002 = float(home_near_002.loc[day_df.index].mean())
        if home_ratio_002 >= 0.85:
            wfh_days += 1

    print("\nCHECK F - WFH day count (weekday, 85%+ rows within 0.002 of home)")
    print(f"  WFH days: {wfh_days}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print_result("1) Total rows", *check_total_rows(df))
    print_result("2) No timestamp gaps (5 min, full 90 days)", *check_timestamp_continuity(df))
    print_result("3) Sleep-hour near home", *check_sleep_home_proximity(df))
    print_result("4) Weekday 09:00-17:55 office-lat ratio >= 70%", *check_weekday_office_lat_ratio(df))
    print_result("5) Output includes actual values", True, "Each check prints observed counts/ratios.")
    run_check_a(df)
    run_check_b(df)
    run_check_c(df)
    run_check_d(df)
    run_check_e(df)
    run_check_f(df)


if __name__ == "__main__":
    main()
