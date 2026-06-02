"""Generate synthetic GPS data with explicit stationary/transit activity labels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FREQ_MINUTES = 5
ROWS_PER_DAY = 24 * 60 // FREQ_MINUTES
STATIONARY_NOISE_STD = 0.0003
TRANSIT_NOISE_STD = 0.001
DEFAULT_SEED = 42

LOCATIONS: dict[str, tuple[float, float]] = {
    "home": (19.1136, 72.8697),
    "office": (19.0176, 72.8562),
    "gym": (19.1298, 72.8364),
    "local_market": (19.1050, 72.8756),
    "dhaba": (19.1243, 72.8598),
    "cafe": (19.0210, 72.8580),
    "mall": (19.0883, 72.8276),
    "friend_area": (19.0595, 72.8307),
}

# Route durations in 5-minute slots.
TRAVEL_SLOTS: dict[tuple[str, str], int] = {
    ("home", "office"): 18,
    ("home", "gym"): 4,
    ("home", "dhaba"): 3,
    ("home", "local_market"): 4,
    ("home", "mall"): 12,
    ("home", "friend_area"): 15,
    ("office", "cafe"): 1,
    ("office", "home"): 18,
}

DET0UR_KEYS = [
    "friday_hangout",
    "late_office",
    "cafe_morning",
    "lunch_out",
    "dhaba_evening",
    "monthly_errand",
]


@dataclass(frozen=True)
class DayAssignment:
    day_type: str  # sick | WFH | normal_weekday | active_weekend | lazy_weekend
    is_weekend: bool
    gym_morning: bool = False
    friday_hangout: bool = False
    friday_hangout_spot: str | None = None
    late_office: bool = False
    cafe_morning: bool = False
    lunch_out: bool = False
    dhaba_evening: bool = False
    monthly_errand: bool = False
    wfh_market_detour: bool = False
    wfh_dhaba_detour: bool = False


@dataclass(frozen=True)
class CoordPoint:
    lat: float
    lon: float
    place_label: str
    activity: str  # stationary | transit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic GPS dataset.")
    parser.add_argument("--days", type=int, default=90, help="Number of days to generate.")
    parser.add_argument("--start-date", type=str, default="2024-03-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--output", type=str, default="data/raw/gps_log.csv", help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    return parser.parse_args()


def minute_of_day(ts: pd.Timestamp) -> int:
    return ts.hour * 60 + ts.minute


def between(m: int, start: int, end: int) -> bool:
    return start <= m < end


def stationary_point(label: str, rng: np.random.Generator) -> CoordPoint:
    lat, lon = LOCATIONS[label]
    lat = lat + float(rng.normal(0, STATIONARY_NOISE_STD))
    lon = lon + float(rng.normal(0, STATIONARY_NOISE_STD))
    return CoordPoint(lat=lat, lon=lon, place_label=label, activity="stationary")


def get_travel_slots(start_loc: str, end_loc: str) -> int:
    if (start_loc, end_loc) in TRAVEL_SLOTS:
        return TRAVEL_SLOTS[(start_loc, end_loc)]
    if (end_loc, start_loc) in TRAVEL_SLOTS:
        return TRAVEL_SLOTS[(end_loc, start_loc)]
    if start_loc != "home" and end_loc != "home":
        return get_travel_slots(start_loc, "home") + get_travel_slots("home", end_loc)
    return 12


def transit_point(
    start_loc: str,
    end_loc: str,
    minute: int,
    depart_minute: int,
    rng: np.random.Generator,
) -> CoordPoint:
    total_slots = max(1, get_travel_slots(start_loc, end_loc))
    elapsed_slots = max(1, (minute - depart_minute) // FREQ_MINUTES + 1)
    slot = int(np.clip(elapsed_slots, 1, total_slots))

    s_lat, s_lon = LOCATIONS[start_loc]
    e_lat, e_lon = LOCATIONS[end_loc]
    t = slot / total_slots
    lat = s_lat + (e_lat - s_lat) * t
    lon = s_lon + (e_lon - s_lon) * t
    lat += float(rng.normal(0, TRANSIT_NOISE_STD))
    lon += float(rng.normal(0, TRANSIT_NOISE_STD))
    return CoordPoint(lat=lat, lon=lon, place_label="transit", activity="transit")


def choose_sick_days(days: list[pd.Timestamp], rng: np.random.Generator) -> set[pd.Timestamp]:
    month_groups: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for day in days:
        month_groups.setdefault((day.year, day.month), []).append(day)

    sick_days: set[pd.Timestamp] = set()
    for month_days in month_groups.values():
        count = min(2, len(month_days))
        if count == 0:
            continue
        picks = rng.choice(month_days, size=count, replace=False).tolist()
        sick_days.update(pd.Timestamp(d).normalize() for d in picks)
    return sick_days


def choose_wfh_days(days: list[pd.Timestamp], sick_days: set[pd.Timestamp], rng: np.random.Generator) -> set[pd.Timestamp]:
    week_groups: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for day in days:
        iso = day.isocalendar()
        week_groups.setdefault((int(iso.year), int(iso.week)), []).append(day)

    wfh_days: set[pd.Timestamp] = set()
    for week_days in week_groups.values():
        candidates = [d for d in week_days if d.dayofweek < 5 and d not in sick_days]
        if not candidates:
            continue
        wfh_days.add(pd.Timestamp(rng.choice(candidates)).normalize())
    return wfh_days


def is_second_wednesday(day: pd.Timestamp) -> bool:
    return day.dayofweek == 2 and 8 <= day.day <= 14


def assign_day(
    day: pd.Timestamp,
    sick_days: set[pd.Timestamp],
    wfh_days: set[pd.Timestamp],
    gym_days: set[pd.Timestamp],
    rng: np.random.Generator,
) -> DayAssignment:
    is_weekend = day.dayofweek >= 5

    if day in sick_days:
        return DayAssignment(day_type="sick", is_weekend=is_weekend)
    if day in wfh_days:
        market = bool(rng.random() < 0.35)
        dhaba = bool(rng.random() < 0.20) and not market
        return DayAssignment(
            day_type="WFH",
            is_weekend=is_weekend,
            wfh_market_detour=market,
            wfh_dhaba_detour=dhaba,
        )
    if is_weekend:
        active = bool(rng.random() < 0.60)
        return DayAssignment(
            day_type="active_weekend" if active else "lazy_weekend",
            is_weekend=True,
            gym_morning=False,
            friday_hangout_spot=("mall" if rng.random() < 0.50 else "friend_area") if active else None,
        )

    gym_morning = day in gym_days
    friday_hangout = day.dayofweek == 4 and bool(rng.random() < 0.60)
    hangout_spot = ("friend_area" if rng.random() < 0.50 else "mall") if friday_hangout else None
    late_office = (day.dayofweek in {0, 1}) and bool(rng.random() < 0.20)
    if friday_hangout:
        late_office = False
    cafe_morning = (day.dayofweek in {0, 2, 4}) and bool(rng.random() < 0.70)
    lunch_out = (day.dayofweek in {1, 3}) and bool(rng.random() < 0.75)
    dhaba_evening = (day.dayofweek in {2, 3}) and bool(rng.random() < 0.65)
    if friday_hangout or late_office:
        dhaba_evening = False
    monthly_errand = is_second_wednesday(day)

    return DayAssignment(
        day_type="normal_weekday",
        is_weekend=False,
        gym_morning=gym_morning,
        friday_hangout=friday_hangout,
        friday_hangout_spot=hangout_spot,
        late_office=late_office,
        cafe_morning=cafe_morning,
        lunch_out=lunch_out,
        dhaba_evening=dhaba_evening,
        monthly_errand=monthly_errand,
    )


def get_weekday_normal_point(m: int, plan: DayAssignment, rng: np.random.Generator) -> CoordPoint:
    if between(m, 0, 360):
        return stationary_point("home", rng)
    if between(m, 360, 420):
        return stationary_point("gym" if plan.gym_morning else "home", rng)

    morning_start = "gym" if plan.gym_morning else "home"
    if between(m, 420, 510):
        return transit_point(morning_start, "office", m, 420, rng)
    if plan.cafe_morning and between(m, 510, 525):
        return stationary_point("cafe", rng)
    if plan.cafe_morning and between(m, 525, 530):
        return transit_point("cafe", "office", m, 525, rng)

    if between(m, 540, 780):
        return stationary_point("office", rng)
    if between(m, 780, 840):
        return stationary_point("cafe", rng) if plan.lunch_out else stationary_point("office", rng)
    if between(m, 840, 1080):
        return stationary_point("office", rng)

    if plan.friday_hangout:
        spot = plan.friday_hangout_spot or "friend_area"
        if between(m, 1080, 1080 + get_travel_slots("office", spot) * 5):
            return transit_point("office", spot, m, 1080, rng)
        if between(m, 1140, 1320):
            return stationary_point(spot, rng)
        if between(m, 1320, 1320 + get_travel_slots(spot, "home") * 5):
            return transit_point(spot, "home", m, 1320, rng)
        return stationary_point("home", rng)

    if plan.late_office:
        if between(m, 1080, 1230):
            return stationary_point("office", rng)
        if between(m, 1230, 1230 + get_travel_slots("office", "home") * 5):
            return transit_point("office", "home", m, 1230, rng)
        return stationary_point("home", rng)

    if plan.monthly_errand:
        if between(m, 1080, 1080 + get_travel_slots("office", "local_market") * 5):
            return transit_point("office", "local_market", m, 1080, rng)
        if between(m, 1110, 1170):
            return stationary_point("local_market", rng)
        if between(m, 1170, 1170 + get_travel_slots("local_market", "home") * 5):
            return transit_point("local_market", "home", m, 1170, rng)
        return stationary_point("home", rng)

    if plan.dhaba_evening:
        if between(m, 1080, 1080 + get_travel_slots("office", "home") * 5):
            return transit_point("office", "home", m, 1080, rng)
        if between(m, 1170, 1230):
            return stationary_point("dhaba", rng)
        if between(m, 1230, 1230 + get_travel_slots("dhaba", "home") * 5):
            return transit_point("dhaba", "home", m, 1230, rng)
        return stationary_point("home", rng)

    if between(m, 1080, 1080 + get_travel_slots("office", "home") * 5):
        return transit_point("office", "home", m, 1080, rng)
    return stationary_point("home", rng)


def get_wfh_point(m: int, plan: DayAssignment, rng: np.random.Generator) -> CoordPoint:
    if plan.wfh_market_detour:
        if between(m, 1090, 1090 + get_travel_slots("home", "local_market") * 5):
            return transit_point("home", "local_market", m, 1090, rng)
        if between(m, 1110, 1170):
            return stationary_point("local_market", rng)
        if between(m, 1170, 1170 + get_travel_slots("local_market", "home") * 5):
            return transit_point("local_market", "home", m, 1170, rng)

    if plan.wfh_dhaba_detour:
        if between(m, 1155, 1155 + get_travel_slots("home", "dhaba") * 5):
            return transit_point("home", "dhaba", m, 1155, rng)
        if between(m, 1170, 1230):
            return stationary_point("dhaba", rng)
        if between(m, 1230, 1230 + get_travel_slots("dhaba", "home") * 5):
            return transit_point("dhaba", "home", m, 1230, rng)
    return stationary_point("home", rng)


def get_weekend_point(m: int, plan: DayAssignment, rng: np.random.Generator) -> CoordPoint:
    if plan.day_type == "lazy_weekend":
        if between(m, 930, 990) and bool(rng.random() < 0.35):
            return stationary_point("local_market", rng)
        return stationary_point("home", rng)

    spot = plan.friday_hangout_spot or "mall"
    if between(m, 0, 420):
        return stationary_point("home", rng)
    if between(m, 420, 480):
        return stationary_point("gym" if plan.gym_morning else "home", rng)
    if between(m, 480, 480 + get_travel_slots("gym", "home") * 5) and plan.gym_morning:
        return transit_point("gym", "home", m, 480, rng)
    if between(m, 780, 780 + get_travel_slots("home", spot) * 5):
        return transit_point("home", spot, m, 780, rng)
    if between(m, 840, 1020):
        return stationary_point(spot, rng)
    if between(m, 1020, 1020 + get_travel_slots(spot, "home") * 5):
        return transit_point(spot, "home", m, 1020, rng)
    if between(m, 1110, 1170) and bool(rng.random() < 0.40):
        return stationary_point("local_market", rng)
    return stationary_point("home", rng)


def coordinate_for_timestamp(ts: pd.Timestamp, plan: DayAssignment, rng: np.random.Generator) -> CoordPoint:
    m = minute_of_day(ts)
    if plan.day_type == "sick":
        return stationary_point("home", rng)
    if plan.day_type == "WFH":
        return get_wfh_point(m, plan, rng)
    if plan.day_type in {"active_weekend", "lazy_weekend"}:
        return get_weekend_point(m, plan, rng)
    return get_weekday_normal_point(m, plan, rng)


def choose_gym_days(
    days: list[pd.Timestamp],
    sick_days: set[pd.Timestamp],
    wfh_days: set[pd.Timestamp],
    rng: np.random.Generator,
) -> set[pd.Timestamp]:
    week_groups: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for day in days:
        iso = day.isocalendar()
        week_groups.setdefault((int(iso.year), int(iso.week)), []).append(day)

    gym_days: set[pd.Timestamp] = set()
    for week_days in week_groups.values():
        candidates = [
            d
            for d in week_days
            if d.dayofweek < 5 and d.dayofweek != 0 and d not in sick_days and d not in wfh_days
        ]
        if not candidates:
            continue
        pick_n = min(3, len(candidates))
        picks = rng.choice(candidates, size=pick_n, replace=False).tolist()
        gym_days.update(pd.Timestamp(d).normalize() for d in picks)
    return gym_days


def build_assignments(days: list[pd.Timestamp], rng: np.random.Generator) -> dict[pd.Timestamp, DayAssignment]:
    sick_days = choose_sick_days(days, rng)
    wfh_days = choose_wfh_days(days, sick_days, rng)
    gym_days = choose_gym_days(days, sick_days, wfh_days, rng)
    return {day: assign_day(day, sick_days, wfh_days, gym_days, rng) for day in days}


def print_first_detour_debug(assignments: dict[pd.Timestamp, DayAssignment]) -> None:
    first_dhaba_day = next(
        (
            day
            for day, plan in sorted(assignments.items())
            if plan.day_type == "normal_weekday" and plan.dhaba_evening
        ),
        None,
    )
    if first_dhaba_day is not None:
        print(
            "Dhaba day: "
            f"{first_dhaba_day.date()}, slots 234–245 at {LOCATIONS['dhaba']}"
        )
    else:
        print("Dhaba day: not found")

    first_gym_day = next(
        (
            day
            for day, plan in sorted(assignments.items())
            if plan.gym_morning and day.dayofweek < 5
        ),
        None,
    )
    if first_gym_day is not None:
        print(
            "Gym day: "
            f"{first_gym_day.date()}, slots 72–83 at {LOCATIONS['gym']}"
        )
    else:
        print("Gym day: not found")


def generate_data(days: int, start_date: str, seed: int) -> tuple[pd.DataFrame, dict[pd.Timestamp, DayAssignment]]:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date).normalize()
    day_index = [start + pd.Timedelta(days=i) for i in range(days)]
    assignments = build_assignments(day_index, rng)

    timestamps = pd.date_range(start=start, periods=days * ROWS_PER_DAY, freq=f"{FREQ_MINUTES}min")
    records: list[dict[str, object]] = []
    for ts in timestamps:
        plan = assignments[ts.normalize()]
        pt = coordinate_for_timestamp(ts, plan, rng)
        records.append(
            {
                "timestamp": ts,
                "latitude": round(float(pt.lat), 6),
                "longitude": round(float(pt.lon), 6),
                "day_of_week": int(ts.dayofweek),
                "is_weekend": bool(ts.dayofweek >= 5),
                "hour": int(ts.hour),
                "minute": int(ts.minute),
                "place_label": str(pt.place_label),
                "activity": str(pt.activity),
            }
        )
    return pd.DataFrame(records), assignments


def validate_output(df: pd.DataFrame, expected_days: int) -> None:
    expected_rows = expected_days * ROWS_PER_DAY
    if len(df) != expected_rows:
        raise ValueError(f"Unexpected row count: {len(df)} (expected {expected_rows}).")

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        raise ValueError("Found invalid timestamps.")

    expected_ts = pd.date_range(start=ts.iloc[0], periods=expected_rows, freq=f"{FREQ_MINUTES}min")
    if not ts.reset_index(drop=True).equals(pd.Series(expected_ts)):
        raise ValueError("Timestamp sequence has gaps.")

    if df[["latitude", "longitude", "activity"]].isna().any().any():
        raise ValueError("Found missing coordinate/activity values.")


def print_summary(assignments: dict[pd.Timestamp, DayAssignment]) -> None:
    day_type_counts = {"sick": 0, "WFH": 0, "normal_weekday": 0, "active_weekend": 0, "lazy_weekend": 0}
    detour_counts = {k: 0 for k in DET0UR_KEYS}

    for plan in assignments.values():
        day_type_counts[plan.day_type] += 1
        if plan.day_type != "normal_weekday":
            continue
        if plan.friday_hangout:
            detour_counts["friday_hangout"] += 1
        if plan.late_office:
            detour_counts["late_office"] += 1
        if plan.cafe_morning:
            detour_counts["cafe_morning"] += 1
        if plan.lunch_out:
            detour_counts["lunch_out"] += 1
        if plan.dhaba_evening:
            detour_counts["dhaba_evening"] += 1
        if plan.monthly_errand:
            detour_counts["monthly_errand"] += 1

    print("\nDay-type summary:")
    print(f"  sick: {day_type_counts['sick']}")
    print(f"  WFH: {day_type_counts['WFH']}")
    print(f"  normal weekday: {day_type_counts['normal_weekday']}")
    print(f"  active weekend: {day_type_counts['active_weekend']}")
    print(f"  lazy weekend: {day_type_counts['lazy_weekend']}")

    print("\nDetour counts (days active):")
    for key in DET0UR_KEYS:
        print(f"  {key}: {detour_counts[key]}")


def main() -> None:
    args = parse_args()
    if args.days <= 0:
        raise ValueError("--days must be positive.")

    df, assignments = generate_data(days=args.days, start_date=args.start_date, seed=args.seed)
    validate_output(df, expected_days=args.days)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print_first_detour_debug(assignments)
    print_summary(assignments)


if __name__ == "__main__":
    main()
