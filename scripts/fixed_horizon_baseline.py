
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "cyclones_clean.csv"
OUTPUT = ROOT / "data" / "processed" / "fixed_horizon_features.csv"
PREDICTIONS = ROOT / "data" / "processed" / "fixed_horizon_predictions.csv"

EARTH_RADIUS_KM = 6371.0
HORIZONS = [6, 12, 24]
TOLERANCE_HOURS = 1.5


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_KM * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


def bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlon = lon2 - lon1

    y = np.sin(dlon) * np.cos(lat2)

    x = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def destination_point(lat, lon, distance_km, bearing_deg):
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    theta = np.radians(bearing_deg)

    angular_distance = distance_km / EARTH_RADIUS_KM

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular_distance)
        + np.cos(lat1) * np.sin(angular_distance) * np.cos(theta)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(theta) * np.sin(angular_distance) * np.cos(lat1),
        np.cos(angular_distance) - np.sin(lat1) * np.sin(lat2)
    )

    return np.degrees(lat2), (np.degrees(lon2) + 540) % 360 - 180


def create_history_features(df):
    """Calculate movement features using only past and current data."""

    df = df.sort_values(["SID", "ISO_TIME"]).copy()

    grouped = df.groupby("SID", sort=False)

    df["PREV_LAT"] = grouped["LAT"].shift(1)
    df["PREV_LON"] = grouped["LON"].shift(1)
    df["PREV_TIME"] = grouped["ISO_TIME"].shift(1)

    df["DELTA_HOURS"] = (
        df["ISO_TIME"] - df["PREV_TIME"]
    ).dt.total_seconds() / 3600

    df["DISTANCE_KM"] = haversine(
        df["PREV_LAT"],
        df["PREV_LON"],
        df["LAT"],
        df["LON"]
    )

    df["SPEED_KMH"] = (
        df["DISTANCE_KM"] / df["DELTA_HOURS"]
    )

    df["HEADING_DEG"] = bearing(
        df["PREV_LAT"],
        df["PREV_LON"],
        df["LAT"],
        df["LON"]
    )

    radians = np.radians(df["HEADING_DEG"])

    df["HEADING_SIN"] = np.sin(radians)
    df["HEADING_COS"] = np.cos(radians)

    # Require a valid recent movement estimate.
    valid = (
        df["PREV_LAT"].notna()
        & df["DELTA_HOURS"].gt(0)
        & df["DELTA_HOURS"].le(12)
        & df["SPEED_KMH"].between(0, 150)
        & np.isfinite(df["HEADING_DEG"])
    )

    return df[valid].copy()



def match_future_position(history, observations, horizon):
    """Match input history rows to the closest original future observation."""

    history = history.sort_values("ISO_TIME").copy()
    observations = observations.sort_values("ISO_TIME").reset_index(drop=True)

    times = observations["ISO_TIME"].astype("int64").to_numpy()
    origin_times = history["ISO_TIME"].astype("int64").to_numpy()

    target_ns = origin_times + int(horizon * 3600 * 1e9)
    insertion = np.searchsorted(times, target_ns, side="left")

    future_idx = np.clip(insertion, 0, len(times) - 1)
    previous_idx = np.clip(insertion - 1, 0, len(times) - 1)

    future_error = np.abs(times[future_idx] - target_ns)
    previous_error = np.abs(times[previous_idx] - target_ns)

    chosen = np.where(
        previous_error < future_error,
        previous_idx,
        future_idx
    )

    actual_ns = times[chosen]

    actual_horizon = (actual_ns - origin_times) / 3.6e12
    tolerance_error = np.abs(actual_horizon - horizon)

    valid = (
        (actual_horizon > 0)
        & (tolerance_error <= TOLERANCE_HOURS)
    )

    targets = observations.iloc[chosen]

    result = history.copy()

    result["HORIZON_HOURS"] = horizon
    result["TARGET_LAT"] = targets["LAT"].to_numpy()
    result["TARGET_LON"] = targets["LON"].to_numpy()
    result["TARGET_TIME"] = targets["ISO_TIME"].to_numpy()
    result["FUTURE_HOURS"] = actual_horizon

    return result.loc[valid].copy()


def build_fixed_horizon_dataset(df):
    """Build independent samples for each forecast horizon."""

    all_samples = []

    for sid, group in df.groupby("SID", sort=False):
        history = create_history_features(group)

        if history.empty:
            continue

        for horizon in HORIZONS:
            matched = match_future_position(history,group, horizon)

            if not matched.empty:
                all_samples.append(matched)

    if not all_samples:
        raise ValueError("No valid future targets found.")

    return pd.concat(all_samples, ignore_index=True)


def evaluate(df):
    """Evaluate constant-velocity forecasts on unseen cyclones."""

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(df, groups=df["SID"])
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    print("\nChecking projection intervals:")
    print(test.groupby("HORIZON_HOURS")["FUTURE_HOURS"].agg(["mean", "min", "max"]))
    assert set(train["SID"]).isdisjoint(set(test["SID"]))

    assert set(train["SID"]) | set(test["SID"]) == set(df["SID"])
    print("Train/test cyclone leakage check: PASSED")
    print("\nTraining cyclones:", train["SID"].nunique())
    print("Testing cyclones:", test["SID"].nunique())

    predicted_lat, predicted_lon = destination_point(
        test["LAT"].to_numpy(),
        test["LON"].to_numpy(),
        (
            test["SPEED_KMH"] * test["FUTURE_HOURS"]
        ).to_numpy(),
        test["HEADING_DEG"].to_numpy()
    )

    test["PRED_LAT"] = predicted_lat
    test["PRED_LON"] = predicted_lon

    test["ERROR_KM"] = haversine(
        test["PRED_LAT"],
        test["PRED_LON"],
        test["TARGET_LAT"],
        test["TARGET_LON"]
    )

    test.to_csv(PREDICTIONS, index=False)

    print("\n========== FIXED HORIZON BASELINE ==========")

    for horizon in HORIZONS:
        subset = test[test["HORIZON_HOURS"] == horizon]

        if subset.empty:
            print(f"\n{horizon}h: No test samples")
            continue

        print(f"\n{horizon}-hour forecast")
        print(f"Samples: {len(subset):,}")
        print(f"Mean error: {subset['ERROR_KM'].mean():.2f} km")
        print(f"Median error: {subset['ERROR_KM'].median():.2f} km")
        print(
            f"RMSE: "
            f"{np.sqrt(np.mean(subset['ERROR_KM'] ** 2)):.2f} km"
        )
        print(
            f"Actual horizon: "
            f"{subset['FUTURE_HOURS'].min():.2f}–"
            f"{subset['FUTURE_HOURS'].max():.2f} hours"
        )

    return train, test


def main():
    df = pd.read_csv(INPUT, parse_dates=["ISO_TIME"])

    print("Building fixed-horizon targets...")

    features = build_fixed_horizon_dataset(df)

    features.to_csv(OUTPUT, index=False)

    print("\n========== FIXED HORIZON DATASET ==========")
    print("Total samples:", len(features))
    print("Unique cyclones:", features["SID"].nunique())

    print("\nSamples by forecast horizon:")
    print(features.groupby("HORIZON_HOURS").size())

    print("\nActual forecast intervals:")
    print(
        features.groupby("HORIZON_HOURS")["FUTURE_HOURS"]
        .agg(["count", "mean", "min", "max"])
    )

    print("\nSaved:", OUTPUT)

    evaluate(features)


if __name__ == "__main__":
    main()
