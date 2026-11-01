
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "cyclones_clean.csv"
FEATURE_FILE = ROOT / "data" / "processed" / "cyclone_features.csv"
PREDICTION_FILE = ROOT / "data" / "processed" / "baseline_predictions.csv"

EARTH_RADIUS_KM = 6371.0
MAX_INTERVAL_HOURS = 12.0


def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in kilometres."""

    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_KM * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


def bearing(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, in degrees."""

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
    """Project a point along a great-circle bearing."""

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


def create_features(df):
    """Create historical movement features and next-position targets."""

    df = df.sort_values(["SID", "ISO_TIME"]).copy()

    grouped = df.groupby("SID", sort=False)

    df["PREV_LAT"] = grouped["LAT"].shift(1)
    df["PREV_LON"] = grouped["LON"].shift(1)
    df["PREV_TIME"] = grouped["ISO_TIME"].shift(1)

    df["TARGET_LAT"] = grouped["LAT"].shift(-1)
    df["TARGET_LON"] = grouped["LON"].shift(-1)
    df["TARGET_TIME"] = grouped["ISO_TIME"].shift(-1)

    df["DELTA_HOURS"] = (
        df["ISO_TIME"] - df["PREV_TIME"]
    ).dt.total_seconds() / 3600

    df["FUTURE_HOURS"] = (
        df["TARGET_TIME"] - df["ISO_TIME"]
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

    heading_rad = np.radians(df["HEADING_DEG"])

    # Circular encoding avoids the 359-degree / 1-degree problem.
    df["HEADING_SIN"] = np.sin(heading_rad)
    df["HEADING_COS"] = np.cos(heading_rad)

    # Keep only valid consecutive three-point sequences.
    valid = (
        df["PREV_LAT"].notna()
        & df["TARGET_LAT"].notna()
        & df["DELTA_HOURS"].gt(0)
        & df["FUTURE_HOURS"].gt(0)
        & df["DELTA_HOURS"].le(MAX_INTERVAL_HOURS)
        & df["FUTURE_HOURS"].le(MAX_INTERVAL_HOURS)
    )

    df = df[valid].copy()

    # Remove impossible or clearly corrupted movement speeds.
    df = df[
        np.isfinite(df["SPEED_KMH"])
        & df["SPEED_KMH"].between(0, 150)
    ].copy()

    return df.reset_index(drop=True)


def evaluate_baseline(df):
    """Evaluate constant-velocity extrapolation on unseen cyclones."""

    if df["SID"].nunique() < 4:
        raise ValueError(
            "At least four cyclones are recommended for a "
            "meaningful train/test split."
        )

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

    print(f"\nTraining cyclones: {train['SID'].nunique()}")
    print(f"Testing cyclones: {test['SID'].nunique()}")
    print(f"Training samples: {len(train)}")
    print(f"Testing samples: {len(test)}")

    # Distance travelled in the previous interval.
    # Extrapolate the same speed and heading over the next interval.
    projected_distance = (
        test["SPEED_KMH"] * test["FUTURE_HOURS"]
    )

    pred_lat, pred_lon = destination_point(
        test["LAT"].to_numpy(),
        test["LON"].to_numpy(),
        projected_distance.to_numpy(),
        test["HEADING_DEG"].to_numpy()
    )

    test["PRED_LAT"] = pred_lat
    test["PRED_LON"] = pred_lon

    test["ERROR_KM"] = haversine(
        test["PRED_LAT"],
        test["PRED_LON"],
        test["TARGET_LAT"],
        test["TARGET_LON"]
    )

    # Save predictions for later comparison and map visualization.
    test.to_csv(PREDICTION_FILE, index=False)

    mae = test["ERROR_KM"].mean()
    median_error = test["ERROR_KM"].median()
    rmse = np.sqrt(np.mean(test["ERROR_KM"] ** 2))

    print("\n========== BASELINE RESULTS ==========")
    print(f"Mean track error:   {mae:.2f} km")
    print(f"Median track error: {median_error:.2f} km")
    print(f"RMSE:               {rmse:.2f} km")

    print("\nError by forecast horizon:")

    test["HORIZON_GROUP"] = pd.cut(
        test["FUTURE_HOURS"],
        bins=[0, 3, 6, 9, 12],
        include_lowest=True
    )

    print(
        test.groupby(
            "HORIZON_GROUP",
            observed=True
        )["ERROR_KM"].agg(["count", "mean", "median"])
    )

    print(f"\nPredictions saved: {PREDICTION_FILE}")

    return train, test


def main():
    print("Loading cleaned cyclone data...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["ISO_TIME"]
    )

    print("Generating movement features...")

    features = create_features(df)

    features.to_csv(FEATURE_FILE, index=False)

    print("\n========== FEATURE REPORT ==========")
    print(f"Feature rows: {len(features):,}")
    print(f"Unique cyclones: {features['SID'].nunique():,}")
    print(f"Mean speed: {features['SPEED_KMH'].mean():.2f} km/h")
    print(f"Median speed: {features['SPEED_KMH'].median():.2f} km/h")
    print(f"Feature file: {FEATURE_FILE}")

    if features.empty:
        raise ValueError("No valid feature rows were generated.")

    evaluate_baseline(features)


if __name__ == "__main__":
    main()
