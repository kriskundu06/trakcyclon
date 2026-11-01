
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "fixed_horizon_features.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed"

HORIZONS = [6, 12, 24]

FEATURES = [
    "LAT", "LON",
    "PREV_LAT", "PREV_LON",
    "DELTA_HOURS",
    "DISTANCE_KM",
    "SPEED_KMH",
    "HEADING_SIN",
    "HEADING_COS",
    "MONTH_SIN",
    "MONTH_COS",
    "HOUR_SIN",
    "HOUR_COS",
]

# --------------------------------------------------
# GEOGRAPHICAL UTILITIES
# --------------------------------------------------

def calculate_displacement(lat1, lon1, lat2, lon2):
    """Calculate eastward and northward displacement in km."""

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    central_angle = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    distance = 6371.0 * central_angle

    bearing = np.arctan2(
        np.sin(dlon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )

    east = distance * np.sin(bearing)
    north = distance * np.cos(bearing)

    return east, north


def displacement_to_coordinates(lat, lon, east, north):
    """Convert east/north displacement into latitude/longitude."""

    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))

    distance = np.sqrt(east ** 2 + north ** 2)
    bearing = np.arctan2(east, north)

    angular_distance = distance / 6371.0

    new_lat = np.arcsin(
        np.sin(lat) * np.cos(angular_distance)
        + np.cos(lat) * np.sin(angular_distance) * np.cos(bearing)
    )

    new_lon = lon + np.arctan2(
        np.sin(bearing) * np.sin(angular_distance) * np.cos(lat),
        np.cos(angular_distance) - np.sin(lat) * np.sin(new_lat),
    )

    return (
        np.degrees(new_lat),
        (np.degrees(new_lon) + 180) % 360 - 180,
    )


def haversine_error(lat1, lon1, lat2, lon2):
    """Calculate geographical prediction error in kilometres."""

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

def prepare_data(df):

    df = df.copy()

    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True)
    df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")

    month = df["ISO_TIME"].dt.month
    hour = df["ISO_TIME"].dt.hour

    df["MONTH_SIN"] = np.sin(2 * np.pi * month / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * month / 12)

    df["HOUR_SIN"] = np.sin(2 * np.pi * hour / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * hour / 24)

    return df


# --------------------------------------------------
# MODEL LOADING AND PREDICTION
# --------------------------------------------------

def load_model(path):

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)


def predict_displacement(artifact, X):

    # Tuned candidates were saved as dictionaries
    # containing model and imputer separately.

    if isinstance(artifact, dict):

        model = artifact["model"]
        imputer = artifact.get("imputer")

        if imputer is not None:
            X = imputer.transform(X)

        return model.predict(X)

    # Original models may be saved as sklearn Pipelines.
    # Pipeline.predict() handles its own preprocessing.

    return artifact.predict(X)


# --------------------------------------------------
# CONSTANT VELOCITY BASELINE
# --------------------------------------------------

def predict_constant_velocity(test):

    east, north = calculate_displacement(
        test["PREV_LAT"].values,
        test["PREV_LON"].values,
        test["LAT"].values,
        test["LON"].values,
    )

    delta_hours = test["DELTA_HOURS"].values
    future_hours = test["FUTURE_HOURS"].values

    scale = future_hours / delta_hours

    east *= scale
    north *= scale

    return displacement_to_coordinates(
        test["LAT"].values,
        test["LON"].values,
        east,
        north,
    )


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(errors):

    return {
        "mean_error_km": np.mean(errors),
        "median_error_km": np.median(errors),
        "rmse_km": np.sqrt(np.mean(errors ** 2)),
        "p90_error_km": np.percentile(errors, 90),
    }


def evaluate_predictions(test, predicted_lat, predicted_lon):

    return haversine_error(
        predicted_lat,
        predicted_lon,
        test["TARGET_LAT"].values,
        test["TARGET_LON"].values,
    )


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)
    df = prepare_data(df)

    # Strictly untouched temporal test period.
    test_df = df[
        (df["SEASON"] >= 2018)
        & (df["SEASON"] <= 2026)
    ].copy()

    print(f"Test samples: {len(test_df)}")
    print(f"Test cyclones: {test_df['SID'].nunique()}")

    all_metrics = []
    all_predictions = []

    for horizon in HORIZONS:

        print("\n" + "=" * 60)
        print(f"EVALUATING {horizon}-HOUR MODELS")
        print("=" * 60)

        test = test_df[
            test_df["HORIZON_HOURS"] == horizon
        ].copy()

        if test.empty:
            raise ValueError(f"No test samples for {horizon}h")

        X_test = test[FEATURES]

        # ------------------------------------------
        # 1. ORIGINAL XGBOOST
        # ------------------------------------------

        original_path = MODEL_DIR / f"temporal_best_{horizon}h.joblib"
        original_model = load_model(original_path)

        original_displacement = predict_displacement(
            original_model, X_test
        )

        original_lat, original_lon = displacement_to_coordinates(
            test["LAT"].values,
            test["LON"].values,
            original_displacement[:, 0],
            original_displacement[:, 1],
        )

        original_errors = evaluate_predictions(
            test, original_lat, original_lon
        )

        # ------------------------------------------
        # 2. TUNED XGBOOST
        # ------------------------------------------

        tuned_path = MODEL_DIR / f"tuned_candidate_{horizon}h.joblib"
        tuned_model = load_model(tuned_path)

        tuned_displacement = predict_displacement(
            tuned_model, X_test
        )

        tuned_lat, tuned_lon = displacement_to_coordinates(
            test["LAT"].values,
            test["LON"].values,
            tuned_displacement[:, 0],
            tuned_displacement[:, 1],
        )

        tuned_errors = evaluate_predictions(
            test, tuned_lat, tuned_lon
        )

        # ------------------------------------------
        # 3. CONSTANT VELOCITY
        # ------------------------------------------

        cv_lat, cv_lon = predict_constant_velocity(test)

        cv_errors = evaluate_predictions(
            test, cv_lat, cv_lon
        )

        # ------------------------------------------
        # STORE RESULTS
        # ------------------------------------------

        model_results = {
            "Original_XGBoost": original_errors,
            "Tuned_XGBoost": tuned_errors,
            "Constant_Velocity": cv_errors,
        }

        for model_name, errors in model_results.items():

            metrics = calculate_metrics(errors)

            all_metrics.append({
                "horizon": horizon,
                "model": model_name,
                "test_samples": len(test),
                **metrics,
            })

            print(f"\n{model_name}")
            print(f"Mean error:   {metrics['mean_error_km']:.2f} km")
            print(f"Median error: {metrics['median_error_km']:.2f} km")
            print(f"RMSE:         {metrics['rmse_km']:.2f} km")
            print(f"P90 error:    {metrics['p90_error_km']:.2f} km")

        # Save per-observation predictions for later analysis.
        prediction_df = test[
            ["SID", "SEASON", "NAME", "HORIZON_HOURS",
             "LAT", "LON", "TARGET_LAT", "TARGET_LON"]
        ].copy()

        prediction_df["original_error_km"] = original_errors
        prediction_df["tuned_error_km"] = tuned_errors
        prediction_df["cv_error_km"] = cv_errors

        all_predictions.append(prediction_df)

        improvement = (
            (np.mean(original_errors) - np.mean(tuned_errors))
            / np.mean(original_errors)
        ) * 100

        print("\nTuned vs original improvement:")
        print(f"{improvement:.2f}%")

    # --------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------

    metrics_df = pd.DataFrame(all_metrics)

    metrics_path = OUTPUT_DIR / "tuned_test_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_df = pd.concat(
        all_predictions, ignore_index=True
    )

    predictions_path = OUTPUT_DIR / "tuned_test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    print("\n" + "=" * 60)
    print("FINAL TEST COMPARISON")
    print("=" * 60)

    print(
        metrics_df[
            ["horizon", "model", "mean_error_km",
             "median_error_km", "rmse_km", "p90_error_km"]
        ].round(2).to_string(index=False)
    )

    print(f"\nMetrics saved: {metrics_path}")
    print(f"Predictions saved: {predictions_path}")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
