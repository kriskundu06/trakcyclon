
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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

    distance = 6371.0 * 2 * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )

    bearing = np.arctan2(
        np.sin(dlon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )

    east = distance * np.sin(bearing)
    north = distance * np.cos(bearing)

    return east, north


def displacement_to_coordinates(lat, lon, east, north):

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

    return 6371.0 * 2 * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

def prepare_data(df):

    df = df.copy()

    df["ISO_TIME"] = pd.to_datetime(
        df["ISO_TIME"], utc=True
    )

    df["SEASON"] = pd.to_numeric(
        df["SEASON"], errors="coerce"
    )

    month = df["ISO_TIME"].dt.month
    hour = df["ISO_TIME"].dt.hour

    df["MONTH_SIN"] = np.sin(2 * np.pi * month / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * month / 12)

    df["HOUR_SIN"] = np.sin(2 * np.pi * hour / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * hour / 24)

    return df


# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------

def predict_model(artifact, X):

    if isinstance(artifact, dict):

        model = artifact["model"]
        imputer = artifact["imputer"]

        X = imputer.transform(X)

        return model.predict(X)

    # Original models may be sklearn pipelines.
    return artifact.predict(X)


def evaluate_model(artifact, test):

    displacement = predict_model(
        artifact,
        test[FEATURES]
    )

    predicted_lat, predicted_lon = displacement_to_coordinates(
        test["LAT"].values,
        test["LON"].values,
        displacement[:, 0],
        displacement[:, 1],
    )

    return haversine_error(
        predicted_lat,
        predicted_lon,
        test["TARGET_LAT"].values,
        test["TARGET_LON"].values,
    )


# --------------------------------------------------
# CONSTANT VELOCITY BASELINE
# --------------------------------------------------

def evaluate_constant_velocity(test):

    east, north = calculate_displacement(
        test["PREV_LAT"].values,
        test["PREV_LON"].values,
        test["LAT"].values,
        test["LON"].values,
    )

    scale = (
        test["FUTURE_HOURS"].values
        / test["DELTA_HOURS"].values
    )

    predicted_lat, predicted_lon = displacement_to_coordinates(
        test["LAT"].values,
        test["LON"].values,
        east * scale,
        north * scale,
    )

    return haversine_error(
        predicted_lat,
        predicted_lon,
        test["TARGET_LAT"].values,
        test["TARGET_LON"].values,
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


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)
    df = prepare_data(df)

    # Keep the final test period completely separate.
    test_df = df[
        (df["SEASON"] >= 2018)
        & (df["SEASON"] <= 2026)
    ].copy()

    print(f"Test samples: {len(test_df)}")
    print(f"Test cyclones: {test_df['SID'].nunique()}")

    all_results = []
    all_predictions = []

    for horizon in HORIZONS:

        print("\n" + "=" * 65)
        print(f"FINAL EVALUATION: {horizon}-HOUR MODEL")
        print("=" * 65)

        test = test_df[
            test_df["HORIZON_HOURS"] == horizon
        ].copy()

        if test.empty:
            raise ValueError(f"No test data for {horizon}h")

        model_files = {
            "Original_XGBoost":
                f"temporal_best_{horizon}h.joblib",

            "Tuned_XGBoost":
                f"tuned_candidate_{horizon}h.joblib",

            "Final_Retrained_XGBoost":
                f"final_cyclone_model_{horizon}h.joblib",
        }

        prediction_df = test[
            [
                "SID", "SEASON", "NAME",
                "HORIZON_HOURS",
                "LAT", "LON",
                "TARGET_LAT", "TARGET_LON",
            ]
        ].copy()

        error_arrays = {}

        # Evaluate the three ML models.
        for model_name, filename in model_files.items():

            model_path = MODEL_DIR / filename

            if not model_path.exists():
                raise FileNotFoundError(model_path)

            print(f"\nEvaluating {model_name}...")

            artifact = joblib.load(model_path)

            errors = evaluate_model(artifact, test)
            error_arrays[model_name] = errors

            metrics = calculate_metrics(errors)

            all_results.append({
                "horizon": horizon,
                "model": model_name,
                "test_samples": len(test),
                **metrics,
            })

            prediction_df[
                f"{model_name}_error_km"
            ] = errors

            print(
                f"Mean:   {metrics['mean_error_km']:.2f} km\n"
                f"Median: {metrics['median_error_km']:.2f} km\n"
                f"RMSE:   {metrics['rmse_km']:.2f} km\n"
                f"P90:    {metrics['p90_error_km']:.2f} km"
            )

        # Evaluate constant velocity.
        cv_errors = evaluate_constant_velocity(test)
        error_arrays["Constant_Velocity"] = cv_errors

        cv_metrics = calculate_metrics(cv_errors)

        all_results.append({
            "horizon": horizon,
            "model": "Constant_Velocity",
            "test_samples": len(test),
            **cv_metrics,
        })

        prediction_df["Constant_Velocity_error_km"] = cv_errors

        print("\nConstant Velocity:")
        print(
            f"Mean: {cv_metrics['mean_error_km']:.2f} km | "
            f"Median: {cv_metrics['median_error_km']:.2f} km | "
            f"RMSE: {cv_metrics['rmse_km']:.2f} km"
        )

        # Compare final retrained model with previous tuned model.
        tuned_mean = np.mean(
            error_arrays["Tuned_XGBoost"]
        )

        final_mean = np.mean(
            error_arrays["Final_Retrained_XGBoost"]
        )

        improvement = (
            (tuned_mean - final_mean) / tuned_mean
        ) * 100

        print("\nFinal retrained vs previous tuned model:")

        if improvement > 0:
            print(f"{improvement:.2f}% lower mean error")
        elif improvement < 0:
            print(f"{abs(improvement):.2f}% higher mean error")
        else:
            print("No change in mean error")

        all_predictions.append(prediction_df)

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(all_results)

    metrics_path = (
        OUTPUT_DIR / "final_model_test_comparison.csv"
    )

    results_df.to_csv(metrics_path, index=False)

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    predictions_path = (
        OUTPUT_DIR / "final_model_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)

    display_columns = [
        "horizon",
        "model",
        "mean_error_km",
        "median_error_km",
        "rmse_km",
        "p90_error_km",
    ]

    print(
        results_df[display_columns]
        .round(2)
        .to_string(index=False)
    )

    print(f"\nMetrics saved: {metrics_path}")
    print(f"Predictions saved: {predictions_path}")

    print("\nFINAL EVALUATION COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
