
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterSampler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "fixed_horizon_features.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

TARGETS = ["EAST_DISPLACEMENT", "NORTH_DISPLACEMENT"]

RANDOM_STATE = 42
N_ITER = 12

# --------------------------------------------------
# GEOGRAPHICAL UTILITIES
# --------------------------------------------------

def calculate_displacement(lat1, lon1, lat2, lon2):
    """Convert target coordinates into east/north displacement."""

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

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
    """Convert predicted east/north displacement into coordinates."""

    distance = np.sqrt(east ** 2 + north ** 2)
    bearing = np.arctan2(east, north)

    angular_distance = distance / 6371.0

    lat1 = np.radians(lat)
    lon1 = np.radians(lon)

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular_distance)
        + np.cos(lat1) * np.sin(angular_distance) * np.cos(bearing)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(angular_distance) * np.cos(lat1),
        np.cos(angular_distance) - np.sin(lat1) * np.sin(lat2),
    )

    return np.degrees(lat2), (np.degrees(lon2) + 180) % 360 - 180


def haversine_error(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between predicted and actual points."""

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

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

    month = df["ISO_TIME"].dt.month
    hour = df["ISO_TIME"].dt.hour

    df["MONTH_SIN"] = np.sin(2 * np.pi * month / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * month / 12)

    df["HOUR_SIN"] = np.sin(2 * np.pi * hour / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * hour / 24)

    east, north = calculate_displacement(
        df["LAT"].values,
        df["LON"].values,
        df["TARGET_LAT"].values,
        df["TARGET_LON"].values,
    )

    df["EAST_DISPLACEMENT"] = east
    df["NORTH_DISPLACEMENT"] = north

    # Match the temporal split used in temporal_validation.py
    df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")

    return df


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def evaluate_model(model, imputer, X, validation):

    X_imputed = imputer.transform(X)

    predicted_displacement = model.predict(X_imputed)

    predicted_lat, predicted_lon = displacement_to_coordinates(
        validation["LAT"].values,
        validation["LON"].values,
        predicted_displacement[:, 0],
        predicted_displacement[:, 1],
    )

    errors = haversine_error(
        predicted_lat,
        predicted_lon,
        validation["TARGET_LAT"].values,
        validation["TARGET_LON"].values,
    )

    return {
        "mean_error_km": float(np.mean(errors)),
        "median_error_km": float(np.median(errors)),
        "rmse_km": float(np.sqrt(np.mean(errors ** 2))),
        "p90_error_km": float(np.percentile(errors, 90)),
    }


# --------------------------------------------------
# HYPERPARAMETER SEARCH
# --------------------------------------------------

PARAMETER_SPACE = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [1, 3, 5],
}


def create_model(params):

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **params,
    )

    return MultiOutputRegressor(base_model, n_jobs=1)


def tune_horizon(df, horizon):

    print("\n" + "=" * 60)
    print(f"TUNING {horizon}-HOUR MODEL")
    print("=" * 60)

    horizon_df = df[df["HORIZON_HOURS"] == horizon].copy()

    train = horizon_df[horizon_df["SEASON"] <= 2010]
    validation = horizon_df[
        (horizon_df["SEASON"] >= 2011)
        & (horizon_df["SEASON"] <= 2017)
    ]

    print(f"Training samples: {len(train)}")
    print(f"Validation samples: {len(validation)}")

    if train.empty or validation.empty:
        raise ValueError(f"Missing training or validation data for {horizon}h")

    X_train = train[FEATURES]
    y_train = train[TARGETS]

    X_val = validation[FEATURES]

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)

    configurations = list(
        ParameterSampler(
            PARAMETER_SPACE,
            n_iter=N_ITER,
            random_state=RANDOM_STATE + horizon,
        )
    )

    results = []
    best_error = float("inf")
    best_model = None
    best_params = None
    best_metrics = None

    for index, params in enumerate(configurations, start=1):

        print(f"\nConfiguration {index}/{N_ITER}")
        print(params)

        model = create_model(params)
        model.fit(X_train_imputed, y_train)

        metrics = evaluate_model(
            model,
            imputer,
            X_val,
            validation,
        )

        print(
            f"Mean error: {metrics['mean_error_km']:.2f} km | "
            f"Median: {metrics['median_error_km']:.2f} km | "
            f"RMSE: {metrics['rmse_km']:.2f} km"
        )

        result = {
            "horizon": horizon,
            "configuration": index,
            **params,
            **metrics,
        }

        results.append(result)

        if metrics["mean_error_km"] < best_error:

            best_error = metrics["mean_error_km"]
            best_model = model
            best_params = params.copy()
            best_metrics = metrics.copy()

            print("New best configuration!")

    # Save the best candidate separately.
    model_path = MODEL_DIR / f"tuned_candidate_{horizon}h.joblib"

    joblib.dump(
        {
            "model": best_model,
            "imputer": imputer,
            "features": FEATURES,
            "params": best_params,
            "validation_metrics": best_metrics,
        },
        model_path,
    )

    print(f"\nBest {horizon}h validation error: {best_error:.2f} km")
    print(f"Saved candidate: {model_path}")

    return results


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Loading fixed-horizon dataset...")

    df = pd.read_csv(DATA_PATH)
    df = prepare_data(df)

    all_results = []

    for horizon in HORIZONS:
        results = tune_horizon(df, horizon)
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    results_path = OUTPUT_DIR / "hyperparameter_tuning_results.csv"
    results_df.to_csv(results_path, index=False)

    best_df = (
        results_df.sort_values("mean_error_km")
        .groupby("horizon", as_index=False)
        .first()
    )

    best_df.to_csv(
        OUTPUT_DIR / "hyperparameter_tuning_best.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 60)

    print(best_df[
        ["horizon", "mean_error_km", "median_error_km", "rmse_km"]
    ].to_string(index=False))

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
