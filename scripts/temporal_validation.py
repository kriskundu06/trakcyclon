from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "fixed_horizon_features.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed"

HORIZONS = [6, 12, 24]
RANDOM_STATE = 42
TRAIN_END_YEAR = 2010
VALIDATION_END_YEAR = 2017

EARTH_RADIUS_KM = 6371.0

FEATURES = [
    "LAT",
    "LON",
    "PREV_LAT",
    "PREV_LON",
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

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GEOGRAPHICAL FUNCTIONS
# ============================================================

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
        np.cos(angular_distance) - np.sin(lat1) * np.sin(lat2),
    )

    return np.degrees(lat2), (np.degrees(lon2) + 540) % 360 - 180


# ============================================================
# DATA PREPARATION
# ============================================================

def load_and_prepare_data():
    print("Loading fixed-horizon dataset...")
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "SID", "SEASON", "ISO_TIME", "HORIZON_HOURS",
        "LAT", "LON", "TARGET_LAT", "TARGET_LON",
        "FUTURE_HOURS",
    ]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True, errors="coerce")
    df = df.dropna(subset=["SID", "SEASON", "ISO_TIME"]).copy()
    df["SEASON"] = df["SEASON"].astype(int)

    # Time-of-year and time-of-day features, matching train_models.py.
    month_angle = 2 * np.pi * df["ISO_TIME"].dt.month / 12
    hour_angle = 2 * np.pi * df["ISO_TIME"].dt.hour / 24

    df["MONTH_SIN"] = np.sin(month_angle)
    df["MONTH_COS"] = np.cos(month_angle)
    df["HOUR_SIN"] = np.sin(hour_angle)
    df["HOUR_COS"] = np.cos(hour_angle)

    # Predict displacement in east/north kilometres, matching train_models.py.
    distance = haversine(
        df["LAT"].to_numpy(),
        df["LON"].to_numpy(),
        df["TARGET_LAT"].to_numpy(),
        df["TARGET_LON"].to_numpy(),
    )

    direction = np.radians(
        bearing(
            df["LAT"].to_numpy(),
            df["LON"].to_numpy(),
            df["TARGET_LAT"].to_numpy(),
            df["TARGET_LON"].to_numpy(),
        )
    )

    df["TARGET_EAST_KM"] = distance * np.sin(direction)
    df["TARGET_NORTH_KM"] = distance * np.cos(direction)

    if df[FEATURES].isna().all(axis=1).any():
        raise ValueError("Some rows have no usable input features.")

    return df


# ============================================================
# MODELS
# ============================================================

def create_models():
    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    xgboost = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            n_jobs=1,
        )),
    ])

    return {
        "random_forest": random_forest,
        "xgboost": xgboost,
    }


# ============================================================
# PREDICTION AND EVALUATION
# ============================================================

def predict_coordinates(model, data):
    predictions = model.predict(data[FEATURES])

    east_pred = predictions[:, 0]
    north_pred = predictions[:, 1]

    predicted_distance = np.sqrt(east_pred ** 2 + north_pred ** 2)
    predicted_bearing = (
        np.degrees(np.arctan2(east_pred, north_pred)) + 360
    ) % 360

    return destination_point(
        data["LAT"].to_numpy(),
        data["LON"].to_numpy(),
        predicted_distance,
        predicted_bearing,
    )


def calculate_metrics(errors):
    errors = np.asarray(errors, dtype=float)
    return {
        "mean_error_km": float(np.mean(errors)),
        "median_error_km": float(np.median(errors)),
        "rmse_km": float(np.sqrt(np.mean(errors ** 2))),
        "p90_error_km": float(np.percentile(errors, 90)),
    }


def evaluate_model(model, data):
    pred_lat, pred_lon = predict_coordinates(model, data)

    errors = haversine(
        pred_lat,
        pred_lon,
        data["TARGET_LAT"].to_numpy(),
        data["TARGET_LON"].to_numpy(),
    )

    return pred_lat, pred_lon, errors, calculate_metrics(errors)


def evaluate_constant_velocity(data):
    # Use the actual matched interval, not nominal 6/12/24 hours.
    pred_lat, pred_lon = destination_point(
        data["LAT"].to_numpy(),
        data["LON"].to_numpy(),
        (data["SPEED_KMH"] * data["FUTURE_HOURS"]).to_numpy(),
        data["HEADING_DEG"].to_numpy(),
    )

    errors = haversine(
        pred_lat,
        pred_lon,
        data["TARGET_LAT"].to_numpy(),
        data["TARGET_LON"].to_numpy(),
    )

    return pred_lat, pred_lon, errors, calculate_metrics(errors)


def assert_no_cyclone_leakage(train, validation, test):
    train_ids = set(train["SID"].unique())
    validation_ids = set(validation["SID"].unique())
    test_ids = set(test["SID"].unique())

    assert train_ids.isdisjoint(validation_ids), "Train/validation cyclone overlap!"
    assert train_ids.isdisjoint(test_ids), "Train/test cyclone overlap!"
    assert validation_ids.isdisjoint(test_ids), "Validation/test cyclone overlap!"

    print("Cyclone leakage check: PASSED")


def print_partition_summary(name, data):
    print(
        f"{name}: {len(data):,} observations | "
        f"{data['SID'].nunique():,} cyclones | "
        f"seasons {data['SEASON'].min()}–{data['SEASON'].max()}"
    )


# ============================================================
# TEMPORAL TRAINING WORKFLOW
# ============================================================

def main():
    df = load_and_prepare_data()

    train_df = df[df["SEASON"] <= TRAIN_END_YEAR].copy()
    validation_df = df[
        (df["SEASON"] > TRAIN_END_YEAR)
        & (df["SEASON"] <= VALIDATION_END_YEAR)
    ].copy()
    test_df = df[df["SEASON"] > VALIDATION_END_YEAR].copy()

    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError("One or more temporal partitions are empty.")

    print("\n========== TEMPORAL PARTITIONS ==========")
    print_partition_summary("Training", train_df)
    print_partition_summary("Validation", validation_df)
    print_partition_summary("Testing", test_df)

    assert_no_cyclone_leakage(train_df, validation_df, test_df)

    validation_metrics = []
    test_metrics = []
    all_predictions = []

    for horizon in HORIZONS:
        print("\n" + "=" * 65)
        print(f"FORECAST HORIZON: {horizon} HOURS")
        print("=" * 65)

        train_h = train_df[train_df["HORIZON_HOURS"] == horizon].copy()
        validation_h = validation_df[
            validation_df["HORIZON_HOURS"] == horizon
        ].copy()
        test_h = test_df[test_df["HORIZON_HOURS"] == horizon].copy()

        if train_h.empty or validation_h.empty or test_h.empty:
            print(f"Skipping {horizon}h: a partition has no samples.")
            continue

        print(
            f"Samples — train: {len(train_h):,}, "
            f"validation: {len(validation_h):,}, test: {len(test_h):,}"
        )

        X_train = train_h[FEATURES]
        y_train = train_h[["TARGET_EAST_KM", "TARGET_NORTH_KM"]]

        # First evaluate Constant Velocity and both ML models on validation.
        _, _, cv_val_errors, cv_val_metrics = evaluate_constant_velocity(
            validation_h
        )
        validation_metrics.append({
            "horizon": horizon,
            "model": "constant_velocity",
            "partition": "validation",
            "samples": len(validation_h),
            **cv_val_metrics,
        })

        validation_model_results = {}

        for model_name, model in create_models().items():
            print(f"\nTraining {model_name} on seasons <= {TRAIN_END_YEAR}...")
            model.fit(X_train, y_train)

            pred_lat, pred_lon, errors, metrics = evaluate_model(
                model, validation_h
            )

            validation_model_results[model_name] = {
                "model": model,
                "metrics": metrics,
            }

            validation_metrics.append({
                "horizon": horizon,
                "model": model_name,
                "partition": "validation",
                "samples": len(validation_h),
                **metrics,
            })

            print(
                f"Validation {model_name}: "
                f"mean={metrics['mean_error_km']:.2f} km | "
                f"median={metrics['median_error_km']:.2f} km | "
                f"RMSE={metrics['rmse_km']:.2f} km | "
                f"P90={metrics['p90_error_km']:.2f} km"
            )

            val_predictions = pd.DataFrame({
                "SID": validation_h["SID"].to_numpy(),
                "SEASON": validation_h["SEASON"].to_numpy(),
                "HORIZON_HOURS": horizon,
                "PARTITION": "validation",
                "MODEL": model_name,
                "ACTUAL_LAT": validation_h["TARGET_LAT"].to_numpy(),
                "ACTUAL_LON": validation_h["TARGET_LON"].to_numpy(),
                "PRED_LAT": pred_lat,
                "PRED_LON": pred_lon,
                "ERROR_KM": errors,
            })
            all_predictions.append(val_predictions)

        print(
            f"\nValidation Constant Velocity: "
            f"mean={cv_val_metrics['mean_error_km']:.2f} km | "
            f"median={cv_val_metrics['median_error_km']:.2f} km | "
            f"RMSE={cv_val_metrics['rmse_km']:.2f} km | "
            f"P90={cv_val_metrics['p90_error_km']:.2f} km"
        )

        # Select one model per horizon using validation mean error only.
        selected_name = min(
            validation_model_results,
            key=lambda name: validation_model_results[name]["metrics"]["mean_error_km"],
        )
        print(f"\nSelected for {horizon}h based on validation: {selected_name}")

        # Refit the selected model on all data available before the test period.
        development_df = pd.concat(
            [train_h, validation_h],
            ignore_index=True,
        )
        final_model = create_models()[selected_name]
        final_model.fit(
            development_df[FEATURES],
            development_df[["TARGET_EAST_KM", "TARGET_NORTH_KM"]],
        )

        model_path = MODEL_DIR / f"temporal_best_{horizon}h.joblib"
        joblib.dump(final_model, model_path)
        print(f"Saved final model: {model_path}")

        # The test set is evaluated only after model selection is complete.
        test_pred_lat, test_pred_lon, test_errors, final_metrics = evaluate_model(
            final_model, test_h
        )

        test_metrics.append({
            "horizon": horizon,
            "selected_model": selected_name,
            "partition": "test",
            "samples": len(test_h),
            **final_metrics,
        })

        print(
            f"\nFINAL TEST — {selected_name}: "
            f"mean={final_metrics['mean_error_km']:.2f} km | "
            f"median={final_metrics['median_error_km']:.2f} km | "
            f"RMSE={final_metrics['rmse_km']:.2f} km | "
            f"P90={final_metrics['p90_error_km']:.2f} km"
        )

        # Constant Velocity test baseline.
        cv_pred_lat, cv_pred_lon, cv_test_errors, cv_test_metrics = (
            evaluate_constant_velocity(test_h)
        )
        test_metrics.append({
            "horizon": horizon,
            "selected_model": "constant_velocity",
            "partition": "test",
            "samples": len(test_h),
            **cv_test_metrics,
        })

        print(
            f"FINAL TEST — Constant Velocity: "
            f"mean={cv_test_metrics['mean_error_km']:.2f} km | "
            f"median={cv_test_metrics['median_error_km']:.2f} km | "
            f"RMSE={cv_test_metrics['rmse_km']:.2f} km | "
            f"P90={cv_test_metrics['p90_error_km']:.2f} km"
        )

        all_predictions.append(pd.DataFrame({
            "SID": test_h["SID"].to_numpy(),
            "SEASON": test_h["SEASON"].to_numpy(),
            "HORIZON_HOURS": horizon,
            "PARTITION": "test",
            "MODEL": selected_name,
            "ACTUAL_LAT": test_h["TARGET_LAT"].to_numpy(),
            "ACTUAL_LON": test_h["TARGET_LON"].to_numpy(),
            "PRED_LAT": test_pred_lat,
            "PRED_LON": test_pred_lon,
            "ERROR_KM": test_errors,
        }))

        all_predictions.append(pd.DataFrame({
            "SID": test_h["SID"].to_numpy(),
            "SEASON": test_h["SEASON"].to_numpy(),
            "HORIZON_HOURS": horizon,
            "PARTITION": "test",
            "MODEL": "constant_velocity",
            "ACTUAL_LAT": test_h["TARGET_LAT"].to_numpy(),
            "ACTUAL_LON": test_h["TARGET_LON"].to_numpy(),
            "PRED_LAT": cv_pred_lat,
            "PRED_LON": cv_pred_lon,
            "ERROR_KM": cv_test_errors,
        }))

    # Save temporal results separately; preserve existing random-split outputs.
    validation_path = OUTPUT_DIR / "temporal_validation_metrics.csv"
    test_path = OUTPUT_DIR / "temporal_test_metrics.csv"
    predictions_path = OUTPUT_DIR / "temporal_predictions.csv"

    pd.DataFrame(validation_metrics).to_csv(validation_path, index=False)
    pd.DataFrame(test_metrics).to_csv(test_path, index=False)

    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            predictions_path, index=False
        )

    print("\n========== TEMPORAL VALIDATION SUMMARY ==========")
    print(pd.DataFrame(validation_metrics).to_string(index=False))

    print("\n========== FINAL TEMPORAL TEST SUMMARY ==========")
    print(pd.DataFrame(test_metrics).to_string(index=False))

    print("\nSaved:")
    print(validation_path)
    print(test_path)
    print(predictions_path)
    print("\nTemporal training and evaluation complete.")


if __name__ == "__main__":
    main()
