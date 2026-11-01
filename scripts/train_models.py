
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = "data/processed/fixed_horizon_features.csv"
MODEL_DIR = "models"

HORIZONS = [6, 12, 24]
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# ==========================================
# GEOGRAPHICAL FUNCTIONS
# ==========================================

EARTH_RADIUS_KM = 6371.0


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

    x = np.sin(dlon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def destination_point(lat, lon, distance_km, bearing_deg):
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    theta = np.radians(bearing_deg)

    angular_distance = distance_km / EARTH_RADIUS_KM

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular_distance)
        + np.cos(lat1) * np.sin(angular_distance)
        * np.cos(theta)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(theta) * np.sin(angular_distance)
        * np.cos(lat1),
        np.cos(angular_distance)
        - np.sin(lat1) * np.sin(lat2)
    )

    return np.degrees(lat2), (
        (np.degrees(lon2) + 540) % 360
    ) - 180


# ==========================================
# LOAD AND PREPARE DATA
# ==========================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["ISO_TIME"] = pd.to_datetime(
    df["ISO_TIME"], utc=True
)

# Cyclical time features
month_angle = 2 * np.pi * df["ISO_TIME"].dt.month / 12
hour_angle = 2 * np.pi * df["ISO_TIME"].dt.hour / 24

df["MONTH_SIN"] = np.sin(month_angle)
df["MONTH_COS"] = np.cos(month_angle)

df["HOUR_SIN"] = np.sin(hour_angle)
df["HOUR_COS"] = np.cos(hour_angle)


# ==========================================
# INPUT FEATURES
# ==========================================

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
    "HOUR_COS"
]


# ==========================================
# CREATE TARGETS
# ==========================================

print("Calculating movement targets...")

distance = haversine(
    df["LAT"].values,
    df["LON"].values,
    df["TARGET_LAT"].values,
    df["TARGET_LON"].values
)

direction = np.radians(
    bearing(
        df["LAT"].values,
        df["LON"].values,
        df["TARGET_LAT"].values,
        df["TARGET_LON"].values
    )
)

# Positive east = eastward movement
df["TARGET_EAST_KM"] = distance * np.sin(direction)

# Positive north = northward movement
df["TARGET_NORTH_KM"] = distance * np.cos(direction)


# ==========================================
# STORM-WISE TRAIN / TEST SPLIT
# ==========================================

print("Creating cyclone-wise train/test split...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(df, groups=df["SID"])
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print("Training cyclones:", train_df["SID"].nunique())
print("Testing cyclones:", test_df["SID"].nunique())


# ==========================================
# MODEL DEFINITIONS
# ==========================================

def create_models():

    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
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
                n_jobs=-1
            ),
            n_jobs=1
        ))
    ])

    return {
        "random_forest": random_forest,
        "xgboost": xgboost
    }


# ==========================================
# EVALUATION
# ==========================================

def evaluate_model(model, X_test, test_data):

    predictions = model.predict(X_test)

    east_pred = predictions[:, 0]
    north_pred = predictions[:, 1]

    predicted_distance = np.sqrt(
        east_pred ** 2 + north_pred ** 2
    )

    predicted_bearing = (
        np.degrees(np.arctan2(east_pred, north_pred))
        + 360
    ) % 360

    pred_lat, pred_lon = destination_point(
        test_data["LAT"].values,
        test_data["LON"].values,
        predicted_distance,
        predicted_bearing
    )

    errors = haversine(
        pred_lat,
        pred_lon,
        test_data["TARGET_LAT"].values,
        test_data["TARGET_LON"].values
    )

    return {
        "mean_error_km": float(np.mean(errors)),
        "median_error_km": float(np.median(errors)),
        "rmse_km": float(np.sqrt(np.mean(errors ** 2))),
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "errors": errors
    }


# ==========================================
# TRAIN ALL MODELS
# ==========================================

all_metrics = []
all_predictions = []

for horizon in HORIZONS:

    print("\n" + "=" * 50)
    print(f"TRAINING FOR {horizon}-HOUR FORECAST")
    print("=" * 50)

    train_h = train_df[
        train_df["HORIZON_HOURS"] == horizon
    ].copy()

    test_h = test_df[
        test_df["HORIZON_HOURS"] == horizon
    ].copy()

    X_train = train_h[FEATURES]
    X_test = test_h[FEATURES]

    y_train = train_h[
        ["TARGET_EAST_KM", "TARGET_NORTH_KM"]
    ]

    models = create_models()

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        result = evaluate_model(
            model, X_test, test_h
        )

        model_path = (
            f"{MODEL_DIR}/{model_name}_{horizon}h.joblib"
        )

        joblib.dump(model, model_path)

        print("Mean error:", round(
            result["mean_error_km"], 2
        ), "km")

        print("Median error:", round(
            result["median_error_km"], 2
        ), "km")

        print("RMSE:", round(
            result["rmse_km"], 2
        ), "km")

        all_metrics.append({
            "horizon": horizon,
            "model": model_name,
            "mean_error_km": result["mean_error_km"],
            "median_error_km": result["median_error_km"],
            "rmse_km": result["rmse_km"]
        })

        predictions_df = pd.DataFrame({
            "SID": test_h["SID"].values,
            "HORIZON_HOURS": horizon,
            "MODEL": model_name,
            "ACTUAL_LAT": test_h["TARGET_LAT"].values,
            "ACTUAL_LON": test_h["TARGET_LON"].values,
            "PRED_LAT": result["pred_lat"],
            "PRED_LON": result["pred_lon"],
            "ERROR_KM": result["errors"]
        })

        all_predictions.append(predictions_df)


# ==========================================
# SAVE RESULTS
# ==========================================

metrics_df = pd.DataFrame(all_metrics)

metrics_df.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_df.to_csv(
    "data/processed/ml_predictions.csv",
    index=False
)

print("\n========== FINAL MODEL COMPARISON ==========")

print(
    metrics_df.sort_values(
        ["horizon", "mean_error_km"]
    ).to_string(index=False)
)

print("\nModels saved in:", MODEL_DIR)
print("Metrics saved to model_comparison.csv")
print("Predictions saved to ml_predictions.csv")
print("\nTRAINING COMPLETE!")
