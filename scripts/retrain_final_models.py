
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "fixed_horizon_features.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

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

HYPERPARAMETERS = {
    6: {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 3,
    },
    12: {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 1,
    },
    24: {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 1.0,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "reg_lambda": 3,
    },
}


def calculate_displacement(lat1, lon1, lat2, lon2):

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

    distance = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    bearing = np.arctan2(
        np.sin(dlon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )

    east = distance * np.sin(bearing)
    north = distance * np.cos(bearing)

    return east, north


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

    east, north = calculate_displacement(
        df["LAT"].values,
        df["LON"].values,
        df["TARGET_LAT"].values,
        df["TARGET_LON"].values,
    )

    df["EAST_DISPLACEMENT"] = east
    df["NORTH_DISPLACEMENT"] = north

    return df


def train_horizon(df, horizon):

    print("\n" + "=" * 60)
    print(f"FINAL RETRAINING: {horizon}-HOUR MODEL")
    print("=" * 60)

    # Use all development data, excluding the final test period.
    train = df[df["SEASON"] <= 2017].copy()

    train = train[
        train["HORIZON_HOURS"] == horizon
    ].copy()

    print(f"Training samples: {len(train)}")
    print(f"Unique cyclones: {train['SID'].nunique()}")

    if train.empty:
        raise ValueError(f"No development data for {horizon}h")

    X = train[FEATURES]
    y = train[TARGETS]

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    model = MultiOutputRegressor(
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            **HYPERPARAMETERS[horizon],
        ),
        n_jobs=1,
    )

    model.fit(X_imputed, y)

    artifact = {
        "model": model,
        "imputer": imputer,
        "features": FEATURES,
        "targets": TARGETS,
        "horizon": horizon,
        "hyperparameters": HYPERPARAMETERS[horizon],
        "training_period": "up to 2017",
    }

    output_path = MODEL_DIR / f"final_cyclone_model_{horizon}h.joblib"

    joblib.dump(artifact, output_path)

    print(f"Model saved: {output_path}")
    print("Training completed successfully.")

    return {
        "horizon": horizon,
        "samples": len(train),
        "cyclones": train["SID"].nunique(),
        "model_path": str(output_path),
    }


def main():

    print("Loading fixed-horizon dataset...")

    df = pd.read_csv(DATA_PATH)
    df = prepare_data(df)

    results = []

    for horizon in [6, 12, 24]:
        results.append(train_horizon(df, horizon))

    summary = pd.DataFrame(results)

    summary_path = (
        ROOT / "data" / "processed" / "final_training_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("ALL FINAL MODELS TRAINED")
    print("=" * 60)

    print(summary.to_string(index=False))
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
