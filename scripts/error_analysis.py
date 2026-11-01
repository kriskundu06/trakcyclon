
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "fixed_horizon_features.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed"

HORIZONS = [6, 12, 24]
RANDOM_STATE = 42

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

EARTH_RADIUS_KM = 6371.0


# ==========================================
# GEOGRAPHICAL FUNCTIONS
# ==========================================

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

    return (
        np.degrees(np.arctan2(x, y)) + 360
    ) % 360


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

    return (
        np.degrees(lat2),
        (np.degrees(lon2) + 540) % 360 - 180
    )


# ==========================================
# LOAD DATA AND RECREATE TEST SPLIT
# ==========================================

def load_test_data():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    df["ISO_TIME"] = pd.to_datetime(
        df["ISO_TIME"], utc=True
    )

    # Same time features used during model training
    month_angle = 2 * np.pi * df["ISO_TIME"].dt.month / 12
    hour_angle = 2 * np.pi * df["ISO_TIME"].dt.hour / 24

    df["MONTH_SIN"] = np.sin(month_angle)
    df["MONTH_COS"] = np.cos(month_angle)

    df["HOUR_SIN"] = np.sin(hour_angle)
    df["HOUR_COS"] = np.cos(hour_angle)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(df, groups=df["SID"])
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    assert set(train["SID"]).isdisjoint(set(test["SID"]))

    print("Training cyclones:", train["SID"].nunique())
    print("Testing cyclones:", test["SID"].nunique())
    print("Cyclone leakage check: PASSED")

    return test


# ==========================================
# GENERATE PREDICTIONS
# ==========================================

def generate_predictions(test):

    all_results = []

    for horizon in HORIZONS:

        print(f"\nEvaluating {horizon}-hour forecasts...")

        subset = test[
            test["HORIZON_HOURS"] == horizon
        ].copy()

        X = subset[FEATURES]

        # ------------------------------
        # CONSTANT VELOCITY
        # ------------------------------

        cv_distance = (
            subset["SPEED_KMH"] * subset["FUTURE_HOURS"]
        ).to_numpy()

        cv_lat, cv_lon = destination_point(
            subset["LAT"].to_numpy(),
            subset["LON"].to_numpy(),
            cv_distance,
            subset["HEADING_DEG"].to_numpy()
        )

        # ------------------------------
        # RANDOM FOREST
        # ------------------------------

        rf_path = MODEL_DIR / f"random_forest_{horizon}h.joblib"

        rf_model = joblib.load(rf_path)

        rf_pred = rf_model.predict(X)

        rf_distance = np.sqrt(
            rf_pred[:, 0] ** 2 + rf_pred[:, 1] ** 2
        )

        rf_bearing = (
            np.degrees(
                np.arctan2(rf_pred[:, 0], rf_pred[:, 1])
            ) + 360
        ) % 360

        rf_lat, rf_lon = destination_point(
            subset["LAT"].to_numpy(),
            subset["LON"].to_numpy(),
            rf_distance,
            rf_bearing
        )

        # ------------------------------
        # XGBOOST
        # ------------------------------

        xgb_path = MODEL_DIR / f"xgboost_{horizon}h.joblib"

        xgb_model = joblib.load(xgb_path)

        xgb_pred = xgb_model.predict(X)

        xgb_distance = np.sqrt(
            xgb_pred[:, 0] ** 2 + xgb_pred[:, 1] ** 2
        )

        xgb_bearing = (
            np.degrees(
                np.arctan2(xgb_pred[:, 0], xgb_pred[:, 1])
            ) + 360
        ) % 360

        xgb_lat, xgb_lon = destination_point(
            subset["LAT"].to_numpy(),
            subset["LON"].to_numpy(),
            xgb_distance,
            xgb_bearing
        )

        # ------------------------------
        # SAVE ROW-LEVEL ERRORS
        # ------------------------------

        result = pd.DataFrame({
            "SID": subset["SID"].values,
            "NAME": subset["NAME"].values,
            "SEASON": subset["SEASON"].values,
            "SUBBASIN": subset["SUBBASIN"].values,
            "ISO_TIME": subset["ISO_TIME"].values,
            "HORIZON_HOURS": horizon,
            "FUTURE_HOURS": subset["FUTURE_HOURS"].values,
            "WMO_WIND": subset["WMO_WIND"].values,
            "ACTUAL_LAT": subset["TARGET_LAT"].values,
            "ACTUAL_LON": subset["TARGET_LON"].values,

            "CV_PRED_LAT": cv_lat,
            "CV_PRED_LON": cv_lon,

            "RF_PRED_LAT": rf_lat,
            "RF_PRED_LON": rf_lon,

            "XGB_PRED_LAT": xgb_lat,
            "XGB_PRED_LON": xgb_lon
        })

        result["CV_ERROR_KM"] = haversine(
            result["CV_PRED_LAT"],
            result["CV_PRED_LON"],
            result["ACTUAL_LAT"],
            result["ACTUAL_LON"]
        )

        result["RF_ERROR_KM"] = haversine(
            result["RF_PRED_LAT"],
            result["RF_PRED_LON"],
            result["ACTUAL_LAT"],
            result["ACTUAL_LON"]
        )

        result["XGB_ERROR_KM"] = haversine(
            result["XGB_PRED_LAT"],
            result["XGB_PRED_LON"],
            result["ACTUAL_LAT"],
            result["ACTUAL_LON"]
        )

        all_results.append(result)

    return pd.concat(all_results, ignore_index=True)


# ==========================================
# MODEL PERFORMANCE SUMMARY
# ==========================================

def performance_summary(results):

    rows = []

    error_columns = {
        "Constant Velocity": "CV_ERROR_KM",
        "Random Forest": "RF_ERROR_KM",
        "XGBoost": "XGB_ERROR_KM"
    }

    for horizon in HORIZONS:

        subset = results[
            results["HORIZON_HOURS"] == horizon
        ]

        for model_name, error_col in error_columns.items():

            errors = subset[error_col]

            rows.append({
                "HORIZON_HOURS": horizon,
                "MODEL": model_name,
                "SAMPLES": len(errors),
                "MEAN_ERROR_KM": errors.mean(),
                "MEDIAN_ERROR_KM": errors.median(),
                "RMSE_KM": np.sqrt(np.mean(errors ** 2)),
                "P90_ERROR_KM": errors.quantile(0.90)
            })

    summary = pd.DataFrame(rows)

    print("\n========== MODEL PERFORMANCE ==========")
    print(summary.round(2).to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / "error_analysis_summary.csv",
        index=False
    )

    return summary


# ==========================================
# WORST PREDICTIONS
# ==========================================

def analyze_worst_predictions(results):

    print("\n========== WORST XGBOOST PREDICTIONS ==========")

    for horizon in HORIZONS:

        subset = results[
            results["HORIZON_HOURS"] == horizon
        ]

        worst = subset.nlargest(
            10, "XGB_ERROR_KM"
        )

        print(f"\nWorst 10 predictions: {horizon} hours")

        print(
            worst[
                [
                    "SID",
                    "NAME",
                    "ISO_TIME",
                    "SUBBASIN",
                    "XGB_ERROR_KM",
                    "CV_ERROR_KM",
                    "RF_ERROR_KM"
                ]
            ].round(2).to_string(index=False)
        )

    worst_all = results.nlargest(
        50, "XGB_ERROR_KM"
    )

    worst_all.to_csv(
        OUTPUT_DIR / "worst_predictions.csv",
        index=False
    )


# ==========================================
# REGIONAL ANALYSIS
# ==========================================

def analyze_regions(results):

    print("\n========== REGIONAL PERFORMANCE ==========")

    region_summary = (
        results.groupby(
            ["HORIZON_HOURS", "SUBBASIN"],
            dropna=False
        )
        .agg(
            SAMPLES=("XGB_ERROR_KM", "size"),
            CV_MEAN=("CV_ERROR_KM", "mean"),
            RF_MEAN=("RF_ERROR_KM", "mean"),
            XGB_MEAN=("XGB_ERROR_KM", "mean")
        )
        .reset_index()
    )

    print(region_summary.round(2).to_string(index=False))

    region_summary.to_csv(
        OUTPUT_DIR / "regional_error_analysis.csv",
        index=False
    )


# ==========================================
# INTENSITY ANALYSIS
# ==========================================

def analyze_intensity(results):

    print("\n========== CYCLONE INTENSITY ANALYSIS ==========")

    # WMO_WIND is conventionally recorded in knots.
    # These categories are approximate intensity bands.
    intensity = results.dropna(
        subset=["WMO_WIND"]
    ).copy()

    intensity["INTENSITY"] = pd.cut(
        intensity["WMO_WIND"],
        bins=[-np.inf, 33, 63, 89, 119, np.inf],
        labels=[
            "Below 34 kt",
            "34-63 kt",
            "64-89 kt",
            "90-119 kt",
            "120+ kt"
        ]
    )

    summary = (
        intensity.groupby(
            ["HORIZON_HOURS", "INTENSITY"],
            observed=False
        )
        .agg(
            SAMPLES=("XGB_ERROR_KM", "size"),
            CV_MEAN=("CV_ERROR_KM", "mean"),
            RF_MEAN=("RF_ERROR_KM", "mean"),
            XGB_MEAN=("XGB_ERROR_KM", "mean")
        )
        .reset_index()
    )

    print("Rows with available wind measurements:", len(intensity))
    print(summary.round(2).to_string(index=False))

    summary.to_csv(
        OUTPUT_DIR / "intensity_error_analysis.csv",
        index=False
    )


# ==========================================
# XGBOOST FEATURE IMPORTANCE
# ==========================================

def analyze_feature_importance():

    print("\n========== XGBOOST FEATURE IMPORTANCE ==========")

    for horizon in HORIZONS:

        model_path = MODEL_DIR / f"xgboost_{horizon}h.joblib"

        pipeline = joblib.load(model_path)

        # MultiOutputRegressor has one estimator per target:
        # eastward movement and northward movement.
        multi_model = pipeline.named_steps["model"]

        east_importance = (
            multi_model.estimators_[0].feature_importances_
        )

        north_importance = (
            multi_model.estimators_[1].feature_importances_
        )

        average_importance = (
            east_importance + north_importance
        ) / 2

        importance = pd.DataFrame({
            "FEATURE": FEATURES,
            "IMPORTANCE": average_importance
        }).sort_values(
            "IMPORTANCE", ascending=False
        )

        print(f"\n{horizon}-hour model")
        print(importance.round(4).to_string(index=False))

        importance.to_csv(
            OUTPUT_DIR / f"xgboost_feature_importance_{horizon}h.csv",
            index=False
        )


# ==========================================
# MAIN
# ==========================================

def main():

    test = load_test_data()

    results = generate_predictions(test)

    results.to_csv(
        OUTPUT_DIR / "all_model_predictions.csv",
        index=False
    )

    performance_summary(results)

    analyze_worst_predictions(results)

    analyze_regions(results)

    analyze_intensity(results)

    analyze_feature_importance()

    print("\n========== ERROR ANALYSIS COMPLETE ==========")

    print("Generated files:")
    print("- error_analysis_summary.csv")
    print("- all_model_predictions.csv")
    print("- worst_predictions.csv")
    print("- regional_error_analysis.csv")
    print("- intensity_error_analysis.csv")
    print("- xgboost_feature_importance_6h.csv")
    print("- xgboost_feature_importance_12h.csv")
    print("- xgboost_feature_importance_24h.csv")


if __name__ == "__main__":
    main()
