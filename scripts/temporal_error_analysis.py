from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "fixed_horizon_features.csv"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "temporal_predictions.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed"

HORIZONS = [6, 12, 24]


def metrics(group):
    errors = group["ERROR_KM"].dropna().to_numpy(dtype=float)
    if len(errors) == 0:
        return pd.Series({
            "samples": 0,
            "mean_error_km": np.nan,
            "median_error_km": np.nan,
            "rmse_km": np.nan,
            "p90_error_km": np.nan,
        })

    return pd.Series({
        "samples": len(errors),
        "mean_error_km": errors.mean(),
        "median_error_km": np.median(errors),
        "rmse_km": np.sqrt(np.mean(errors ** 2)),
        "p90_error_km": np.percentile(errors, 90),
    })


def load_and_attach_metadata():
    print("Loading temporal predictions and original features...")
    predictions = pd.read_csv(PREDICTIONS_PATH)
    features = pd.read_csv(DATA_PATH)

    required_prediction_columns = {
        "SID", "SEASON", "HORIZON_HOURS", "PARTITION",
        "MODEL", "ACTUAL_LAT", "ACTUAL_LON", "ERROR_KM",
    }
    missing = required_prediction_columns - set(predictions.columns)
    if missing:
        raise ValueError(f"Temporal predictions missing columns: {sorted(missing)}")

    required_feature_columns = {
        "SID", "SEASON", "HORIZON_HOURS", "TARGET_LAT", "TARGET_LON",
        "SUBBASIN", "WMO_WIND", "NAME", "LAT", "LON", "ISO_TIME",
    }
    missing = required_feature_columns - set(features.columns)
    if missing:
        raise ValueError(f"Fixed-horizon features missing columns: {sorted(missing)}")

    # Reconstruct the test rows in exactly the same source-file order used
    # by temporal_validation.py. Its CSV filtering preserves row order.
    features["SEASON"] = pd.to_numeric(features["SEASON"], errors="coerce")
    features["HORIZON_HOURS"] = pd.to_numeric(
        features["HORIZON_HOURS"], errors="coerce"
    )
    features["ISO_TIME"] = pd.to_datetime(
        features["ISO_TIME"], utc=True, errors="coerce"
    )

    test_features = features[
        features["SEASON"] > 2017
    ].copy()

    # Each model's prediction rows were generated from the same test_h
    # dataframe order. Attach metadata positionally, after strict checks.
    pieces = []
    for horizon in HORIZONS:
        horizon_features = test_features[
            test_features["HORIZON_HOURS"] == horizon
        ].reset_index(drop=True)

        for model_name in ["xgboost", "constant_velocity"]:
            pred = predictions[
                (predictions["PARTITION"] == "test")
                & (predictions["HORIZON_HOURS"] == horizon)
                & (predictions["MODEL"] == model_name)
            ].copy().reset_index(drop=True)

            if len(pred) != len(horizon_features):
                raise ValueError(
                    f"Cannot safely align {horizon}h {model_name}: "
                    f"{len(pred)} predictions vs {len(horizon_features)} source rows."
                )

            # Verify the row ordering using storm IDs, season and target coordinates.
            checks = (
                pred["SID"].astype(str).to_numpy()
                == horizon_features["SID"].astype(str).to_numpy()
            )
            checks &= np.isclose(
                pred["ACTUAL_LAT"].to_numpy(dtype=float),
                horizon_features["TARGET_LAT"].to_numpy(dtype=float),
                atol=1e-6,
            )
            checks &= np.isclose(
                pred["ACTUAL_LON"].to_numpy(dtype=float),
                horizon_features["TARGET_LON"].to_numpy(dtype=float),
                atol=1e-6,
            )
            if not checks.all():
                raise ValueError(
                    f"Prediction/source row alignment check failed for "
                    f"{horizon}h {model_name}. Stop: metadata must not be guessed."
                )

            metadata_columns = [
                "SUBBASIN", "WMO_WIND", "NAME", "LAT", "LON", "ISO_TIME"
            ]
            aligned = pd.concat(
                [pred, horizon_features[metadata_columns].reset_index(drop=True)],
                axis=1,
            )
            pieces.append(aligned)

    result = pd.concat(pieces, ignore_index=True)
    result["WMO_WIND"] = pd.to_numeric(result["WMO_WIND"], errors="coerce")
    result["SUBBASIN"] = result["SUBBASIN"].fillna("Unknown").astype(str)
    return result


def grouped_metrics(df, group_columns):
    rows = []
    for keys, group in df.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(metrics(group).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def save_metric_table(df, group_columns, output_name):
    table = grouped_metrics(df, group_columns)
    table.to_csv(OUTPUT_DIR / output_name, index=False)
    return table


def analyze_worst_cyclones(df):
    xgb = df[df["MODEL"] == "xgboost"].copy()

    # Average error for each cyclone and horizon avoids ranking storms
    # simply because they have more observations.
    result = (
        xgb.groupby(["SID", "NAME", "SEASON", "HORIZON_HOURS"])
        .agg(
            samples=("ERROR_KM", "size"),
            mean_error_km=("ERROR_KM", "mean"),
            median_error_km=("ERROR_KM", "median"),
            max_error_km=("ERROR_KM", "max"),
            subbasin=("SUBBASIN", "first"),
        )
        .reset_index()
        .sort_values(
            ["HORIZON_HOURS", "mean_error_km"],
            ascending=[True, False],
        )
    )

    result.to_csv(OUTPUT_DIR / "temporal_worst_cyclones.csv", index=False)
    print("\n========== WORST CYCLONES BY HORIZON ==========")
    for horizon in HORIZONS:
        print(f"\n{horizon}h:")
        print(
            result[result["HORIZON_HOURS"] == horizon]
            .head(10)
            .to_string(index=False)
        )


def analyze_error_distribution(df):
    xgb = df[df["MODEL"] == "xgboost"].copy()
    rows = []

    for horizon, group in xgb.groupby("HORIZON_HOURS"):
        errors = group["ERROR_KM"]
        rows.append({
            "horizon": horizon,
            "samples": len(group),
            "within_50km_pct": (errors <= 50).mean() * 100,
            "within_100km_pct": (errors <= 100).mean() * 100,
            "within_200km_pct": (errors <= 200).mean() * 100,
            "within_500km_pct": (errors <= 500).mean() * 100,
            "over_500km_pct": (errors > 500).mean() * 100,
            "maximum_error_km": errors.max(),
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "temporal_error_distribution.csv", index=False)
    print("\n========== XGBOOST ERROR DISTRIBUTION ==========")
    print(result.round(2).to_string(index=False))


def analyze_feature_importance():
    rows = []

    for horizon in HORIZONS:
        model_path = MODEL_DIR / f"temporal_best_{horizon}h.joblib"
        if not model_path.exists():
            print(f"Model not found, skipping importance: {model_path}")
            continue

        pipeline = joblib.load(model_path)
        estimator = pipeline.named_steps["model"]

        if hasattr(estimator, "estimators_"):
            importances = np.mean(
                [tree.feature_importances_ for tree in estimator.estimators_],
                axis=0,
            )
        elif hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        else:
            print(f"Cannot extract feature importance from {model_path}")
            continue

        feature_names = pipeline.feature_names_in_
        for name, importance in zip(feature_names, importances):
            rows.append({
                "horizon": horizon,
                "feature": name,
                "importance": float(importance),
            })

    if rows:
        result = pd.DataFrame(rows).sort_values(
            ["horizon", "importance"], ascending=[True, False]
        )
        result.to_csv(
            OUTPUT_DIR / "temporal_feature_importance.csv", index=False
        )
        print("\n========== XGBOOST FEATURE IMPORTANCE ==========")
        for horizon in HORIZONS:
            print(f"\n{horizon}h:")
            print(
                result[result["horizon"] == horizon]
                .head(10)
                .to_string(index=False)
            )


def main():
    df = load_and_attach_metadata()

    print("\n========== TEMPORAL TEST ERROR ANALYSIS ==========")
    print("Test prediction rows:", len(df))
    print("Unique cyclones:", df["SID"].nunique())
    print("Models:", sorted(df["MODEL"].unique()))

    # Overall comparison of selected XGBoost against Constant Velocity.
    overall = grouped_metrics(df, ["HORIZON_HOURS", "MODEL"])
    overall.to_csv(
        OUTPUT_DIR / "temporal_error_summary.csv", index=False
    )
    print("\n========== OVERALL MODEL COMPARISON ==========")
    print(overall.round(2).to_string(index=False))

    # Regional performance. IBTrACS subbasin codes are retained as supplied.
    regional = save_metric_table(
        df,
        ["HORIZON_HOURS", "MODEL", "SUBBASIN"],
        "temporal_regional_errors.csv",
    )
    print("\n========== REGIONAL ERROR ANALYSIS ==========")
    print(regional.round(2).to_string(index=False))

    # Wind speeds in IBTrACS are recorded in knots. These bins are descriptive,
    # not official North Indian Ocean intensity classifications.
    wind_df = df[df["WMO_WIND"].notna()].copy()
    wind_df["WIND_BIN_KT"] = pd.cut(
        wind_df["WMO_WIND"],
        bins=[-np.inf, 34, 64, 96, np.inf],
        labels=["<34 kt", "34–<64 kt", "64–<96 kt", ">=96 kt"],
        right=False,
    )
    intensity = save_metric_table(
        wind_df,
        ["HORIZON_HOURS", "MODEL", "WIND_BIN_KT"],
        "temporal_intensity_errors.csv",
    )
    print("\n========== ERROR BY RECORDED WIND SPEED ==========")
    print(intensity.round(2).to_string(index=False))
    print(
        f"\nWind analysis includes {len(wind_df):,} prediction rows "
        f"with recorded WMO_WIND; missing wind values are excluded."
    )

    analyze_worst_cyclones(df)
    analyze_error_distribution(df)
    analyze_feature_importance()

    print("\n========== ANALYSIS COMPLETE ==========")
    print("Saved outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
