
from datetime import datetime, timezone
from pathlib import Path
import os
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATASET_PATH = ROOT / "data" / "processed" / "cyclones_clean.csv"
HORIZONS = [6, 12, 24]
EARTH_RADIUS_KM = 6371.0

MODEL_PATHS = {
    h: MODEL_DIR / f"final_cyclone_model_{h}h.joblib"
    for h in HORIZONS
}

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


# =====================================================
# FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="CycloneTrack AI API",
    description=(
        "North Indian Ocean cyclone position forecasts "
        "at 6, 12 and 24 hours."
    ),
    version="1.0.0",
)


# =====================================================
# CORS CONFIGURATION
# =====================================================

origins = [
    item.strip()
    for item in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://localhost:5174,http://localhost:3000",
    ).split(",")
    if item.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# REQUEST / RESPONSE SCHEMAS
# =====================================================

class CycloneObservation(BaseModel):

    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value):

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "Timestamp must include a timezone, "
                "e.g. 2026-10-31T06:00:00Z."
            )

        return value.astimezone(timezone.utc)


class PredictionRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    previous: CycloneObservation
    current: CycloneObservation


class ForecastPoint(BaseModel):

    horizon_hours: int
    latitude: float
    longitude: float


class PredictionResponse(BaseModel):

    model_name: str
    input_interval_hours: float
    movement_speed_kmh: float
    movement_heading_degrees: float
    forecasts: list[ForecastPoint]


# =====================================================
# LOAD TRAINED MODELS
# =====================================================

models = {}


def load_models():

    loaded = {}

    for horizon, path in MODEL_PATHS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing trained model: {path}"
            )

        artifact = joblib.load(path)

        required_keys = {
            "model",
            "imputer",
            "features",
        }

        if (
            not isinstance(artifact, dict)
            or not required_keys.issubset(artifact)
        ):
            raise ValueError(
                f"Unexpected model artifact format in {path}. "
                "Expected model, imputer and features."
            )

        if artifact["features"] != FEATURES:
            raise ValueError(
                f"Feature order mismatch in {path}"
            )

        loaded[horizon] = artifact

    return loaded


@app.on_event("startup")
def startup_event():

    global models

    try:

        models = load_models()

        print(
            f"Loaded XGBoost models for {HORIZONS} hours."
        )

    except Exception as exc:

        models = {}

        print(f"Model loading failed: {exc}")


# =====================================================
# GEOGRAPHICAL UTILITIES
# =====================================================

def haversine(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = np.radians(
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return float(
        2 * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )


def bearing(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = np.radians(
        [lat1, lon1, lat2, lon2]
    )

    dlon = lon2 - lon1

    y = np.sin(dlon) * np.cos(lat2)

    x = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    return float(
        (np.degrees(np.arctan2(y, x)) + 360) % 360
    )


def destination_point(
    lat,
    lon,
    distance_km,
    bearing_degrees
):

    lat1 = np.radians(lat)
    lon1 = np.radians(lon)

    theta = np.radians(bearing_degrees)

    angular = distance_km / EARTH_RADIUS_KM

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular)
        + np.cos(lat1)
        * np.sin(angular)
        * np.cos(theta)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(theta) * np.sin(angular) * np.cos(lat1),
        np.cos(angular) - np.sin(lat1) * np.sin(lat2),
    )

    return (
        float(np.degrees(lat2)),
        float((np.degrees(lon2) + 540) % 360 - 180),
    )


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def build_features(previous, current):

    delta_hours = (
        current.timestamp - previous.timestamp
    ).total_seconds() / 3600

    if delta_hours <= 0:

        raise HTTPException(
            422,
            "Current observation must be later than previous observation."
        )

    if delta_hours > 12:

        raise HTTPException(
            422,
            "Observations must be no more than 12 hours apart."
        )

    distance = haversine(
        previous.latitude,
        previous.longitude,
        current.latitude,
        current.longitude,
    )

    speed = distance / delta_hours

    if speed > 150:

        raise HTTPException(
            422,
            "Calculated speed exceeds 150 km/h. "
            "Check coordinates and timestamps."
        )

    heading = bearing(
        previous.latitude,
        previous.longitude,
        current.latitude,
        current.longitude,
    )

    heading_rad = np.radians(heading)

    timestamp = current.timestamp.astimezone(timezone.utc)

    month_angle = 2 * np.pi * timestamp.month / 12
    hour_angle = 2 * np.pi * timestamp.hour / 24

    row = {

        "LAT": current.latitude,
        "LON": current.longitude,

        "PREV_LAT": previous.latitude,
        "PREV_LON": previous.longitude,

        "DELTA_HOURS": delta_hours,

        "DISTANCE_KM": distance,
        "SPEED_KMH": speed,

        "HEADING_SIN": np.sin(heading_rad),
        "HEADING_COS": np.cos(heading_rad),

        "MONTH_SIN": np.sin(month_angle),
        "MONTH_COS": np.cos(month_angle),

        "HOUR_SIN": np.sin(hour_angle),
        "HOUR_COS": np.cos(hour_angle),
    }

    return (
        pd.DataFrame([row], columns=FEATURES),
        delta_hours,
        speed,
        heading,
    )


# =====================================================
# API ENDPOINTS
# =====================================================

@app.get("/")
def root():

    return {
        "message": "CycloneTrack AI API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():

    loaded = sorted(models.keys())

    return {

        "status": (
            "healthy"
            if len(loaded) == len(HORIZONS)
            else "degraded"
        ),

        "models_loaded": loaded,

        "required_horizons": HORIZONS,
    }

# =====================================================
# HISTORICAL DATASET STATISTICS
# =====================================================

@lru_cache(maxsize=1)
def load_dataset_stats():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned cyclone dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = {
        "SID",
        "SUBBASIN",
        "ISO_TIME",
        "WMO_WIND",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    # Total historical observations and unique cyclones
    total_observations = len(df)
    total_cyclones = int(df["SID"].nunique())

    # Count monitored regions
    region_names = {
        "AS": "Arabian Sea",
        "BB": "Bay of Bengal",
    }

    region_stats = []

    for code, name in region_names.items():

        region_df = df[df["SUBBASIN"] == code]

        region_stats.append({
            "code": code,
            "region": name,
            "observations": int(len(region_df)),
            "cyclones": int(region_df["SID"].nunique()),
        })

    # Maximum recorded wind speed.
    # IBTrACS WMO_WIND is in knots; convert to km/h.
    wind = pd.to_numeric(
        df["WMO_WIND"],
        errors="coerce"
    )

    max_wind_knots = wind.max()

    max_wind_kmh = (
        round(float(max_wind_knots) * 1.852, 2)
        if pd.notna(max_wind_knots)
        else None
    )

    # Dataset coverage
    dates = pd.to_datetime(
        df["ISO_TIME"],
        errors="coerce",
        utc=True
    )

    valid_dates = dates.dropna()

    dataset_start = (
        valid_dates.min().strftime("%Y-%m-%d")
        if not valid_dates.empty
        else None
    )

    dataset_end = (
        valid_dates.max().strftime("%Y-%m-%d")
        if not valid_dates.empty
        else None
    )

    return {
        "total_observations": total_observations,
        "total_cyclones": total_cyclones,
        "monitored_regions": len(region_stats),
        "max_wind_kmh": max_wind_kmh,
        "dataset_start": dataset_start,
        "dataset_end": dataset_end,
        "regions": region_stats,
    }

@app.get("/dataset-stats")
def dataset_stats():

    try:
        return load_dataset_stats()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load dataset statistics: {exc}"
        )




# =====================================================
# HISTORICAL CYCLONE RECORDS
# =====================================================

@lru_cache(maxsize=1)
def load_historical_cyclones():
    """
    Build one archive record per cyclone SID from the cleaned IBTrACS dataset.
    The result is cached because the source CSV is unchanged between requests.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned cyclone dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"SID", "ISO_TIME", "SUBBASIN", "WMO_WIND"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    if "NAME" not in df.columns:
        df["NAME"] = "UNNAMED"

    df = df.copy()
    df["ISO_TIME"] = pd.to_datetime(
        df["ISO_TIME"], errors="coerce", utc=True
    )
    df["WMO_WIND"] = pd.to_numeric(
        df["WMO_WIND"], errors="coerce"
    )
    df = df.dropna(subset=["SID", "ISO_TIME"])

    region_names = {
        "AS": "Arabian Sea",
        "BB": "Bay of Bengal",
    }

    records = []

    for sid, group in df.groupby("SID", sort=False):
        group = group.sort_values("ISO_TIME")
        latest = group.iloc[-1]

        # Use the sub-basin with the most observations as the cyclone's
        # representative region. Cyclones can cross between sub-basins.
        subbasin_counts = group["SUBBASIN"].dropna().value_counts()
        subbasin_code = (
            str(subbasin_counts.index[0]) if not subbasin_counts.empty else ""
        )
        region = region_names.get(subbasin_code, "Other / Unclassified")

        peak_wind_knots = group["WMO_WIND"].max()
        peak_wind_kmh = (
            round(float(peak_wind_knots) * 1.852, 1)
            if pd.notna(peak_wind_knots)
            else None
        )

        name_values = (
            group["NAME"].dropna().astype(str).str.strip()
            if "NAME" in group.columns
            else pd.Series(dtype=str)
        )
        name_values = name_values[
            (name_values != "") & (name_values.str.upper() != "UNNAMED")
        ]
        cyclone_name = (
            name_values.iloc[0].title()
            if not name_values.empty
            else "Unnamed Cyclone"
        )

        records.append({
            "id": str(sid),
            "name": cyclone_name,
            "region": region,
            "region_code": subbasin_code,
            "start_date": group["ISO_TIME"].min().strftime("%Y-%m-%d"),
            "end_date": latest["ISO_TIME"].strftime("%Y-%m-%d"),
            "peak_wind_kmh": peak_wind_kmh,
            "observations": int(len(group)),
        })

    records.sort(key=lambda item: item["end_date"], reverse=True)
    return records


@app.get("/history")
def cyclone_history(
    search: str = "",
    region: str = "all",
    limit: int = 50,
    offset: int = 0,
):
    """
    Return paginated historical cyclone records.
    Supported region values: all, AS, BB.
    """
    if limit < 1 or limit > 200:
        raise HTTPException(
            status_code=422,
            detail="limit must be between 1 and 200.",
        )
    if offset < 0:
        raise HTTPException(
            status_code=422,
            detail="offset cannot be negative.",
        )

    try:
        records = load_historical_cyclones()

        normalized_region = region.strip().lower()
        if normalized_region not in {"all", "as", "bb"}:
            raise HTTPException(
                status_code=422,
                detail="region must be one of: all, AS, BB.",
            )

        if normalized_region != "all":
            records = [
                item for item in records
                if item["region_code"].lower() == normalized_region
            ]

        normalized_search = search.strip().lower()
        if normalized_search:
            records = [
                item for item in records
                if normalized_search in item["name"].lower()
                or normalized_search in item["id"].lower()
            ]

        total = len(records)
        page_records = records[offset:offset + limit]

        all_records = load_historical_cyclones()
        region_totals = {
            "bay_of_bengal": sum(
                1 for item in all_records if item["region_code"] == "BB"
            ),
            "arabian_sea": sum(
                1 for item in all_records if item["region_code"] == "AS"
            ),
        }

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "region_totals": region_totals,
            "records": page_records,
        }

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load historical cyclone records: {exc}",
        )


@app.get("/history/filters")
def cyclone_history_filters():
    """Return region filter options available in the historical archive."""
    return {
        "regions": [
            {"code": "all", "name": "All Regions"},
            {"code": "BB", "name": "Bay of Bengal"},
            {"code": "AS", "name": "Arabian Sea"},
        ]
    }



@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    if len(models) != len(HORIZONS):

        raise HTTPException(
            503,
            "Final models are not loaded. "
            "Check the three final model files."
        )

    features, delta_hours, speed, heading = build_features(
        request.previous,
        request.current,
    )

    forecasts = []

    for horizon in HORIZONS:

        artifact = models[horizon]

        # Apply the saved imputer.
        model_input = artifact["imputer"].transform(
            features[artifact["features"]]
        )

        # Predict eastward and northward displacement.
        displacement = artifact["model"].predict(
            model_input
        )

        east = float(displacement[0][0])
        north = float(displacement[0][1])

        # Convert displacement into distance and direction.
        distance = float(
            np.hypot(east, north)
        )

        direction = float(
            (
                np.degrees(np.arctan2(east, north))
                + 360
            ) % 360
        )

        # Calculate forecast coordinates.
        lat, lon = destination_point(
            request.current.latitude,
            request.current.longitude,
            distance,
            direction,
        )

        forecasts.append(
            ForecastPoint(
                horizon_hours=horizon,
                latitude=round(lat, 5),
                longitude=round(lon, 5),
            )
        )

    return PredictionResponse(

        model_name="XGBoost",

        input_interval_hours=round(
            delta_hours, 3
        ),

        movement_speed_kmh=round(
            speed, 2
        ),

        movement_heading_degrees=round(
            heading, 2
        ),

        forecasts=forecasts,
    )
