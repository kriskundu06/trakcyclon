
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)

RAW_FILE = RAW_DIR / "ibtracs_ni.csv"
OUTPUT_FILE = PROCESSED_DIR / "cyclones_clean.csv"

COLUMNS = [
    "SID",
    "SEASON",
    "BASIN",
    "SUBBASIN",
    "NAME",
    "ISO_TIME",
    "LAT",
    "LON",
    "WMO_WIND",
    "WMO_PRES",
]


def download_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists() and RAW_FILE.stat().st_size > 0:
        print("Raw dataset already exists. Skipping download.")
        return

    print("Downloading NOAA IBTrACS North Indian Ocean dataset...")

    response = requests.get(URL, timeout=180)
    response.raise_for_status()

    RAW_FILE.write_bytes(response.content)

    print(f"Downloaded: {RAW_FILE}")
    print(f"Size: {RAW_FILE.stat().st_size / 1024:.2f} KB")


def clean_data():
    print("\nLoading dataset...")

    # Row 1 contains units, not observations.
    df = pd.read_csv(
        RAW_FILE,
        skiprows=[1],
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    missing = set(COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[COLUMNS].copy()

    # Filter North Indian Ocean.
    df = df[df["BASIN"].astype(str).str.strip() == "NI"]

    # Convert numerical fields.
    for col in ["LAT", "LON", "WMO_WIND", "WMO_PRES"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse UTC timestamps.
    df["ISO_TIME"] = pd.to_datetime(
        df["ISO_TIME"],
        errors="coerce",
        utc=True
    )

    # Invalid coordinates and timestamps cannot be used.
    df = df.dropna(
        subset=["SID", "ISO_TIME", "LAT", "LON"]
    )

    df = df[
        df["LAT"].between(-90, 90)
        & df["LON"].between(-180, 180)
    ]

    # Remove duplicate storm observations.
    df = df.drop_duplicates(
        subset=["SID", "ISO_TIME"]
    )

    # Chronological ordering is essential.
    df = df.sort_values(
        ["SID", "ISO_TIME"]
    ).reset_index(drop=True)

    # Missing intensity values remain missing.
    df.to_csv(OUTPUT_FILE, index=False)

    track_lengths = df.groupby("SID").size()

    print("\n========== CLEANING REPORT ==========")
    print(f"Observations: {len(df):,}")
    print(f"Unique cyclones: {df['SID'].nunique():,}")
    print(f"Year range: {df['SEASON'].min()} - {df['SEASON'].max()}")
    print(f"Bay of Bengal records: {(df['SUBBASIN'] == 'BB').sum():,}")
    print(f"Arabian Sea records: {(df['SUBBASIN'] == 'AS').sum():,}")
    print(f"Storms with 2+ observations: {(track_lengths >= 2).sum():,}")
    print(f"Missing wind values: {df['WMO_WIND'].isna().sum():,}")
    print(f"Missing pressure values: {df['WMO_PRES'].isna().sum():,}")
    print(f"\nSaved cleaned dataset: {OUTPUT_FILE}")


if __name__ == "__main__":
    download_data()
    clean_data()
