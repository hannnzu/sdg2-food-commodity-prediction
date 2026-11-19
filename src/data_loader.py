"""Module for loading, cleaning, and chronologically splitting commodity price and weather data.
Developed for Week 2 (Minggu 2) - Data Understanding & Preparation.
PIC: Raihan Firdaus Alfaritsi
"""

from pathlib import Path
import pandas as pd
import numpy as np
import requests

# Default paths
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Constants
JAVA_PROVINCES = [
    "Banten",
    "DKI Jakarta",
    "Jawa Barat",
    "Jawa Tengah",
    "DI Yogyakarta",
    "Jawa Timur",
]

COORDINATES = {
    "Banten": (-6.1104, 106.1640),
    "DKI Jakarta": (-6.2088, 106.8456),
    "Jawa Barat": (-6.9175, 107.6191),
    "Jawa Tengah": (-6.9667, 110.4167),
    "DI Yogyakarta": (-7.7956, 110.3695),
    "Jawa Timur": (-7.2575, 112.7521),
}


def fetch_open_meteo_weather(
    start_date: str = "2022-01-01",
    end_date: str = "2026-02-12",
    save_raw: bool = True,
) -> pd.DataFrame:
    """Fetch daily mean temperature for the 6 Java provinces from Open-Meteo Archive API.
    
    If cached CSV exists in data/raw, loads from disk to avoid redundant API requests.
    """
    cache_path = RAW_DIR / "weather_openmeteo_jawa_2022_2026.csv"
    if cache_path.exists():
        df_cached = pd.read_csv(cache_path, parse_dates=["tanggal"])
        return df_cached

    lats = [COORDINATES[p][0] for p in JAVA_PROVINCES]
    lons = [COORDINATES[p][1] for p in JAVA_PROVINCES]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "Asia/Jakarta",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    dates = data[0]["daily"]["time"]
    weather_dict = {"tanggal": dates}
    for i, p in enumerate(JAVA_PROVINCES):
        col_name = f"suhu_{p.lower().replace(' ', '_')}"
        weather_dict[col_name] = data[i]["daily"]["temperature_2m_mean"]

    df_weather = pd.DataFrame(weather_dict)
    temp_cols = [c for c in df_weather.columns if c.startswith("suhu_")]
    df_weather["suhu_rata_rata_jawa"] = df_weather[temp_cols].mean(axis=1).round(2)
    df_weather["tanggal"] = pd.to_datetime(df_weather["tanggal"])

    if save_raw:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df_weather.to_csv(cache_path, index=False)

    return df_weather


def load_raw_commodity_price(
    commodity: str = "cabai_rawit",
) -> pd.DataFrame:
    """Load raw commodity price data from PIHPS."""
    file_name = f"final_komoditas_tradisional_{commodity}_2022_2026.csv"
    price_path = (
        RAW_DIR
        / "data_komoditas_pangan"
        / "pasar_tradisional"
        / "Cleaned_After_Imputation"
        / file_name
    )
    if not price_path.exists():
        raise FileNotFoundError(f"File {price_path} tidak ditemukan.")

    df = pd.read_csv(price_path)
    df["tanggal"] = pd.to_datetime(df["Date_Param"])
    df = df.sort_values("tanggal").reset_index(drop=True)
    return df


def prepare_clean_dataset(
    commodity: str = "cabai_rawit",
    reindex_daily: bool = True,
) -> pd.DataFrame:
    """Clean, aggregate 6 Java provinces, and merge with weather data.
    
    Uses strictly backward-looking forward-fill (anti-leakage).
    """
    df_price = load_raw_commodity_price(commodity)
    df_weather = fetch_open_meteo_weather()

    # Calculate average price across 6 Java provinces
    df_price["harga_jawa"] = df_price[JAVA_PROVINCES].mean(axis=1).round(2)

    if reindex_daily:
        min_date = df_price["tanggal"].min()
        max_date = df_price["tanggal"].max()
        full_index = pd.date_range(min_date, max_date, freq="D", name="tanggal")
        df_clean = df_price.set_index("tanggal").reindex(full_index)
        
        # Forward-fill missing non-trading days/weekends (anti-leakage)
        df_clean["harga"] = df_clean["harga_jawa"].ffill()
        for p in JAVA_PROVINCES:
            df_clean[p] = df_clean[p].ffill()
        df_clean = df_clean.reset_index()
    else:
        df_clean = df_price.copy()
        df_clean["harga"] = df_clean["harga_jawa"]

    # Merge weather
    df_clean = pd.merge(
        df_clean,
        df_weather[["tanggal", "suhu_rata_rata_jawa"]].rename(
            columns={"suhu_rata_rata_jawa": "suhu"}
        ),
        on="tanggal",
        how="left",
    )

    return df_clean


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    save_splits: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform strict chronological train/val/test splitting without shuffling (anti-leakage)."""
    n_total = len(df)
    n_train = int(round(train_ratio * n_total))
    n_val = int(round(val_ratio * n_total))
    n_test = n_total - n_train - n_val

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    if save_splits:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
        val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
        test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    return train_df, val_df, test_df

