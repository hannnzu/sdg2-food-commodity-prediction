"""Feature engineering module for commodity price time series forecasting.
Developed for Week 2 (Minggu 2) - Data Preparation.
PIC: Muhammad Raufa Hafid Widodo & Raihan Firdaus Alfaritsi
"""

import numpy as np
import pandas as pd

# Kalender hari raya (2018–2029)
IDUL_FITRI = ["2022-05-02", "2023-04-22", "2024-04-10", "2025-03-31", "2026-03-20"]
_years = range(2018, 2030)
_fixed = [f"{y}-01-01" for y in _years] + [f"{y}-12-25" for y in _years]
HARI_RAYA = pd.DatetimeIndex(sorted(pd.to_datetime(IDUL_FITRI + _fixed)))


def dekat_hari_raya(dates, tol: int = 7) -> np.ndarray:
    """Mengembalikan 1 jika tanggal berjarak <= tol hari dari hari raya terdekat.
    
    Kalender hari raya diketahui sebelumnya, sehingga tidak menimbulkan data leakage.
    """
    d = pd.DatetimeIndex(dates)
    diff = np.abs(
        (d.values[:, None] - HARI_RAYA.values[None, :])
        .astype("timedelta64[D]")
        .astype(int)
    )
    return (diff.min(axis=1) <= tol).astype(int)


def build_supervised(
    series_df: pd.DataFrame,
    date_col: str = "tanggal",
    price_col: str = "harga",
    lags: list[int] = [1, 2, 7],
    roll_windows: list[int] = [7, 14],
    horizon: int = 1,
    extra_cols: list[str] = None,
) -> pd.DataFrame:
    """Membangun fitur tabular berurutan waktu (anti-leakage) untuk deret harga harian.
    
    - Fitur lag murni bergeser ke belakang (shift >= 1)
    - Fitur rolling dihitung hanya dari masa lalu hingga t
    - Target y adalah harga pada t + horizon
    """
    df = series_df.copy().reset_index(drop=True)
    p = df[price_col]
    d = pd.to_datetime(df[date_col])

    out = pd.DataFrame({date_col: d, "harga_t": p})

    # Fitur Lag
    for k in lags:
        out[f"lag_{k}"] = p.shift(k)

    # Fitur Rolling Statistics (hanya memakai t dan sebelumnya, shift dilakukan bila diperlukan)
    for w in roll_windows:
        out[f"roll_mean_{w}"] = p.rolling(w).mean()
        out[f"roll_std_{w}"] = p.rolling(w).std()

    # Fitur Selisih Harian
    out["selisih_1"] = p.diff(1)

    # Fitur Kalender Siklikal
    out["dow"] = d.dt.dayofweek
    out["month"] = d.dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dekat_hari_raya"] = dekat_hari_raya(d)

    # Fitur Eksogen Ekstra (misal: suhu) jika ada
    if extra_cols:
        for c in extra_cols:
            if c in df.columns and c not in out.columns:
                out[c] = df[c]

    # Target Satu Langkah ke Depan (Horizon)
    out["target_date"] = d.shift(-horizon)
    out["y"] = p.shift(-horizon)

    return out


def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    window: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Mengubah deret 2D menjadi tensor 3D (n_samples, window, n_features) untuk LSTM."""
    X_win, y_win = [], []
    for i in range(window, len(X)):
        X_win.append(X[i - window : i])
        y_win.append(y[i])
    return np.array(X_win), np.array(y_win)

