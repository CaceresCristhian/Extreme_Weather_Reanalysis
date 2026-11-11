"""
Download authentic physical land-station climate records from NOAA NCEI GHCN-Daily.
Covers 8 contrasting climate stations:
- Phoenix, AZ (Desert / Extreme Heat)
- Las Vegas, NV (Global Heat Benchmark)
- Houston, TX (Subtropical Flood / Heavy Rain)
- Miami, FL (Tropical Storm / Maritime)
- Denver, CO (High Altitude / Cold Front)
- Minneapolis, MN (Polar Vortex / Sub-Zero Freeze)
- Seattle, WA (Atmospheric River / Persistent Rain)
- Chicago, IL (Great Lakes Freeze / Wind Extremes)
"""

from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np
import io

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "01_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

STATIONS = {
    "Phoenix_AZ": {"id": "USW00023183", "type": "Extreme Heat / Desert", "lat": 33.428, "lon": -112.004},
    "Las_Vegas_NV": {"id": "USW00023169", "type": "Global Heat Benchmark", "lat": 36.083, "lon": -115.153},
    "Houston_TX": {"id": "USW00012960", "type": "Subtropical Flood / Hurricane", "lat": 29.980, "lon": -95.360},
    "Miami_FL": {"id": "USW00012839", "type": "Tropical Storm / Humidity", "lat": 25.790, "lon": -80.316},
    "Denver_CO": {"id": "USW00003017", "type": "High Altitude / Cold Front", "lat": 39.847, "lon": -104.656},
    "Minneapolis_MN": {"id": "USW00014922", "type": "Sub-Zero Freeze Surge", "lat": 44.883, "lon": -93.229},
    "Seattle_WA": {"id": "USW00024233", "type": "Atmospheric River / Rain", "lat": 47.444, "lon": -122.314},
    "Chicago_IL": {"id": "USW00094846", "type": "Lake-Effect Polar Vortex", "lat": 41.960, "lon": -87.932},
}


def download_ghcn_stations():
    print("Downloading authentic NOAA GHCN-Daily station records from NOAA NCEI...")
    all_dfs = []
    
    for name, meta in STATIONS.items():
        stid = meta["id"]
        url = f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{stid}.csv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        print(f"Fetching {name} ({stid})...")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                df = pd.read_csv(io.BytesIO(resp.read()), low_memory=False)
                df["DATE"] = pd.to_datetime(df["DATE"])
                df = df[df["DATE"] >= "2020-01-01"].sort_values("DATE").reset_index(drop=True)
                
                n_rows = len(df)
                tmax = df["TMAX"].astype(float) / 10.0 if "TMAX" in df.columns else pd.Series([np.nan] * n_rows)
                tmin = df["TMIN"].astype(float) / 10.0 if "TMIN" in df.columns else pd.Series([np.nan] * n_rows)
                prcp = df["PRCP"].astype(float) / 10.0 if "PRCP" in df.columns else pd.Series([0.0] * n_rows)
                
                if "WSFG" in df.columns and df["WSFG"].notnull().sum() > n_rows * 0.3:
                    wind = df["WSFG"].astype(float) / 10.0
                elif "AWND" in df.columns:
                    wind = df["AWND"].astype(float) / 10.0
                else:
                    wind = pd.Series([0.0] * n_rows)
                    
                clean = pd.DataFrame({
                    "station": [name] * n_rows,
                    "station_id": [stid] * n_rows,
                    "date": df["DATE"].dt.strftime("%Y-%m-%d"),
                    "lat": [meta["lat"]] * n_rows,
                    "lon": [meta["lon"]] * n_rows,
                    "climate_type": [meta["type"]] * n_rows,
                    "tmax_c": tmax.ffill(),
                    "tmin_c": tmin.ffill(),
                    "prcp_mm": prcp.fillna(0.0),
                    "wind_gust_ms": wind.ffill().fillna(0.0)
                }).dropna(subset=["tmax_c", "tmin_c"]).reset_index(drop=True)
                
                all_dfs.append(clean)
                print(f"  {name}: {len(clean)} days from {clean['date'].min()} to {clean['date'].max()}")
        except Exception as e:
            print(f"  Error fetching {name}: {e}")

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = RAW_DIR / "noaa_ghcn_daily_stations_2020_2024.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved {len(combined)} authentic daily observations across {len(all_dfs)} stations to {out_path}")


if __name__ == "__main__":
    download_ghcn_stations()
