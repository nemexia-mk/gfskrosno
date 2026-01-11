#!/usr/bin/env python3
# icon_final_fixed_time_snow.py
import os
import shutil
import requests
import pandas as pd
import xarray as xr
import bz2
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from ftplib import FTP
from dotenv import load_dotenv

# -----------------------
# CONFIG
# -----------------------
OUTPUT_DIR = "icon_krosno_full"
TEMP_DIR = "temp_grib_icon_fix"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

KROSNO_LAT = 49.69
KROSNO_LON = 21.77

# Pełna lista parametrów
PARAMS = [
    "t_2m", "td_2m", "pmsl", "tot_prec", "h_snow",
    "clct", "clcl", "clcm", "clch", "u_10m", "v_10m", "vmax_10m",
    "cape_ml", "cin_ml", "vis"
]

# Mapa nazw zmiennych z GRIB -> nazwa w DataFrame
RENAME_MAP = {
    "t2m": "t_2m", "2t": "t_2m",
    "d2m": "td_2m", "2d": "td_2m",
    "prmsl": "pmsl", "pmsl": "pmsl",
    "tp": "tot_prec",
    
    # Kluczowe dla śniegu
    "sde": "h_snow", "sd": "h_snow", "h_snow": "h_snow", "snow_depth": "h_snow",
    
    "clct": "clct", "clcl": "clcl", "clcm": "clcm", "clch": "clch",
    "u10": "u_10m", "10u": "u_10m",
    "v10": "v_10m", "10v": "v_10m",
    "fg10": "vmax_10m",
    "cape_ml": "cape_ml", "cin_ml": "cin_ml",
    "vis": "vis"
}

# -----------------------
# FUNKCJE POMOCNICZE
# -----------------------
def get_latest_run():
    now_utc = datetime.utcnow()
    current_time = now_utc.time()
    cutoffs = [dt_time(3, 45), dt_time(9, 45), dt_time(15, 45), dt_time(21, 45)]

    if current_time < cutoffs[0]:
        selected_run = ("18", now_utc - timedelta(days=1))
    elif current_time < cutoffs[1]:
        selected_run = ("00", now_utc)
    elif current_time < cutoffs[2]:
        selected_run = ("06", now_utc)
    elif current_time < cutoffs[3]:
        selected_run = ("12", now_utc)
    else:
        selected_run = ("18", now_utc)
    
    return selected_run[1].strftime("%Y%m%d"), selected_run[0]

def clean_temp():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

def upload_ftp(files):
    load_dotenv()
    host = os.getenv("FTP_HOST")
    user = os.getenv("FTP_USER")
    passwd = os.getenv("FTP_PASS")

    if not all([host, user, passwd]):
        print("⚠️ Brak konfiguracji FTP w pliku .env")
        return

    try:
        print("📡 Łączenie z FTP...")
        ftp = FTP(host, user, passwd, timeout=60)
        ftp.cwd("/stacja.meteo-krosno.pl/")

        for fpath in files:
            if not os.path.exists(fpath):
                continue
            fname = os.path.basename(fpath)
            with open(fpath, "rb") as f:
                if "icon-tab.csv" in fname:
                    ftp.storbinary("STOR icon-tab.csv", f)
                elif "icon-tab.xlsx" in fname:
                    ftp.storbinary("STOR icon-tab.xlsx", f)
                else:
                    try:
                        ftp.cwd("/stacja.meteo-krosno.pl/archiv")
                    except:
                        pass 
                    f.seek(0)
                    ftp.storbinary(f"STOR {fname}", f)
                    ftp.cwd("/stacja.meteo-krosno.pl/") 
        
        ftp.quit()
        print("📤 FTP Upload zakończony pomyślnie")
    except Exception as e:
        print(f"❌ Błąd podczas wysyłania na FTP: {e}")

# -----------------------
# POBIERANIE I PRZETWARZANIE
# -----------------------
def process_single_param(param, run_date, run_hour):
    base_url = f"https://opendata.dwd.de/weather/nwp/icon-eu/grib/{run_hour.lower()}/{param}/"
    clean_temp()
    
    print(f"🔄 Przetwarzam: {param}...")

    try:
        resp = requests.get(base_url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f" ⚠️ Błąd sieci/URL dla {param}: {e}")
        return None

    files_to_dl = [
        line.split('"')[1]
        for line in resp.text.splitlines()
        if '.grib2.bz2' in line
        and f"{run_date}{run_hour}" in line
        and 'regular-lat-lon' in line
    ]

    if not files_to_dl:
        print(f" ⚠️ Brak plików na serwerze dla {param}")
        return None

    # Pobieramy wszystkie pliki (można odkomentować [:3] dla testów)
    # files_to_dl = files_to_dl[:3] 
    
    grib_files = []
    print(f" ⬇️ Pobieranie {len(files_to_dl)} plików...")

    for fname in files_to_dl:
        local_bz2 = os.path.join(TEMP_DIR, fname)
        local_grib = local_bz2.replace(".bz2", "")
        url = base_url + fname
        
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(local_bz2, "wb") as f:
                    for chunk in r.iter_content(chunk_size=32768):
                        f.write(chunk)
            
            with bz2.open(local_bz2, "rb") as fbz, open(local_grib, "wb") as fg:
                fg.write(fbz.read())
            
            grib_files.append(local_grib)
            os.remove(local_bz2)
        except Exception:
            continue

    if not grib_files:
        return None

    try:
        ds = xr.open_mfdataset(
            grib_files,
            engine="cfgrib",
            combine="nested",
            concat_dim="valid_time",
            parallel=False,
            backend_kwargs={'errors': 'ignore', 'indexpath': ''}
        )
        
        if 'latitude' in ds.coords:
            point = ds.sel(latitude=KROSNO_LAT, longitude=KROSNO_LON, method="nearest")
        else:
            point = ds.sel(lat=KROSNO_LAT, lon=KROSNO_LON, method="nearest")
            
        df = point.to_dataframe().reset_index()
        ds.close()

    except Exception as e:
        print(f" ❌ Błąd xarray dla {param}: {e}")
        return None

    coords_cols = [
        'time', 'valid_time', 'step', 'latitude', 'longitude', 'lat', 'lon', 
        'surface', 'heightAboveGround', 'number', 'meanSea', 'depthBelowLandLayer',
        'entireAtmosphere', 'level'
    ]
    
    data_col = None
    for col in df.columns:
        if col.lower() in RENAME_MAP:
            data_col = col
            break
            
    if not data_col:
        potential = [c for c in df.columns if c not in coords_cols and "time" not in c]
        if potential:
            data_col = potential[0]

    if not data_col:
        return None

    time_series = df.get("valid_time")
    if time_series is None:
        time_series = df.get("time")
    
    if time_series is None:
        return None

    val_series = df[data_col]
    std_name = RENAME_MAP.get(data_col.lower(), param)

    df_out = pd.DataFrame({
        "time": time_series,
        std_name: val_series
    })
    
    df_out = df_out.sort_values("time").drop_duplicates(subset="time")
    return df_out

# -----------------------
# MAIN
# -----------------------
def main():
    run_date, run_hour = get_latest_run()
    run_label = f"{run_date}_{run_hour}"

    print(f"🕒 Czas UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
    print(f"🎯 Run ICON-EU: {run_label}Z")

    final_df = pd.DataFrame()

    for param in PARAMS:
        df_param = process_single_param(param, run_date, run_hour)
        if df_param is not None and not df_param.empty:
            if final_df.empty:
                final_df = df_param
            else:
                final_df = pd.merge(final_df, df_param, on="time", how="outer")
        clean_temp()

    if final_df.empty:
        print("❌ Nie udało się zebrać żadnych danych.")
        return

    final_df = final_df.sort_values("time").reset_index(drop=True)

    fill_zeros = ["tot_prec", "h_snow", "cape_ml", "u_10m", "v_10m"]
    for c in fill_zeros:
        if c in final_df.columns:
            final_df[c] = final_df[c].fillna(0.0)

    # --- OBLICZENIA ---
    if "t_2m" in final_df.columns:
        final_df["T2M [°C]"] = (final_df["t_2m"] - 273.15).round(1)
    if "td_2m" in final_df.columns:
        final_df["D2M [°C]"] = (final_df["td_2m"] - 273.15).round(1)
    
    if "pmsl" in final_df.columns:
        final_df["MSLP [hPa]"] = (final_df["pmsl"] / 100).round(1)

    if "h_snow" in final_df.columns:
        final_df["SNOW_DEPTH [cm]"] = (final_df["h_snow"] * 100).round(1)
    else:
        final_df["SNOW_DEPTH [cm]"] = 0.0

    for raw, out in [("clct", "CC"), ("clcl", "CL"), ("clcm", "CM"), ("clch", "CH")]:
        if raw in final_df.columns:
            vals = final_df[raw]
            if vals.max() <= 1.1:
                vals = vals * 100
            final_df[f"{out} [%]"] = vals.round(0).astype(int)

    if "u_10m" in final_df.columns and "v_10m" in final_df.columns:
        final_df["WSPD [m/s]"] = np.sqrt(final_df["u_10m"]**2 + final_df["v_10m"]**2).round(1)
        final_df["WDIR [°]"] = (np.degrees(np.arctan2(final_df["v_10m"], final_df["u_10m"])) + 360) % 360
        final_df["WDIR [°]"] = final_df["WDIR [°]"].round(0).astype(int)
    
    if "vmax_10m" in final_df.columns:
        final_df["GUST [m/s]"] = final_df["vmax_10m"].round(1)
    elif "WSPD [m/s]" in final_df.columns:
        final_df["GUST [m/s]"] = final_df["WSPD [m/s]"]

    if "tot_prec" in final_df.columns:
        final_df["RRR [mm]"] = final_df["tot_prec"].diff().fillna(0).clip(lower=0)
        final_df["RRR [mm/3h]"] = final_df["RRR [mm]"].rolling(3, min_periods=1).sum().round(1)
    
    final_df["CAPE [J/kg]"] = final_df.get("cape_ml", 0).round(0).astype(int)
    
    if "vis" in final_df.columns:
        final_df["VIS [km]"] = (final_df["vis"] / 1000).round(1)
    else:
        final_df["VIS [km]"] = 50.0

    start_time = final_df["time"].iloc[0]
    final_df["T+ (h)"] = ((final_df["time"] - start_time).dt.total_seconds() / 3600).astype(int)

    # --- ZMIANA NAZW I FORMATOWANIE ---
    final_df = final_df.rename(columns={"time": "Czas"})

    # Definiujemy kolejność kolumn (Czas na początku, Śnieg za RRR)
    final_cols_order = [
        "Czas", 
        "T+ (h)", 
        "T2M [°C]", 
        "D2M [°C]", 
        "MSLP [hPa]",
        "CL [%]", "CM [%]", "CH [%]", "CC [%]", 
        "RRR [mm/3h]",
        "SNOW_DEPTH [cm]",  # PRZESUNIĘTO TUTAJ
        "WSPD [m/s]", 
        "GUST [m/s]", 
        "WDIR [°]", 
        "CAPE [J/kg]", 
        "VIS [km]"
    ]
    
    # Wybieramy tylko te kolumny, które udało się obliczyć
    cols_to_save = [c for c in final_cols_order if c in final_df.columns]
    df_export = final_df[cols_to_save]

    # Zapis plików
    csv_path = os.path.join(OUTPUT_DIR, "icon-tab.csv")
    df_export.to_csv(csv_path, index=False)
    
    arch_path = os.path.join(OUTPUT_DIR, f"icon-arch-{run_label}.csv")
    df_export.to_csv(arch_path, index=False)

    xlsx_path = os.path.join(OUTPUT_DIR, "icon-tab.xlsx")
    df_export.to_excel(xlsx_path, index=False)
    
    arch_xlsx_path = os.path.join(OUTPUT_DIR, f"icon-arch-{run_label}.xlsx")
    df_export.to_excel(arch_xlsx_path, index=False)
    
    print(f"💾 Zapisano pliki w {OUTPUT_DIR}")

    upload_ftp([csv_path, arch_path, xlsx_path, arch_xlsx_path])
    clean_temp()
    print("\n=== Koniec ===")

if __name__ == "__main__":
    main()
