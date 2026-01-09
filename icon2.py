#!/usr/bin/env python3
# icon_krosno_full_fixed.py - Pełna wersja z działającą metodologią + wszystkie parametry + pełny zakres godzin
import os
import shutil
import requests
import pandas as pd
import xarray as xr
import bz2
from datetime import datetime, timedelta, time as dt_time
import numpy as np

# -----------------------
# CONFIG
# -----------------------
OUTPUT_DIR = "icon_krosno_full"
TEMP_DIR = "temp_grib_icon_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

KROSNO_LAT = 49.69
KROSNO_LON = 21.77

# Pełna lista parametrów (bez t – czyli bez T850)
PARAMS = [
    "t_2m", "td_2m", "pmsl", "tot_prec", "snow_con", "h_snow",
    "clct", "clcl", "clcm", "clch", "u_10m", "v_10m", "vmax_10m",
    "cape_ml", "cin_ml", "vis"
]

RENAME_MAP = {
    "t2m": "t_2m", "2t": "t_2m",
    "d2m": "td_2m", "2d": "td_2m",
    "prmsl": "pmsl",
    "tp": "tot_prec",
    "csfwe": "snow_con", "lsfwe": "snow_con",
    "sde": "h_snow",
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


# -----------------------
# GŁÓWNA LOGIKA POBIERANIA PARAMETRU
# -----------------------
def process_single_param(param, run_date, run_hour):
    base_url = f"https://opendata.dwd.de/weather/nwp/icon-eu/grib/{run_hour.lower()}/{param}/"
    clean_temp()

    print(f"🔄 Przetwarzam: {param}...")

    # 1. Pobranie listy plików
    try:
        resp = requests.get(base_url, timeout=15)
        if resp.status_code != 200:
            print(f" ⚠️ Brak dostępu do {param} (status {resp.status_code})")
            return None

        files_to_dl = [
            line.split('"')[1]
            for line in resp.text.splitlines()
            if '.grib2.bz2' in line
            and f"{run_date}_{run_hour}" in line           # ← poprawione filtrowanie!
            and 'regular-lat-lon' in line
        ]

        if not files_to_dl:
            print(f" ⚠️ Brak plików dla {param}")
            return None

        print(f"   → Znaleziono {len(files_to_dl)} plików")

    except Exception as e:
        print(f" ⚠️ Błąd sieciowy przy liście plików: {e}")
        return None

    # 2. Pobieranie plików
    grib_files = []
    print(f" ⬇️ Pobieranie {len(files_to_dl)} plików...")

    for fname in files_to_dl:
        local_bz2 = os.path.join(TEMP_DIR, fname)
        local_grib = local_bz2.replace(".bz2", "")
        url = base_url + fname

        try:
            with requests.get(url, stream=True, timeout=90) as r:  # zwiększony timeout
                r.raise_for_status()
                with open(local_bz2, "wb") as f:
                    for chunk in r.iter_content(chunk_size=32768):
                        f.write(chunk)

            with bz2.open(local_bz2, "rb") as fbz, open(local_grib, "wb") as fg:
                fg.write(fbz.read())

            grib_files.append(local_grib)
            os.remove(local_bz2)

        except Exception as e:
            print(f"   Pomijam {fname} → {str(e)}")
            continue

    if not grib_files:
        print(" ❌ Żaden plik nie został pobrany")
        return None

    # 3. Otwieranie w xarray
    try:
        xr.set_options(use_new_combine_kwarg_defaults=True)
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

    # 4. Bezpieczna rekonstrukcja DataFrame (metoda z działającego kodu)
    coords_cols = ['time', 'valid_time', 'step', 'latitude', 'longitude', 'lat', 'lon',
                   'surface', 'heightAboveGround', 'number', 'meanSea']

    data_col = None
    for col in df.columns:
        if col.lower() in RENAME_MAP:
            data_col = col
            break

    if not data_col:
        potential = [c for c in df.columns if c not in coords_cols]
        if potential:
            data_col = potential[0]

    if not data_col:
        print(" ⚠️ Nie znaleziono kolumny z danymi")
        return None

    time_series = df.get("valid_time")
    if time_series is None:
        time_series = df.get("time")

    if time_series is None:
        print(" ⚠️ Brak poprawnej kolumny czasu")
        return None

    val_series = df[data_col]
    std_name = RENAME_MAP.get(data_col.lower(), param)

    df_out = pd.DataFrame({
        "time": time_series,
        std_name: val_series
    })

    df_out = df_out.drop_duplicates(subset="time")
    df_out = df_out.sort_values("time").reset_index(drop=True)

    print(f"   → Pobrano {len(df_out)} prognoz dla {std_name}")
    return df_out


# -----------------------
# MAIN
# -----------------------
def main():
    run_date, run_hour = get_latest_run()
    run_label = f"{run_date}_{run_hour}"

    print(f"🕒 Czas UTC: {datetime.utcnow().strftime('%H:%M')}")
    print(f"🎯 Wybrany run ICON-EU: {run_label}Z")

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

    print("\n✅ Pobieranie zakończone. Wykonywanie obliczeń meteo...")

    df = final_df.sort_values("time").reset_index(drop=True)

    # Wypełnianie zerami
    cols_zero = ["tot_prec", "snow_con", "h_snow", "cape_ml", "u_10m", "v_10m"]
    for c in cols_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Temperatura i Punkt Rosy
    if "t_2m" in df.columns:
        df["T2M [°C]"] = (df["t_2m"] - 273.15).round(1)
    if "td_2m" in df.columns:
        df["D2M [°C]"] = (df["td_2m"] - 273.15).round(1)

    # Ciśnienie
    if "pmsl" in df.columns:
        df["MSLP [hPa]"] = (df["pmsl"] / 100).round(1)

    # Chmury
    for raw, out in [("clct", "CC"), ("clcl", "CL"), ("clcm", "CM"), ("clch", "CH")]:
        if raw in df.columns:
            vals = df[raw]
            if vals.max() <= 1.1:
                vals = vals * 100
            df[f"{out} [%]"] = vals.round(0)
        else:
            df[f"{out} [%]"] = 0

    # Wiatr
    if "u_10m" in df.columns and "v_10m" in df.columns:
        df["WSPD [m/s]"] = np.sqrt(df["u_10m"]**2 + df["v_10m"]**2).round(1)
        df["WDIR [°]"] = (np.degrees(np.arctan2(df["v_10m"], df["u_10m"])) + 360) % 360
        df["WDIR [°]"] = df["WDIR [°]"].round(0)

    if "vmax_10m" in df.columns:
        df["GUST [m/s]"] = df["vmax_10m"].round(1)
    else:
        df["GUST [m/s]"] = df.get("WSPD [m/s]", 0)

    # Opad
    if "tot_prec" in df.columns:
        df["RRR [mm]"] = df["tot_prec"].diff().fillna(0)
        df.loc[df["RRR [mm]"] < 0, "RRR [mm]"] = 0
        df["RRR [mm/3h]"] = df["RRR [mm]"].rolling(3, min_periods=1).sum().round(1)
    else:
        df["RRR [mm/3h]"] = 0.0

    # Śnieg
    if "h_snow" in df.columns:
        df["SNOW_DEPTH [cm]"] = (df["h_snow"] * 100).round(1)
    else:
        df["SNOW_DEPTH [cm]"] = 0.0

    if "snow_con" in df.columns:
        df["SNOW [cm]"] = df["snow_con"].round(1)
    else:
        df["SNOW [cm]"] = 0.0

    # Inne
    df["CAPE [J/kg]"] = df.get("cape_ml", 0).round(0)
    df["CIN [J/kg]"] = df.get("cin_ml", 0).round(0)

    if "vis" in df.columns:
        df["VIS [km]"] = (df["vis"] / 1000).round(1)
    else:
        df["VIS [km]"] = 50.0

    # Czas prognozy T+
    start_time = df["time"].iloc[0]
    df["T+ (h)"] = ((df["time"] - start_time).dt.total_seconds() / 3600).astype(int)

    # Finalne kolumny
    final_cols = [
        "Czas", "T+ (h)", "T2M [°C]", "D2M [°C]", "MSLP [hPa]",
        "CL [%]", "CM [%]", "CH [%]", "CC [%]", "RRR [mm/3h]", "SNOW [cm]",
        "WSPD [m/s]", "GUST [m/s]", "WDIR [°]", "CAPE [J/kg]", "CIN [J/kg]", "VIS [km]",
        "SNOW_DEPTH [cm]"
    ]

    df = df.rename(columns={"time": "Czas"})
    df_final = df[[c for c in final_cols if c in df.columns]]

    # Zapis do XLSX
    xlsx_path = os.path.join(OUTPUT_DIR, f"icon_full_{run_label}.xlsx")
    df_final.to_excel(xlsx_path, index=False)
    print(f"💾 Zapisano pełną tabelę: {xlsx_path}")

    # Opcjonalnie CSV
    csv_path = os.path.join(OUTPUT_DIR, f"icon_full_{run_label}.csv")
    df_final.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"💾 Zapisano CSV: {csv_path}")


if __name__ == "__main__":
    main()
