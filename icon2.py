#!/usr/bin/env python3
# icon_final_depth_only.py - Wersja BEZ snow_con, tylko h_snow (pokrywa śnieżna)
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
TEMP_DIR = "temp_grib_icon_depth"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

KROSNO_LAT = 49.69
KROSNO_LON = 21.77

PARAMS = [
    "t_2m", "td_2m", "pmsl", "tot_prec", "h_snow",
    "clct", "clcl", "clcm", "clch", "u_10m", "v_10m", "vmax_10m",
    "cape_ml", "cin_ml", "vis"
]

RENAME_MAP = {
    "t2m": "t_2m", "2t": "t_2m",
    "d2m": "td_2m", "2d": "td_2m",
    "prmsl": "pmsl", "pmsl": "pmsl",
    "tp": "tot_prec",

    # Pokrywa śnieżna – różne możliwe nazwy w plikach GRIB
    "sde": "h_snow",
    "sd": "h_snow",
    "h_snow": "h_snow",
    "snow_depth": "h_snow",
    "depth": "h_snow",

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
# GŁÓWNA LOGIKA POBIERANIA
# -----------------------
def process_single_param(param, run_date, run_hour):
    base_url = f"https://opendata.dwd.de/weather/nwp/icon-eu/grib/{run_hour.lower()}/{param}/"
    clean_temp()

    print(f"🔄 Przetwarzam: {param}...")

    try:
        resp = requests.get(base_url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f" ⚠️ Błąd połączenia z {base_url}: {e}")
        return None

    files_to_dl = [
        line.split('"')[1]
        for line in resp.text.splitlines()
        if '.grib2.bz2' in line
        and f"{run_date}{run_hour}" in line
        and 'regular-lat-lon' in line
    ]

    if not files_to_dl:
        print(f" ⚠️ Nie znaleziono plików GRIB dla {param}")
        return None

    grib_files = []
    print(f" ⬇️ Pobieranie {len(files_to_dl)} plików...")

    for fname in files_to_dl:
        local_bz2 = os.path.join(TEMP_DIR, fname)
        local_grib = local_bz2.replace(".bz2", "")
        url = base_url + fname

        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local_bz2, "wb") as f:
                    for chunk in r.iter_content(chunk_size=32768):
                        f.write(chunk)

            with bz2.open(local_bz2, "rb") as fbz, open(local_grib, "wb") as fg:
                fg.write(fbz.read())

            grib_files.append(local_grib)
            os.remove(local_bz2)

        except Exception as e:
            print(f"   Pominięto plik {fname} → {e}")
            continue

    if not grib_files:
        print(" ⚠️ Żaden plik nie został poprawnie pobrany")
        return None

    try:
        ds = xr.open_mfdataset(
            grib_files,
            engine="cfgrib",
            combine="nested",
            concat_dim="valid_time",
            parallel=False,
            backend_kwargs={
                'errors': 'ignore',
                'indexpath': ''
            }
        )

        # Diagnostyka dla pokrywy śnieżnej
        if param == "h_snow":
            found_var = None
            for v in ds.data_vars:
                if v.lower() in ["sde", "sd", "h_snow", "snow_depth", "depth"]:
                    found_var = v
                    break

            if found_var:
                max_val = float(ds[found_var].max())
                print(f" ❄️ h_snow → zmienna: {found_var}, max w Europie: {max_val:.4f} m  ({max_val*100:.1f} cm)")
                if max_val < 0.0001:
                    print("   ⚠️ UWAGA: prawie same zera w polu pokrywy śnieżnej!")
            else:
                print("   ⚠️ Nie znaleziono zmiennej śniegowej! Dostępne zmienne:", list(ds.data_vars))

        # Wybór najbliższego punktu
        if 'latitude' in ds.coords:
            point = ds.sel(latitude=KROSNO_LAT, longitude=KROSNO_LON, method="nearest")
        else:
            point = ds.sel(lat=KROSNO_LAT, lon=KROSNO_LON, method="nearest")

        df = point.to_dataframe().reset_index()
        ds.close()

    except Exception as e:
        print(f" ❌ Błąd podczas otwierania/przetwarzania GRIB ({param}): {e}")
        return None

    # Rekonstrukcja czytelnego DataFrame
    coords_cols = [
        'time', 'valid_time', 'step', 'latitude', 'longitude', 'lat', 'lon',
        'surface', 'heightAboveGround', 'number', 'meanSea', 'depthBelowLandLayer'
    ]

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
        print(f"   Nie udało się znaleźć kolumny z danymi dla {param}")
        return None

    time_series = df.get("valid_time") or df.get("time")
    if time_series is None:
        return None

    val_series = df[data_col]
    std_name = RENAME_MAP.get(data_col.lower(), param)

    df_out = pd.DataFrame({
        "time": time_series,
        std_name: val_series
    }).drop_duplicates(subset="time")

    return df_out


# -----------------------
# GŁÓWNY PROGRAM
# -----------------------
def main():
    run_date, run_hour = get_latest_run()
    run_label = f"{run_date}_{run_hour}"

    print(f"🕒 Czas UTC:   {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
    print(f"🎯 Run ICON-EU: {run_label}Z\n")

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
        print("❌ Nie udało się pobrać żadnych użytecznych danych.")
        return

    print("\n✅ Pobieranie zakończone. Tworzenie tabeli wynikowej...")

    df = final_df.sort_values("time").reset_index(drop=True)

    # Wypełnianie brakujących wartości dla wybranych pól
    cols_zero = ["tot_prec", "h_snow", "cape_ml", "u_10m", "v_10m"]
    for c in cols_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # ------------------- OBLICZENIA -------------------
    if "t_2m" in df.columns:
        df["T2M [°C]"] = (df["t_2m"] - 273.15).round(1)
    if "td_2m" in df.columns:
        df["D2M [°C]"] = (df["td_2m"] - 273.15).round(1)
    if "pmsl" in df.columns:
        df["MSLP [hPa]"] = (df["pmsl"] / 100).round(1)

    for raw, out in [("clct", "CC"), ("clcl", "CL"), ("clcm", "CM"), ("clch", "CH")]:
        if raw in df.columns:
            vals = df[raw]
            if vals.max() <= 1.1:
                vals *= 100
            df[f"{out} [%]"] = vals.round(0).astype(int)

    if "u_10m" in df.columns and "v_10m" in df.columns:
        df["WSPD [m/s]"] = np.sqrt(df["u_10m"]**2 + df["v_10m"]**2).round(1)
        df["WDIR [°]"] = (np.degrees(np.arctan2(df["v_10m"], df["u_10m"])) + 360) % 360
        df["WDIR [°]"] = df["WDIR [°]"].round(0).astype(int)

    df["GUST [m/s]"] = df["vmax_10m"].round(1) if "vmax_10m" in df.columns else df.get("WSPD [m/s]", 0)

    if "tot_prec" in df.columns:
        df["RRR [mm]"] = df["tot_prec"].diff().fillna(0).clip(lower=0)
        df["RRR [mm/3h]"] = df["RRR [mm]"].rolling(3, min_periods=1).sum().round(1)

    if "h_snow" in df.columns:
        df["SNOW_DEPTH [cm]"] = (df["h_snow"] * 100).round(1)
    else:
        df["SNOW_DEPTH [cm]"] = 0.0

    df["CAPE [J/kg]"] = df.get("cape_ml", 0).round(0).astype(int)
    df["VIS [km]"] = (df["vis"] / 1000).round(1) if "vis" in df.columns else 50.0

    start_time = df["time"].iloc[0]
    df["T+ (h)"] = ((df["time"] - start_time).dt.total_seconds() / 3600).astype(int)

    # ------------------- KOŃCOWA TABELA -------------------
    final_cols = [
        "time", "T+ (h)", "T2M [°C]", "D2M [°C]", "MSLP [hPa]",
        "CL [%]", "CM [%]", "CH [%]", "CC [%]", "RRR [mm/3h]",
        "WSPD [m/s]", "GUST [m/s]", "WDIR [°]", "CAPE [J/kg]", "VIS [km]",
        "SNOW_DEPTH [cm]"
    ]

    df = df.rename(columns={"time": "Czas"})
    df_final = df[[c for c in final_cols if c in df.columns]]

    # Zapis
    csv_path = os.path.join(OUTPUT_DIR, "icon-tab.csv")
    df_final.to_csv(csv_path, index=False)
    print(f"💾 Zapisano: {csv_path}")

    arch_path = os.path.join(OUTPUT_DIR, f"icon-arch-{run_label}.csv")
    df_final.to_csv(arch_path, index=False)

    xlsx_path = os.path.join(OUTPUT_DIR, "icon-tab.xlsx")
    df_final.to_excel(xlsx_path, index=False)
    print(f"💾 Zapisano: {xlsx_path}")

    arch_xlsx_path = os.path.join(OUTPUT_DIR, f"icon-arch-{run_label}.xlsx")
    df_final.to_excel(arch_xlsx_path, index=False)

    upload_ftp([csv_path, arch_path, xlsx_path, arch_xlsx_path])

    clean_temp()
    print("\n=== Koniec przetwarzania ===\n")


if __name__ == "__main__":
    main()
