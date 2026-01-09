#!/usr/bin/env python3
# icon2.py - Wersja Stabilna (2025/2026) z T850 i CIN zamiast LIFTED
import os
import shutil
import requests
import pandas as pd
import numpy as np
import xarray as xr
import bz2
from datetime import datetime, timedelta, time as dt_time
from dotenv import load_dotenv
from ftplib import FTP

# -----------------------
# CONFIG
# -----------------------
OUTPUT_DIR = "icon_krosno_full"
TEMP_DIR = "temp_grib_icon"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KROSNO_LAT = 49.69
KROSNO_LON = 21.77

PARAMS = [
    "t_2m", "td_2m", "pmsl", "tot_prec", "snow_con", "h_snow",
    "clct", "clcl", "clcm", "clch", "u_10m", "v_10m", "vmax_10m",
    "cape_ml", "cin_ml", "vis"           # ← dodane: temperatura na poziomach (T850)
]

RENAME_MAP = {
    "t2m": "t_2m", "2t": "t_2m",
    "d2m": "td_2m", "2d": "td_2m",
    "u10": "u_10m", "10u": "u_10m",
    "v10": "v_10m", "10v": "v_10m",
    "fg10": "vmax_10m",
    "tp": "tot_prec",
    "prmsl": "pmsl",
    "sde": "h_snow",
    "clct": "clct", "clcl": "clcl", "clcm": "clcm", "clch": "clch",
    "cape_ml": "cape_ml", "cin_ml": "cin_ml",
    "csfwe": "snow_con", "lsfwe": "snow_con",
    "vis": "vis"
                           # ← temperatura na poziomie ciśnienia
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
# GŁÓWNA LOGIKA POBIERANIA JEDNEGO PARAMETRU
# -----------------------
def process_single_param(param, run_date, run_hour):
    base_url = f"https://opendata.dwd.de/weather/nwp/icon-eu/grib/{run_hour.lower()}/{param}/"
    clean_temp()

    print(f"🔄 Przetwarzam: {param}...")
    print(f" ⬇️ Pobieranie plików...")

    # 1. Lista plików
    try:
        resp = requests.get(base_url, timeout=12)
        resp.raise_for_status()
    except Exception as e:
        print(f" ⚠️ Błąd pobierania listy plików: {e}")
        return None

    files_to_dl = [
        line.split('"')[1]
        for line in resp.text.splitlines()
        if '.grib2.bz2' in line
        and f"{run_date}{run_hour}" in line
        and 'regular-lat-lon' in line
    ]

    if not files_to_dl:
        print(" ⚠️ Brak plików na serwerze")
        return None

    # 2. Pobieranie
    grib_files = []
    for fname in files_to_dl:
        local_bz2 = os.path.join(TEMP_DIR, fname)
        local_grib = local_bz2.replace(".bz2", "")
        url = base_url + fname
        try:
            with requests.get(url, stream=True, timeout=40) as r:
                r.raise_for_status()
                with open(local_bz2, "wb") as f:
                    for chunk in r.iter_content(chunk_size=32768):
                        f.write(chunk)
            with bz2.open(local_bz2, "rb") as fbz, open(local_grib, "wb") as fg:
                fg.write(fbz.read())
            grib_files.append(local_grib)
            os.remove(local_bz2)
        except Exception as e:
            print(f" Pomijam plik {fname} → {e}")
            continue

    if not grib_files:
        print(" ❌ Żaden plik nie został pobrany")
        return None

    # 3. Otwieranie xarray
    try:
        ds = xr.open_mfdataset(
            grib_files,
            engine="cfgrib",
            combine="nested",
            concat_dim="valid_time",
            parallel=False,
            backend_kwargs={'errors': 'ignore', 'indexpath': ''}
        )

        # Wybór poziomu 850 hPa dla temperatury
        if param == "t":
            if 'isobaricInhPa' in ds.coords:
                ds = ds.sel(isobaricInhPa=850, method="nearest")
            elif 'pressure' in ds.coords:
                ds = ds.sel(pressure=85000, method="nearest")  # w Pa

        # Wybór punktu
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
            point = ds.sel(latitude=KROSNO_LAT, longitude=KROSNO_LON, method="nearest")
        elif 'lat' in ds.coords and 'lon' in ds.coords:
            point = ds.sel(lat=KROSNO_LAT, lon=KROSNO_LON, method="nearest")
        else:
            print(" ❌ Brak współrzędnych lat/lon")
            return None

        df = point.to_dataframe().reset_index()

    except Exception as e:
        print(f" ❌ Błąd xarray/cfgrib: {e}")
        return None
    finally:
        ds.close()

    # 4. Bezpieczne wyciągnięcie czasu i wartości (bardzo odporna wersja)
    possible_time_cols = ['valid_time', 'time', 'forecast_reference_time', 'step']
    time_series = None

    for col in possible_time_cols:
        if col in df.columns:
            if isinstance(df[col], pd.Series):
                time_series = df[col]
                break
            elif isinstance(df[col], pd.DataFrame) and len(df[col].columns) > 0:
                time_series = df[col].iloc[:, 0]
                break

    if time_series is None:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_series = df[col]
                break

    if time_series is None:
        print(" ⚠️ Nie udało się znaleźć poprawnej kolumny czasu!")
        return None

    # Kolumna z wartościami
    exclude = {
        'time', 'valid_time', 'step', 'forecast_reference_time',
        'latitude', 'longitude', 'lat', 'lon',
        'number', 'surface', 'heightAboveGround', 'meanSea',
        'isobaricInhPa', 'pressure', 'height'
    }

    data_cols = [
        c for c in df.columns
        if c not in exclude
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and df[c].dtype != 'object'
    ]

    if not data_cols:
        print(" ⚠️ Nie znaleziono kolumny z wartościami parametru")
        return None

    value_col = data_cols[0]
    std_name = RENAME_MAP.get(value_col.lower(), param)

    df_out = pd.DataFrame({
        "time": time_series,
        std_name: df[value_col]
    })

    df_out = df_out.drop_duplicates(subset="time", keep='first')
    df_out = df_out.sort_values("time").reset_index(drop=True)

    return df_out


# -----------------------
# MAIN
# -----------------------
def main():
    run_date, run_hour = get_latest_run()
    print(f"🕒 Czas UTC: {datetime.utcnow().strftime('%H:%M')}")
    print(f"🎯 Wybrany run ICON-EU: {run_date}_{run_hour}Z")

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
        print("❌ Nie udało się pobrać żadnych danych.")
        return

    print("\n✅ Zebrano dane. Wykonuję obliczenia...")

    df = final_df.sort_values("time").reset_index(drop=True)

    # Wypełnianie brakujących wartości
    cols_zero_fill = ["tot_prec", "snow_con", "h_snow", "cape_ml", "u_10m", "v_10m"]
    for c in cols_zero_fill:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Konwersje jednostek
    if "t_2m" in df.columns:
        df["T2M [°C]"] = (df["t_2m"] - 273.15).round(1)
    if "td_2m" in df.columns:
        df["D2M [°C]"] = (df["td_2m"] - 273.15).round(1)
    if "t_850" in df.columns:
        df["T850 [°C]"] = (df["t_850"] - 273.15).round(1)
    if "pmsl" in df.columns:
        df["MSLP [hPa]"] = (df["pmsl"] / 100).round(1)

    # Chmury
    for p, out in [("clct", "CC"), ("clcl", "CL"), ("clcm", "CM"), ("clch", "CH")]:
        if p in df.columns:
            vals = df[p]
            if vals.max() <= 1.1:
                vals = vals * 100
            df[f"{out} [%]"] = vals.round(0).astype(int)
        else:
            df[f"{out} [%]"] = 0

    # Wiatr
    if "u_10m" in df.columns and "v_10m" in df.columns:
        df["WSPD [m/s]"] = np.sqrt(df["u_10m"]**2 + df["v_10m"]**2).round(1)
        df["WDIR [°]"] = (np.degrees(np.arctan2(df["v_10m"], df["u_10m"])) + 360) % 360
        df["WDIR [°]"] = df["WDIR [°]"].round(0).astype(int)

    if "vmax_10m" in df.columns:
        df["GUST [m/s]"] = df["vmax_10m"].round(1)
    else:
        df["GUST [m/s]"] = df.get("WSPD [m/s]", 0.0)

    # Opady
    if "tot_prec" in df.columns:
        df["RRR [mm]"] = df["tot_prec"].diff().fillna(0)
        df.loc[df["RRR [mm]"] < 0, "RRR [mm]"] = 0
        df["RRR [mm/3h]"] = df["RRR [mm]"].rolling(3, min_periods=1).sum().round(1)
    else:
        df["RRR [mm/3h]"] = 0.0

    # Śnieg
    df["SNOW_DEPTH [cm]"] = (df.get("h_snow", 0) * 100).round(1)
    df["SNOW [cm]"] = df.get("snow_con", 0).round(1)

    # Pozostałe
    df["CAPE [J/kg]"] = df.get("cape_ml", 0).round(0).astype(int)
    df["CIN [J/kg]"] = df.get("cin_ml", 0).round(0).astype(int)     # zamiast LIFTED

    df["VIS [km]"] = (df.get("vis", 50000) / 1000).round(1)

    # Godzina prognozy
    start_time = df["time"].iloc[0]
    df["T+ (h)"] = ((df["time"] - start_time).dt.total_seconds() / 3600).astype(int)

    # Finalne kolumny (dokładnie w kolejności, którą chciałeś)
    final_cols = [
        "Czas", "T+ (h)",
        "T2M [°C]", "D2M [°C]", "T850 [°C]",
        "MSLP [hPa]",
        "CL [%]", "CM [%]", "CH [%]", "CC [%]",
        "RRR [mm/3h]", "SNOW [cm]",
        "WSPD [m/s]", "GUST [m/s]", "WDIR [°]",
        "CAPE [J/kg]", "CIN [J/kg]",
        "VIS [km]"
    ]

    df = df.rename(columns={"time": "Czas"})
    available_cols = [c for c in final_cols if c in df.columns]
    df_final = df[available_cols]

    # Zapis
    csv_path = os.path.join(OUTPUT_DIR, "icon-tab.csv")
    df_final.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"💾 Zapisano: {csv_path}")

    arch_path = os.path.join(OUTPUT_DIR, f"icon-arch-{run_date}_{run_hour}.csv")
    df_final.to_csv(arch_path, index=False, encoding='utf-8-sig')

    # FTP
    upload_ftp([csv_path, arch_path])

    clean_temp()


def upload_ftp(files):
    load_dotenv()
    host = os.getenv("FTP_HOST")
    user = os.getenv("FTP_USER")
    passwd = os.getenv("FTP_PASS")

    if not all([host, user, passwd]):
        print("⚠️ Brak konfiguracji FTP → pomijam wysyłanie")
        return

    try:
        ftp = FTP(host, user, passwd, timeout=60)
        ftp.cwd("/stacja.meteo-krosno.pl/")

        for fpath in files:
            if not os.path.exists(fpath):
                continue
            fname = os.path.basename(fpath)
            with open(fpath, "rb") as f:
                if fname == "icon-tab.csv":
                    ftp.storbinary("STOR icon-tab.csv", f)
                    print("📤 FTP Upload: icon-tab.csv OK")
                else:
                    try:
                        ftp.cwd("/stacja.meteo-krosno.pl/archiv")
                    except:
                        pass
                    ftp.storbinary(f"STOR {fname}", f)
                    print(f"📤 FTP Upload: {fname} OK")
                    ftp.cwd("/stacja.meteo-krosno.pl/")

        ftp.quit()
    except Exception as e:
        print(f"❌ Błąd FTP: {e}")


if __name__ == "__main__":
    main()

