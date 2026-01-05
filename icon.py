#!/usr/bin/env python3
# icon_krosno_smart.py
# Prognoza z modelu ICON (DWD) przez Open-Meteo
# Poprawiona logika wyboru runu – ICON ma 4 runy dziennie: 00Z, 06Z, 12Z, 18Z

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from dotenv import load_dotenv
from ftplib import FTP, error_perm

# -----------------------
# CONFIG
# -----------------------
OUTPUT_DIR = "icon_krosno_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KROSNO_LAT = 49.69
KROSNO_LON = 21.77

# Ręczna inicjalizacja aktualnej pokrywy śnieżnej na starcie prognozy
# Zmień na rzeczywistą wartość z obserwacji/IMGW/stacji (np. 20 cm)
AKTUALNA_POKRYWA_NA_START = 20.0  # cm

# -----------------------
# LOGIKA WYBORU RUNU ICON – 4 runy dziennie: 00Z, 06Z, 12Z, 18Z
# -----------------------
def get_run_info():
    """
    ICON uruchamia się co 6 godzin: 00Z, 06Z, 12Z, 18Z
    Wybieramy najnowszy dostępny run w momencie uruchamiania skryptu.
    """
    now_utc = datetime.utcnow()
    current_time = now_utc.time()

    # Punkty odcięcia (po każdej godzinie runu + ok. 2-3h na przetworzenie)
    cutoff_00 = dt_time(3, 0)   # po 00Z dostępny ok. 02:30-03:00
    cutoff_06 = dt_time(9, 0)   # po 06Z dostępny ok. 08:30-09:00
    cutoff_12 = dt_time(15, 0)  # po 12Z dostępny ok. 14:30-15:00
    cutoff_18 = dt_time(21, 0)  # po 18Z dostępny ok. 20:30-21:00

    if current_time < cutoff_00:
        # Bardzo wcześnie rano – bierzemy wczorajszy 18Z
        run_date = (now_utc - timedelta(days=1)).date()
        run_hour = "18"
    elif current_time < cutoff_06:
        run_date = now_utc.date()
        run_hour = "00"
    elif current_time < cutoff_12:
        run_date = now_utc.date()
        run_hour = "06"
    elif current_time < cutoff_18:
        run_date = now_utc.date()
        run_hour = "12"
    else:
        run_date = now_utc.date()
        run_hour = "18"

    run_date_str = run_date.strftime("%Y%m%d")
    return run_date_str, run_hour, now_utc

RUN_DATE_STR, RUN_HOUR_STR, NOW_UTC = get_run_info()
RUN_LABEL = f"{RUN_DATE_STR}_{RUN_HOUR_STR}"

print(f"🕒 Czas UTC: {NOW_UTC.strftime('%H:%M')}")
print(f"🎯 Zidentyfikowany najnowszy RUN ICON: {RUN_LABEL}Z")

# -----------------------
# OPEN-METEO ICON API
# -----------------------
URL = "https://api.open-meteo.com/v1/dwd-icon"

HOURLY_VARS = [
    "temperature_2m", "dew_point_2m", "pressure_msl",
    "precipitation", "snowfall", "cloud_cover", "cloud_cover_low",
    "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m",
    "wind_direction_10m", "wind_gusts_10m", "cape", "lifted_index",
    "visibility", "temperature_850hPa"
]

PARAMS = {
    "latitude": KROSNO_LAT,
    "longitude": KROSNO_LON,
    "hourly": ",".join(HOURLY_VARS),
    "timezone": "UTC",
    "wind_speed_unit": "ms",
    "forecast_days": 10
}

# -----------------------
# DATA FETCHING
# -----------------------
def fetch_icon_data():
    print(f"📡 Pobieranie danych ICON (DWD) dla runu {RUN_LABEL}Z ...")
    try:
        r = requests.get(URL, params=PARAMS, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        hourly_data = data.get('hourly', {})
        if not hourly_data:
            print("❌ API zwróciło pusty obiekt 'hourly'.")
            return pd.DataFrame()

        df = pd.DataFrame(hourly_data)
        
        for col in df.columns:
            if col == "time": continue
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Mapowanie kolumn
        df.rename(columns={
            "time": "Czas",
            "temperature_2m": "T2M [°C]",
            "dew_point_2m": "D2M [°C]",
            "temperature_850hPa": "T850 [°C]",
            "pressure_msl": "MSLP [hPa]",
            "cloud_cover_low": "CL [%]",
            "cloud_cover_mid": "CM [%]",
            "cloud_cover_high": "CH [%]",
            "cloud_cover": "CC [%]",
            "precipitation": "RRR [mm]",  # opad godzinowy
            "snowfall": "SNOW [cm]",      # świeży śnieg w cm/godz.
            "wind_speed_10m": "WSPD [m/s]",
            "wind_gusts_10m": "GUST [m/s]",
            "wind_direction_10m": "WDIR [°]",
            "cape": "CAPE [J/kg]",
            "lifted_index": "LIFTED [°C]",
            "visibility": "VIS [km]"
        }, inplace=True)

        df["Czas"] = pd.to_datetime(df["Czas"])
        first_time = df["Czas"].iloc[0]
        df["T+ (h)"] = ((df["Czas"] - first_time).dt.total_seconds() / 3600).astype(int)

        # Widzialność z metrów na km
        df["VIS [km]"] = (df["VIS [km]"] / 1000).round(1)

        # Opad co 3 godziny (suma z 3 poprzednich godzin)
        df["RRR [mm/3h]"] = df["RRR [mm]"].rolling(window=3, min_periods=1).sum().round(1)
        df.loc[0:1, "RRR [mm/3h]"] = df.loc[0:1, "RRR [mm]"]  # pierwsze dwie godziny bez pełnej sumy

        # Pokrywa śnieżna – realistyczna aproksymacja
        df["SNOW [cm]"] = df["SNOW [cm]"].fillna(0)
        df["SNOW_DEPTH [cm]"] = 0.0
        melt_factor = 0.15  # cm/godz. na +1°C

        for i in range(len(df)):
            new_snow = df.at[i, "SNOW [cm]"]
            t2m = df.at[i, "T2M [°C]"]
            melt = max(0.0, melt_factor * max(0.0, t2m))

            if i == 0:
                depth = AKTUALNA_POKRYWA_NA_START + new_snow
            else:
                prev_depth = df.at[i-1, "SNOW_DEPTH [cm]"]
                depth = prev_depth + new_snow - melt
            
            df.at[i, "SNOW_DEPTH [cm]"] = max(0.0, depth)

        df["SNOW_DEPTH [cm]"] = df["SNOW_DEPTH [cm]"].round(1)

        # Zaokrąglenia
        cols_round_1 = ["T2M [°C]", "D2M [°C]", "T850 [°C]", "MSLP [hPa]", 
                        "GUST [m/s]", "WSPD [m/s]", "VIS [km]", "SNOW [cm]", "SNOW_DEPTH [cm]"]
        for c in cols_round_1:
            if c in df.columns:
                df[c] = df[c].round(1)

        cols_round_0 = ["CL [%]", "CM [%]", "CH [%]", "CC [%]", "CAPE [J/kg]"]
        for c in cols_round_0:
            if c in df.columns:
                df[c] = df[c].round(0)

        df["WDIR [°]"] = df["WDIR [°]"].round(0)

        # Dokładna kolejność kolumn
        final_order = [
            "Czas", "T+ (h)", "T2M [°C]", "D2M [°C]", "T850 [°C]", "MSLP [hPa]",
            "CL [%]", "CM [%]", "CH [%]", "CC [%]", "RRR [mm/3h]", "SNOW [cm]",
            "WSPD [m/s]", "GUST [m/s]", "WDIR [°]", "CAPE [J/kg]", "LIFTED [°C]", "VIS [km]"
        ]
        df = df[[c for c in final_order if c in df.columns]]

        # Opcjonalnie: wstaw pokrywę śnieżną zaraz po SNOW [cm]
        if "SNOW_DEPTH [cm]" in df.columns:
            df.insert(df.columns.get_loc("SNOW [cm]") + 1, "SNOW_DEPTH [cm]", df.pop("SNOW_DEPTH [cm]"))

        return df

    except Exception as e:
        print(f"❌ Błąd pobierania: {e}")
        return pd.DataFrame()

# -----------------------
# ZAPIS DO CSV
# -----------------------
def save_csv(df):
    if df.empty:
        print("⚠️ Brak danych – nie zapisuję CSV.")
        return []

    # Główny plik
    main_csv = os.path.join(OUTPUT_DIR, "icon-tab.csv")
    df.to_csv(main_csv, index=False, encoding='utf-8')
    print(f"✅ Zapisano: {main_csv}")

    # Archiwum z datą i godziną runu
    arch_name = f"icon-arch-{RUN_DATE_STR}_{RUN_HOUR_STR}.csv"
    arch_csv = os.path.join(OUTPUT_DIR, arch_name)
    df.to_csv(arch_csv, index=False, encoding='utf-8')
    print(f"✅ Zapisano archiwum: {arch_csv}")

    return [main_csv, arch_csv]

# -----------------------
# FTP UPLOAD
# -----------------------
def upload_to_ftp(files):
    load_dotenv()
    host = os.getenv("FTP_HOST")
    user = os.getenv("FTP_USER")
    passwd = os.getenv("FTP_PASS")
    if not all([host, user, passwd]):
        print("⚠️ Brak danych FTP w .env – pomijam upload.")
        return

    try:
        ftp = FTP(host, user, passwd, timeout=30)
        ftp.cwd("/stacja.meteo-krosno.pl/")

        for path in files:
            if not os.path.exists(path):
                continue
            fname = os.path.basename(path)
            with open(path, "rb") as f:
                if "icon-tab.csv" in fname:
                    ftp.storbinary("STOR icon-tab.csv", f)
                # Archiwum
                arch_dir = "/stacja.meteo-krosno.pl/archiv"
                try:
                    ftp.cwd(arch_dir)
                except error_perm:
                    ftp.mkd(arch_dir)
                    ftp.cwd(arch_dir)
                f.seek(0)
                ftp.storbinary(f"STOR {fname}", f)
                print(f"📤 Archiwum: {fname}")
                ftp.cwd("/stacja.meteo-krosno.pl/")

        ftp.quit()
        print("✅ FTP Upload zakończony.")
    except Exception as e:
        print(f"❌ Błąd FTP: {e}")

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    df = fetch_icon_data()
    files = save_csv(df)
    upload_to_ftp(files)
    print("🏁 Gotowe.")