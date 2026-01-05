#!/usr/bin/env python3
# icon_full_fix.py
# Wersja naprawiona:
# 1. Pobiera serię czasową (nie tylko 1 plik).
# 2. Obsługuje 0-wymiarowe obiekty (expand_dims).
# 3. Mapuje nazwy zmiennych (clct/CLCT/t2m).

import os
import shutil
import requests
import pandas as pd
import numpy as np
import xarray as xr
import bz2
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

# Parametry shortName
PARAMS = [
    "t_2m", "td_2m", "pmsl", "tot_prec", "snow_con", "h_snow",
    "clct", "clcl", "clcm", "clch", "u_10m", "v_10m", "vmax_10m",
    "cape_ml", "cin_ml", "vis"
]

# -----------------------
# WYBÓR RUNU
# -----------------------
def get_latest_run():
    now_utc = datetime.utcnow()
    current_time = now_utc.time()

    cutoffs = [dt_time(3, 30), dt_time(9, 30), dt_time(15, 30), dt_time(21, 30)]
    run_hours = ["18", "00", "06", "12"]

    # Prosta logika wyboru ostatniego dostępnego runu
    # (DWD udostępnia runy z opóźnieniem ok. 3.5h)
    selected_run = None
    selected_date = now_utc

    if current_time < cutoffs[0]: # Przed 03:30 -> Run 18 z wczoraj
        selected_date = now_utc - timedelta(days=1)
        selected_run = "18"
    elif current_time < cutoffs[1]: # Przed 09:30 -> Run 00 z dziś
        selected_run = "00"
    elif current_time < cutoffs[2]: # Przed 15:30 -> Run 06 z dziś
        selected_run = "06"
    elif current_time < cutoffs[3]: # Przed 21:30 -> Run 12 z dziś
        selected_run = "12"
    else: # Po 21:30 -> Run 18 z dziś
        selected_run = "18"

    return selected_date.strftime("%Y%m%d"), selected_run

RUN_DATE, RUN_HOUR = get_latest_run()
RUN_LABEL = f"{RUN_DATE}_{RUN_HOUR}"

print(f"🕒 Czas UTC: {datetime.utcnow().strftime('%H:%M')}")
print(f"🎯 Wybrany run ICON-EU: {RUN_LABEL}Z")

BASE_URL = f"https://opendata.dwd.de/weather/nwp/icon-eu/grib/{RUN_HOUR.lower()}"

# -----------------------
# POBIERANIE I PRZETWARZANIE
# -----------------------
def fetch_and_process():
    temp_dir = "temp_grib_icon"
    # Czyścimy folder temp na starcie, żeby nie było śmieci .idx
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    grib_files = []

    # --- POBIERANIE ---
    for param in PARAMS:
        param_url = f"{BASE_URL}/{param}/"
        print(f"Szukam {param}...")

        try:
            resp = requests.get(param_url, timeout=10)
            if resp.status_code != 200:
                print(f"  ⚠️ Brak dostępu do katalogu {param}")
                continue

            lines = resp.text.splitlines()
            # Filtrujemy pliki dla danego RUNu
            # Pobieramy WSZYSTKIE dostępne kroki czasowe (regular-lat-lon)
            files = [
                line.split('"')[1] 
                for line in lines 
                if '.grib2.bz2' in line 
                and RUN_DATE + RUN_HOUR in line
                and 'regular-lat-lon' in line
            ]

            if not files:
                print(f"  ⚠️ Brak plików dla {param}")
                continue

            print(f"  ⬇️ Pobieram {len(files)} plików dla {param}...")
            
            # Pobieramy pliki (pętla)
            for file_name in files:
                file_url = param_url + file_name
                local_bz2 = os.path.join(temp_dir, file_name)
                local_grib = local_bz2.replace(".bz2", "")

                if not os.path.exists(local_grib):
                    try:
                        with requests.get(file_url, stream=True, timeout=60) as r:
                            r.raise_for_status()
                            with open(local_bz2, "wb") as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    f.write(chunk)
                        
                        # Rozpakowanie
                        with bz2.open(local_bz2, "rb") as fbz, open(local_grib, "wb") as fg:
                            fg.write(fbz.read())
                        
                        # Usuwamy bz2 żeby oszczędzić miejsce
                        os.remove(local_bz2)
                        
                        grib_files.append(local_grib)
                    except Exception as e:
                        print(f"    Błąd pobierania {file_name}: {e}")

        except Exception as e:
            print(f"  ❌ Błąd sieciowy {param}: {e}")

    if not grib_files:
        print("❌ Nie pobrano żadnych plików GRIB.")
        return pd.DataFrame()

    print(f"✅ Pomyślnie pobrano {len(grib_files)} plików GRIB. Otwieranie datasetu...")

    # --- ODCZYT GRIB (XARRAY) ---
    try:
        # backend_kwargs={'indexpath': ''} zapobiega tworzeniu plików .idx, które sypią błędami
        ds = xr.open_mfdataset(
            grib_files, 
            engine="cfgrib", 
            combine="by_coords", 
            backend_kwargs={'errors': 'ignore', 'indexpath': ''}
        )
    except Exception as e:
        print(f"❌ Błąd otwierania GRIB przez xarray: {e}")
        return pd.DataFrame()

    # --- WYBÓR PUNKTU ---
    try:
        if 'latitude' in ds.coords:
            point = ds.sel(latitude=KROSNO_LAT, longitude=KROSNO_LON, method="nearest")
        elif 'lat' in ds.coords:
            point = ds.sel(lat=KROSNO_LAT, lon=KROSNO_LON, method="nearest")
        else:
            print("❌ Nie znaleziono koordynatów (lat/lon) w pliku.")
            return pd.DataFrame()
    except Exception as e:
        print(f"❌ Błąd wyboru punktu (sel): {e}")
        return pd.DataFrame()

    # --- NAPRAWA WYMIARÓW (FIX 0-DIMENSIONAL ERROR) ---
    # Jeśli pobrano tylko 1 plik, 'time' może nie być wymiarem.
    if "time" not in point.dims and "step" not in point.dims and "valid_time" not in point.dims:
        print("⚠️ Obiekt 0-wymiarowy. Dodaję sztuczny wymiar czasu...")
        point = point.expand_dims("time")

    # --- KONWERSJA DO DATAFRAME ---
    try:
        df = point.to_dataframe().reset_index()
    except Exception as e:
        print(f"❌ Błąd konwersji do DataFrame: {e}")
        # Ostateczna deska ratunku - jeśli to skalar
        try:
             df = pd.DataFrame([point.to_dict(data=True)['data_vars']])
        except:
             return pd.DataFrame()

    # --- 1. NORMALIZACJA NAZW (LOWERCASE) ---
    df.columns = [c.lower() for c in df.columns]
    print(f"🧐 Kolumny surowe: {df.columns.tolist()}")

    # --- 2. NAPRAWA CZASU ---
    if "valid_time" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"valid_time": "time"})
    
    # Jeśli wciąż brak time, a jest step (dla pojedynczego pliku)
    if "time" not in df.columns and "step" in df.columns and not df.empty:
         # Próbujemy odtworzyć czas z atrybutów lub dataDate (rzadkie w to_dataframe)
         # Zakładamy start runu + step
         run_start = datetime.strptime(RUN_DATE + RUN_HOUR, "%Y%m%d%H")
         df["time"] = run_start + df["step"]

    if "time" not in df.columns:
        print("❌ Brak kolumny czasu. Nie można stworzyć tabeli.")
        return pd.DataFrame()

    # --- 3. MAPOWANIE ZMIENNYCH ---
    rename_map = {
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
        "csfwe": "snow_con", "lsfwe": "snow_con"
    }
    
    actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=actual_rename)

    # --- 4. UZUPEŁNIANIE BRAKÓW ---
    required = ["t_2m", "td_2m", "pmsl", "clct", "vmax_10m", "cape_ml", "u_10m", "v_10m", "tot_prec", "snow_con", "h_snow"]
    for col in required:
        if col not in df.columns:
            df[col] = 0.0

    # --- 5. OBLICZENIA METEO ---
    df = df.sort_values("time").reset_index(drop=True)

    df["T2M [°C]"] = (df["t_2m"] - 273.15).round(1)
    df["D2M [°C]"] = (df["td_2m"] - 273.15).round(1)
    df["MSLP [hPa]"] = (df["pmsl"] / 100).round(1)

    # Chmury
    for c in ["clct", "clcl", "clcm", "clch"]:
        if c in df.columns:
            if df[c].max() <= 1.1: df[c] = df[c] * 100
            df[c.upper().replace("CLCT", "CC")[:2] + " [%]"] = df[c].round(0)
    
    # Mapowanie nazw chmur jeśli pętla wyżej nie pokryła nazw wynikowych
    if "clct" in df.columns: df["CC [%]"] = df["clct"].round(0)
    if "clcl" in df.columns: df["CL [%]"] = df["clcl"].round(0)
    if "clcm" in df.columns: df["CM [%]"] = df["clcm"].round(0)
    if "clch" in df.columns: df["CH [%]"] = df["clch"].round(0)

    # Widoczność
    if "vis" in df.columns:
        df["VIS [km]"] = (df["vis"] / 1000).round(1)
    else:
        df["VIS [km]"] = 50.0

    df["GUST [m/s]"] = df["vmax_10m"].round(1)
    df["CAPE [J/kg]"] = df["cape_ml"].round(0)

    # Wiatr wektorowo
    df["WSPD [m/s]"] = np.sqrt(df["u_10m"]**2 + df["v_10m"]**2).round(1)
    df["WDIR [°]"] = (np.degrees(np.arctan2(df["v_10m"], df["u_10m"])) + 360) % 360
    df["WDIR [°]"] = df["WDIR [°]"].round(0)

    # Śnieg
    df["SNOW_DEPTH [cm]"] = (df["h_snow"] * 100).round(1)
    df["SNOW [cm]"] = df["snow_con"].round(1)

    # Opad (diff)
    # Jeśli mamy tylko 1 rekord, diff da NaN, więc fillna(0)
    df["RRR [mm]"] = df["tot_prec"].diff().fillna(0)
    df.loc[df["RRR [mm]"] < 0, "RRR [mm]"] = 0
    df["RRR [mm/3h]"] = df["RRR [mm]"].rolling(window=3, min_periods=1).sum().round(1)

    # T+
    first_time = df["time"].iloc[0]
    df["T+ (h)"] = ((df["time"] - first_time).dt.total_seconds() / 3600).astype(int)

    df["T850 [°C]"] = np.nan
    df["LIFTED [°C]"] = np.nan

    # Finalna tabela
    final_cols = [
        "time", "T+ (h)", "T2M [°C]", "D2M [°C]", "T850 [°C]", "MSLP [hPa]",
        "CL [%]", "CM [%]", "CH [%]", "CC [%]", "RRR [mm/3h]", "SNOW [cm]",
        "WSPD [m/s]", "GUST [m/s]", "WDIR [°]", "CAPE [J/kg]", "LIFTED [°C]", "VIS [km]", "SNOW_DEPTH [cm]"
    ]
    df = df.rename(columns={"time": "Czas"})
    df = df[[c for c in final_cols if c in df.columns]]

    # Sprzątanie
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    return df

# -----------------------
# ZAPIS DO CSV
# -----------------------
def save_csv(df):
    if df.empty:
        print("⚠️ Brak danych – nie zapisuję CSV.")
        return []

    main_csv = os.path.join(OUTPUT_DIR, "icon-tab.csv")
    df.to_csv(main_csv, index=False, encoding='utf-8')
    print(f"✅ Zapisano: {main_csv}")

    arch_name = f"icon-arch-{RUN_DATE}_{RUN_HOUR}.csv"
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
            if not os.path.exists(path): continue
            fname = os.path.basename(path)
            with open(path, "rb") as f:
                if "icon-tab.csv" in fname:
                    ftp.storbinary("STOR icon-tab.csv", f)
                
                arch_dir = "/stacja.meteo-krosno.pl/archiv"
                try:
                    ftp.cwd(arch_dir)
                except error_perm:
                    try:
                        ftp.mkd(arch_dir)
                        ftp.cwd(arch_dir)
                    except: pass
                
                f.seek(0)
                ftp.storbinary(f"STOR {fname}", f)
                print(f"📤 FTP Upload: {fname}")
                ftp.cwd("/stacja.meteo-krosno.pl/")
        ftp.quit()
        print("✅ FTP Upload OK.")
    except Exception as e:
        print(f"❌ Błąd FTP: {e}")

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    df = fetch_and_process()
    if not df.empty:
        print("\n--- Podgląd danych ---")
        print(df.head())
        files = save_csv(df)
        upload_to_ftp(files)
        print("🏁 Gotowe!")
    else:
        print("🏁 Brak danych.") 
