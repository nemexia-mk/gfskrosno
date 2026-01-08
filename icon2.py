#!/usr/bin/env python3
# icon2.py - Wersja Stabilna (Iteracja po parametrach)

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
TEMP_DIR = "temp_grib_icon"  # Folder tymczasowy na jeden parametr
os.makedirs(OUTPUT_DIR, exist_ok=True)

KROSNO_LAT = 49.69
KROSNO_LON = 21.77

# Lista parametrów do pobrania
PARAMS = [
    "t_2m", "td_2m", "pmsl", "tot_prec", "snow_con", "h_snow",
    "clct", "clcl", "clcm", "clch", "u_10m", "v_10m", "vmax_10m",
    "cape_ml", "cin_ml", "vis"
]

# Mapa: Nazwa w pliku GRIB -> Nazwa w naszej tabeli
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
}

# -----------------------
# FUNKCJE POMOCNICZE
# -----------------------

def get_latest_run():
    """Ustala ostatni dostępny run na podstawie godziny UTC"""
    now_utc = datetime.utcnow()
    current_time = now_utc.time()
    
    # Marginesy czasowe (DWD wrzuca dane z opóźnieniem ok. 3.5h)
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
    """Czyści folder tymczasowy"""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

# -----------------------
# GŁÓWNA LOGIKA POBIERANIA
# -----------------------

def process_single_param(param, run_date, run_hour):
    """
    1. Pobiera pliki TYLKO dla danego parametru.
    2. Otwiera je w xarray (bezpiecznie, bo mają tę samą strukturę).
    3. Zwraca DataFrame z serią czasową.
    """
    base_url = f"https://opendata.dwd.de/weather/nwp/icon-eu/grib/{run_hour.lower()}/{param}/"
    clean_temp() # Czyść przed startem
    
    print(f"🔄 Przetwarzam: {param}...")
    
    # 1. Pobranie listy plików z serwera
    try:
        resp = requests.get(base_url, timeout=10)
        if resp.status_code != 200: 
            print(f"  ⚠️ Brak dostępu do {param}")
            return None
        
        # Filtrujemy pliki: tylko regular-lat-lon i właściwy run
        files_to_dl = [
            line.split('"')[1] 
            for line in resp.text.splitlines() 
            if '.grib2.bz2' in line 
            and f"{run_date}{run_hour}" in line 
            and 'regular-lat-lon' in line
        ]
    except Exception as e:
        print(f"  ⚠️ Błąd sieciowy przy liście plików: {e}")
        return None

    if not files_to_dl:
        print(f"  ⚠️ Brak plików na serwerze dla {param}")
        return None

    # 2. Pobieranie plików
    grib_files = []
    # Pobieramy max 50 plików na raz (np. do 48h prognozy), 
    # jeśli chcesz całą prognozę (120h), usuń [:50] lub zwiększ limit.
    # Tutaj pobieramy wszystkie dostępne:
    
    print(f"  ⬇️ Pobieranie {len(files_to_dl)} plików...")
    
    for fname in files_to_dl:
        local_bz2 = os.path.join(TEMP_DIR, fname)
        local_grib = local_bz2.replace(".bz2", "")
        url = base_url + fname
        
        try:
            # Pobieranie
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(local_bz2, "wb") as f:
                    for chunk in r.iter_content(chunk_size=16384): f.write(chunk)
            
            # Rozpakowanie
            with bz2.open(local_bz2, "rb") as fbz, open(local_grib, "wb") as fg:
                fg.write(fbz.read())
            
            grib_files.append(local_grib)
            os.remove(local_bz2) # Usuwamy bz2 od razu
        except Exception:
            # Ignorujemy pojedyncze błędy pobierania
            continue

    if not grib_files: return None

    # 3. Otwieranie w Xarray
    try:
        # combine='nested' i concat_dim='step' (lub valid_time) to klucz do sukcesu
        # gdy mamy pliki jednego parametru
        ds = xr.open_mfdataset(
            grib_files, 
            engine="cfgrib", 
            combine="nested", 
            concat_dim="valid_time",
            parallel=False,
            backend_kwargs={'errors': 'ignore', 'indexpath': ''}
        )
        
        # Wybór punktu
        # ICON czasem używa 'latitude', czasem 'lat'
        if 'latitude' in ds.coords:
            point = ds.sel(latitude=KROSNO_LAT, longitude=KROSNO_LON, method="nearest")
        else:
            point = ds.sel(lat=KROSNO_LAT, lon=KROSNO_LON, method="nearest")
            
        df = point.to_dataframe().reset_index()
        ds.close()
        
    except Exception as e:
        print(f"  ❌ Błąd xarray dla {param}: {e}")
        return None

    # 4. Sprzątanie i formatowanie tabeli
    
    # Szukamy nazwy kolumny z danymi (bo cfgrib może ją nazwać np. '2t', 'Unknown' itp.)
    # Ignorujemy kolumny współrzędnych
    coords_cols = ['time', 'valid_time', 'step', 'latitude', 'longitude', 'lat', 'lon', 'surface', 'heightAboveGround', 'number', 'meanSea']
    data_col = None
    
    # Najpierw sprawdźmy czy jest w naszej mapie
    for col in df.columns:
        if col.lower() in RENAME_MAP:
            data_col = col
            break
            
    # Jeśli nie, bierzemy pierwszą, która nie jest coordem
    if not data_col:
        potential = [c for c in df.columns if c not in coords_cols]
        if potential: data_col = potential[0]

    if not data_col: return None

    # Ujednolicenie czasu
    if "valid_time" in df.columns:
        df = df.rename(columns={"valid_time": "time"})
    
    # Tworzymy wynikowy DF tylko z czasem i wartością
    df_out = df[["time", data_col]].copy()
    
    # Zmieniamy nazwę na naszą standardową (np. '2t' -> 't_2m')
    std_name = RENAME_MAP.get(data_col.lower(), param) # Jeśli nie ma w mapie, użyj nazwy folderu
    df_out = df_out.rename(columns={data_col: std_name})
    
    # Usuwamy duplikaty czasu (czasem DWD dubluje kroki)
    df_out = df_out.drop_duplicates(subset="time")
    
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

    # --- PĘTLA GŁÓWNA ---
    for param in PARAMS:
        df_param = process_single_param(param, run_date, run_hour)
        
        if df_param is not None and not df_param.empty:
            if final_df.empty:
                final_df = df_param
            else:
                # Łączymy (merge) po czasie. 'outer' zapewnia, że nie zgubimy danych 
                # jeśli jeden parametr ma mniej kroków niż inny.
                final_df = pd.merge(final_df, df_param, on="time", how="outer")
        
        # Ważne: Czyścimy dysk po każdym parametrze!
        clean_temp()

    if final_df.empty:
        print("❌ Nie udało się zebrać żadnych danych.")
        return

    print("\n✅ Pobieranie zakończone. Wykonywanie obliczeń meteo...")

    # --- OBLICZENIA I KONWERSJE ---
    df = final_df.sort_values("time").reset_index(drop=True)
    
    # Wypełnianie braków zerami tam, gdzie to bezpieczne
    cols_zero = ["tot_prec", "snow_con", "h_snow", "cape_ml", "u_10m", "v_10m"]
    for c in cols_zero:
        if c in df.columns: df[c] = df[c].fillna(0.0)

    # Temperatura i Punkt Rosy (K -> C)
    if "t_2m" in df.columns: df["T2M [°C]"] = (df["t_2m"] - 273.15).round(1)
    if "td_2m" in df.columns: df["D2M [°C]"] = (df["td_2m"] - 273.15).round(1)
    
    # Ciśnienie (Pa -> hPa)
    if "pmsl" in df.columns: df["MSLP [hPa]"] = (df["pmsl"] / 100).round(1)
    
    # Chmury (0-1 -> %)
    for raw, out in [("clct", "CC"), ("clcl", "CL"), ("clcm", "CM"), ("clch", "CH")]:
        if raw in df.columns:
            vals = df[raw]
            # Czasem są w skali 0-1, czasem 0-100. Normalizacja:
            if vals.max() <= 1.1: vals = vals * 100
            df[f"{out} [%]"] = vals.round(0)
        else:
            df[f"{out} [%]"] = 0

    # Wiatr (Wektory -> Prędkość/Kierunek)
    if "u_10m" in df.columns and "v_10m" in df.columns:
        df["WSPD [m/s]"] = np.sqrt(df["u_10m"]**2 + df["v_10m"]**2).round(1)
        df["WDIR [°]"] = (np.degrees(np.arctan2(df["v_10m"], df["u_10m"])) + 360) % 360
        df["WDIR [°]"] = df["WDIR [°]"].round(0)
    
    # Porywy
    if "vmax_10m" in df.columns:
        df["GUST [m/s]"] = df["vmax_10m"].round(1)
    else:
        df["GUST [m/s]"] = df.get("WSPD [m/s]", 0)

    # Opad (Akumulowany -> Przyrost)
    if "tot_prec" in df.columns:
        # diff() oblicza różnicę między kolejnymi wierszami
        df["RRR [mm]"] = df["tot_prec"].diff().fillna(0)
        df.loc[df["RRR [mm]"] < 0, "RRR [mm]"] = 0 
        # Suma 3-godzinna (rolling)
        df["RRR [mm/3h]"] = df["RRR [mm]"].rolling(3, min_periods=1).sum().round(1)
    else:
        df["RRR [mm/3h]"] = 0.0

    # Śnieg
    if "h_snow" in df.columns:
        df["SNOW_DEPTH [cm]"] = (df["h_snow"] * 100).round(1) # Metry -> cm
    else:
        df["SNOW_DEPTH [cm]"] = 0.0
    
    if "snow_con" in df.columns:
        df["SNOW [cm]"] = df["snow_con"].round(1) # Zwykle kg/m2 ~= mm ~= cm śniegu świeżego
    else:
        df["SNOW [cm]"] = 0.0

    # Inne
    df["CAPE [J/kg]"] = df.get("cape_ml", 0).round(0)
    
    if "vis" in df.columns:
        df["VIS [km]"] = (df["vis"] / 1000).round(1)
    else:
        df["VIS [km]"] = 50.0

    # Czas prognozy T+
    start_time = df["time"].iloc[0]
    df["T+ (h)"] = ((df["time"] - start_time).dt.total_seconds() / 3600).astype(int)

    # Finalny wybór kolumn
    final_cols = [
        "time", "T+ (h)", "T2M [°C]", "D2M [°C]", "MSLP [hPa]",
        "CL [%]", "CM [%]", "CH [%]", "CC [%]", "RRR [mm/3h]", "SNOW [cm]",
        "WSPD [m/s]", "GUST [m/s]", "WDIR [°]", "CAPE [J/kg]", "VIS [km]", "SNOW_DEPTH [cm]"
    ]
    
    df = df.rename(columns={"time": "Czas"})
    # Wybieramy tylko te kolumny, które udało się obliczyć
    df_final = df[[c for c in final_cols if c in df.columns]]
    
    # --- ZAPIS CSV ---
    csv_path = os.path.join(OUTPUT_DIR, "icon-tab.csv")
    df_final.to_csv(csv_path, index=False)
    print(f"💾 Zapisano: {csv_path}")

    # Archiwum
    arch_path = os.path.join(OUTPUT_DIR, f"icon-arch-{run_label}.csv")
    df_final.to_csv(arch_path, index=False)

    # --- FTP UPLOAD ---
    upload_ftp([csv_path, arch_path])
    
    # Sprzątanie końcowe folderu temp
    clean_temp()

def upload_ftp(files):
    load_dotenv()
    host = os.getenv("FTP_HOST")
    user = os.getenv("FTP_USER")
    passwd = os.getenv("FTP_PASS")
    
    if not all([host, user, passwd]):
        print("⚠️ Brak konfiguracji FTP w .env. Pomijam wysyłanie.")
        return

    try:
        ftp = FTP(host, user, passwd, timeout=60)
        ftp.cwd("/stacja.meteo-krosno.pl/")
        
        for fpath in files:
            if not os.path.exists(fpath): continue
            fname = os.path.basename(fpath)
            
            with open(fpath, "rb") as f:
                if "icon-tab.csv" in fname:
                    ftp.storbinary("STOR icon-tab.csv", f)
                    print("📤 FTP Upload: icon-tab.csv OK")
                else:
                    # Archiwum do podkatalogu
                    try: ftp.cwd("/stacja.meteo-krosno.pl/archiv")
                    except: pass
                    f.seek(0)
                    ftp.storbinary(f"STOR {fname}", f)
                    print(f"📤 FTP Upload: {fname} OK")
                    ftp.cwd("/stacja.meteo-krosno.pl/") # powrót
                    
        ftp.quit()
    except Exception as e:
        print(f"❌ Błąd FTP: {e}")

if __name__ == "__main__":
    main()
