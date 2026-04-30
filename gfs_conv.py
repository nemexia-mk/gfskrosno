#!/usr/bin/env python3
# gfs_krosno_conv.py
# Wersja konwekcyjna dla łowców burz. Pobiera GFS, wylicza STP, CAPE, CIN, Shear, Helicity, PWAT.
# Generuje Excel + gfs-conv.csv; retry co 10 min; wysyła wyniki na FTP (credentials z .env).

import os
import requests
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from time import sleep
from dotenv import load_dotenv
from ftplib import FTP, error_perm
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------
# CONFIG
# -----------------------
OUTPUT_DIR = "gfs_krosno_conv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Krosno
TOP_LAT = 50.0
BOTTOM_LAT = 49.4
LEFT_LON = 21.3
RIGHT_LON = 22.01
KROSNO_LAT = 49.69
KROSNO_LON = 21.77

# -----------------------
# LOGIKA DATY I GODZINY RUNU GFS
# -----------------------
now = datetime.utcnow()
current_time = now.time()
print(f"Aktualny czas UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}")

if current_time >= time(20, 0) or current_time < time(3, 0):
    RUN_HOUR = "18"
    if current_time >= time(22, 0):
        RUN_DATE = now.strftime("%Y%m%d")
    else:
        RUN_DATE = (now - timedelta(days=1)).strftime("%Y%m%d")
elif time(3, 0) <= current_time < time(8, 30):
    RUN_HOUR = "00"
    RUN_DATE = now.strftime("%Y%m%d")
elif time(8, 30) <= current_time < time(14, 30):
    RUN_HOUR = "06"
    RUN_DATE = now.strftime("%Y%m%d")
else:
    RUN_HOUR = "12"
    RUN_DATE = now.strftime("%Y%m%d")

print(f"Wybrano run {RUN_HOUR}Z z dnia {RUN_DATE}")

CYCLE_DIR = f"gfs.{RUN_DATE}/{RUN_HOUR}/atmos"
BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
FORECAST_HOURS = list(range(0, 121, 3)) # Ograniczone do 120h (standard dla parametrów konwekcyjnych)

# -----------------------
# TRYB PRZYROSTOWY
# -----------------------
RETRY_INTERVAL_SECONDS = 10 * 60
MAX_TOTAL_WAIT_MINUTES = 90

# Static NOMADS filter - rozszerzony o warstwy konwekcyjne
STATIC_MIDDLE = (
    "&lev_2_m_above_ground=on"
    "&lev_10_m_above_ground=on"
    "&lev_surface=on"
    "&lev_mean_sea_level=on"
    "&lev_entire_atmosphere_%28considered_as_a_single_layer%29=on"
    "&lev_850_mb=on"
    "&lev_500_mb=on"
    "&lev_1000-0_m_above_ground=on"
    "&lev_3000-0_m_above_ground=on"
    "&var_TMP=on"
    "&var_DPT=on"
    "&var_RH=on"
    "&var_APCP=on"
    "&var_GUST=on"
    "&var_UGRD=on"
    "&var_VGRD=on"
    "&var_PRMSL=on"
    "&var_CAPE=on"
    "&var_CIN=on"
    "&var_LFTX=on"
    "&var_HLCY=on"
    "&var_PWAT=on"
    "&subregion=on"
    f"&toplat={TOP_LAT}"
    f"&bottomlat={BOTTOM_LAT}"
    f"&leftlon={LEFT_LON}"
    f"&rightlon={RIGHT_LON}"
)

# -----------------------
# HELPERS
# -----------------------
def build_url(file_name):
    url = f"{BASE_URL}?file={file_name}&dir=/{CYCLE_DIR}{STATIC_MIDDLE}"
    return url.replace("suubregion", "subregion").replace("lev_entire_atmoosphere", "lev_entire_atmosphere")

SHORTNAMES = {
    "t2m": ["t2m", "2t", "tmp2m", "tmp"],
    "d2m": ["d2m", "dew2m", "dpt"],
    "msl": ["msl", "pres", "prmsl", "sp"],
    "cape": ["cape", "sbcape"],
    "cin": ["cin"],
    "lftx": ["lftx"],
    "apcp": ["apcp", "tp"],
    "pwat": ["pwat", "tcwv"],
    "gust": ["gust"],
    "u": ["ugrd", "u", "u10"],
    "v": ["vgrd", "v", "v10"],
}

def try_open_by_filter(file_path, filter_by_keys):
    try:
        return xr.open_dataset(file_path, engine="cfgrib", backend_kwargs={"filter_by_keys": filter_by_keys, "indexpath": ""})
    except Exception:
        return None

def safe_get_point(ds, shortname_list):
    if ds is None:
        return np.nan
    for sn in shortname_list:
        if sn in ds:
            try:
                val = ds[sn].sel(latitude=KROSNO_LAT, longitude=KROSNO_LON, method="nearest")
                return float(np.squeeze(np.array(val)))
            except Exception:
                continue
    return np.nan

def lcl_height_m(t_c, td_c):
    if np.isnan(t_c) or np.isnan(td_c):
        return np.nan
    diff = t_c - td_c
    return float(np.round(125.0 * max(0.0, diff), 1))

def calculate_stp(cape, lcl, srh1, shear06):
    """
    Kalkulacja STP (Significant Tornado Parameter).
    Formuła: (SBCAPE/1500) * ((2000-LCL)/1000) * (SRH01/150) * (BS06/20)
    Z zachowaniem standardowych limitów dla konwekcji.
    """
    if any(np.isnan(x) for x in [cape, lcl, srh1, shear06]):
        return np.nan
    
    # 1. CAPE term
    cape_term = cape / 1500.0
    
    # 2. LCL term
    if lcl < 1000: lcl_term = 1.0
    elif lcl > 2000: lcl_term = 0.0
    else: lcl_term = (2000.0 - lcl) / 1000.0
        
    # 3. SRH1 term
    srh1_term = srh1 / 150.0
    
    # 4. Shear06 term (w m/s) -> 12.5 m/s to ok. 25 kts, 30 m/s to ok. 60 kts
    if shear06 < 12.5: shear_term = 0.0
    elif shear06 > 30.0: shear_term = 1.5
    else: shear_term = shear06 / 20.0
        
    stp = cape_term * lcl_term * srh1_term * shear_term
    return float(np.round(max(0.0, stp), 2))

def handle_404_and_exit():
    print("❌ Błąd 404 dla pierwszej godziny (f000) - przerywam cały skrypt.")
    sys.exit(0)

# -----------------------
# DOWNLOADER
# -----------------------
def download_missing_gribs_parallel(forecast_hours):
    pending = []
    downloaded = []

    for fh in forecast_hours:
        local_path = os.path.join(OUTPUT_DIR, f"krosno_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")
        if os.path.exists(local_path) and os.path.getsize(local_path) > 30000: # 30KB min dla mniejszych plików konwekcyjnych
            downloaded.append(local_path)
            continue
        pending.append(fh)

    if not pending:
        print("   Wszystkie pliki już są na dysku.")
        return downloaded, []

    print(f"   → Rozpoczynam równoległe pobieranie {len(pending)} brakujących plików...")

    def fetch_single(fh):
        fstr = f"{fh:03d}"
        grib_filename = f"gfs.t{RUN_HOUR}z.pgrb2.0p25.f{fstr}"
        local_path = os.path.join(OUTPUT_DIR, f"krosno_{RUN_DATE}_{RUN_HOUR}z_f{fstr}.grib2")
        url = build_url(grib_filename)

        try:
            r = requests.get(url, headers=HEADERS, timeout=90)
            if r.status_code == 404:
                return fh, None, "404"
            if r.status_code != 200 or b"GRIB" not in r.content[:10]:
                return fh, None, f"HTTP {r.status_code}"
            with open(local_path, "wb") as f:
                f.write(r.content)
            size_kb = len(r.content) / 1024
            print(f"   ✓ Pobrano f{fstr}  ({size_kb:.1f} KB)")
            return fh, local_path, None
        except Exception as e:
            return fh, None, str(e)

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_fh = {executor.submit(fetch_single, fh): fh for fh in pending}
        for future in as_completed(future_to_fh):
            fh, path, error = future.result()
            if path:
                downloaded.append(path)

    still_missing = [fh for fh in forecast_hours if not (os.path.exists(os.path.join(OUTPUT_DIR, f"krosno_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")) and os.path.getsize(os.path.join(OUTPUT_DIR, f"krosno_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")) > 30000)]
    return downloaded, still_missing

# -----------------------
# CORE: PRZETWARZANIE DANYCH
# -----------------------
def process_local_gribs(forecast_hours):
    rows = []
    for fh in forecast_hours:
        local_path = os.path.join(OUTPUT_DIR, f"krosno_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")
        if not os.path.exists(local_path):
            continue
        
        # Filtrowanie głównych poziomów
        ds_2m = try_open_by_filter(local_path, {"typeOfLevel": "heightAboveGround", "level": 2})
        ds_10m = try_open_by_filter(local_path, {"typeOfLevel": "heightAboveGround", "level": 10})
        ds_surface = try_open_by_filter(local_path, {"typeOfLevel": "surface"})
        ds_msl = try_open_by_filter(local_path, {"typeOfLevel": "meanSea"})
        ds_pwat = try_open_by_filter(local_path, {"typeOfLevel": "entireAtmosphere"})
        ds_850 = try_open_by_filter(local_path, {"typeOfLevel": "isobaricInhPa", "level": 850})
        ds_500 = try_open_by_filter(local_path, {"typeOfLevel": "isobaricInhPa", "level": 500})
        
        # Helicity (często wymaga precyzyjnego filtru ze względu na topLevel i shortName)
        ds_srh1 = try_open_by_filter(local_path, {"shortName": "hlcy", "topLevel": 1000})
        ds_srh3 = try_open_by_filter(local_path, {"shortName": "hlcy", "topLevel": 3000})

        def get_val(ds_obj, key):
            return safe_get_point(ds_obj, SHORTNAMES.get(key, [key]))

        try:
            # Termika
            t2m = get_val(ds_2m, "t2m") - 273.15
            d2m = get_val(ds_2m, "d2m") - 273.15
            sp = get_val(ds_msl, "msl") / 100.0
            
            # Konwekcja
            cape = get_val(ds_surface, "cape")
            cin = get_val(ds_surface, "cin")
            lftx = get_val(ds_surface, "lftx")
            pwat = get_val(ds_pwat, "pwat")
            
            # Wiatr powierzchnia (10m)
            u10 = get_val(ds_10m, "u")
            v10 = get_val(ds_10m, "v")
            gust = get_val(ds_surface, "gust")
            
            wind_ms = np.nan
            wind_dir = np.nan
            if not np.isnan(u10) and not np.isnan(v10):
                wind_ms = np.sqrt(u10**2 + v10**2)
                wind_dir = (np.degrees(np.arctan2(-u10, -v10)) + 360) % 360
                
            # Opad
            apcp = get_val(ds_surface, "apcp")
            
            # Helicity
            srh1 = get_val(ds_srh1, "hlcy")
            srh3 = get_val(ds_srh3, "hlcy")

            # Shear (0-1 km i 0-6 km proxy na podstawie 850 hPa i 500 hPa)
            u850 = get_val(ds_850, "u")
            v850 = get_val(ds_850, "v")
            u500 = get_val(ds_500, "u")
            v500 = get_val(ds_500, "v")
            
            shear_01 = np.nan
            shear_06 = np.nan
            if not np.isnan(u10) and not np.isnan(u850) and not np.isnan(v10) and not np.isnan(v850):
                shear_01 = np.sqrt((u850 - u10)**2 + (v850 - v10)**2)
            if not np.isnan(u10) and not np.isnan(u500) and not np.isnan(v10) and not np.isnan(v500):
                shear_06 = np.sqrt((u500 - u10)**2 + (v500 - v10)**2)

            # Kalkulacje finalne (LCL, STP)
            lcl = lcl_height_m(t2m, d2m)
            stp = calculate_stp(cape, lcl, srh1, shear_06)

            run_dt = datetime.strptime(RUN_DATE + RUN_HOUR, "%Y%m%d%H")
            valid_time = run_dt + timedelta(hours=fh)
            
            rows.append({
                "Czas": valid_time,
                "T+ (h)": fh,
                "Temp [°C]": np.round(t2m, 1),
                "Punkt Rosy [°C]": np.round(d2m, 1),
                "CAPE [J/kg]": np.round(cape, 0) if not np.isnan(cape) else np.nan,
                "CIN [J/kg]": np.round(cin, 0) if not np.isnan(cin) else np.nan,
                "LI [°C]": np.round(lftx, 1) if not np.isnan(lftx) else np.nan,
                "LCL [m]": np.round(lcl, 0),
                "SRH 0-1km [m2/s2]": np.round(srh1, 0) if not np.isnan(srh1) else np.nan,
                "SRH 0-3km [m2/s2]": np.round(srh3, 0) if not np.isnan(srh3) else np.nan,
                "Shear 0-1km [m/s]": np.round(shear_01, 1) if not np.isnan(shear_01) else np.nan,
                "Shear 0-6km [m/s]": np.round(shear_06, 1) if not np.isnan(shear_06) else np.nan,
                "STP": stp,
                "PWAT [mm]": np.round(pwat, 1) if not np.isnan(pwat) else np.nan,
                "Wiatr [m/s]": np.round(wind_ms, 1),
                "Porywy [m/s]": np.round(gust, 1) if not np.isnan(gust) else np.nan,
                "Kierunek [°]": np.round(wind_dir, 0),
                "Opad [mm]": np.round(apcp, 1) if not np.isnan(apcp) else np.nan,
                "MSLP [hPa]": np.round(sp, 1)
            })
        except Exception as e:
            print(f" - Błąd przetwarzania pliku {local_path}: {e}")
            continue

    if rows:
        df = pd.DataFrame(rows)
        df.sort_values("Czas", inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        df = pd.DataFrame()
    return df

# -----------------------
# SAVE TO EXCEL / CSV
# -----------------------
def save_outputs(df):
    if df.empty:
        print("⚠️ Brak danych do zapisu.")
        return []
    
    xlsx_path = os.path.join(OUTPUT_DIR, f"krosno_gfs_conv_{RUN_DATE}_{RUN_HOUR}z.xlsx")
    csv_path = os.path.join(OUTPUT_DIR, f"gfs-conv.csv")
    
    # Save CSV
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print("✅ CSV zapisany:", csv_path)
    
    # Save Excel (opcjonalnie, formatowanie)
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Konwekcja", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Konwekcja"]
        
        # Kolorowanie STP (żeby wyłapać rzadkie incydenty)
        if "STP" in df.columns:
            col_idx = df.columns.get_loc("STP")
            rng = f"{chr(65+col_idx)}2:{chr(65+col_idx)}{len(df)+1}"
            fmt_high_stp = workbook.add_format({'bg_color': '#FF9999', 'font_color': '#9C0006', 'bold': True})
            worksheet.conditional_format(rng, {'type': 'cell', 'criteria': '>=', 'value': 1.0, 'format': fmt_high_stp})
            
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max() if len(df)>0 else 0, len(col)) + 2
            worksheet.set_column(i, i, max_len)
            
    print("✅ Excel zapisany:", xlsx_path)
    return [csv_path, xlsx_path]

# -----------------------
# FTP UPLOAD
# -----------------------
def upload_to_ftp(files_to_send):
    load_dotenv()
    host = os.getenv("FTP_HOST")
    user = os.getenv("FTP_USER")
    passwd = os.getenv("FTP_PASS")
    if not all([host, user, passwd]):
        print("⚠️ Brak danych FTP (ENV lub .env) – pomijam wysyłkę.")
        return
    try:
        ftp = FTP(host, user, passwd, timeout=30)
        ftp.cwd("/stacja.meteo-krosno.pl/")
        
        for path in files_to_send:
            if not os.path.exists(path):
                continue
            
            if path.endswith('gfs-conv.csv'):
                # Główny plik CSV, nadpisywany za każdym razem na FTP
                with open(path, "rb") as f:
                    ftp.storbinary(f"STOR gfs-conv.csv", f)
                    print(f"📤 Wysłano na FTP (nadpisano): gfs-conv.csv")
                
                # Zapis do archiwum
                arch_dir = "/stacja.meteo-krosno.pl/archiv"
                try:
                    ftp.cwd(arch_dir)
                except error_perm:
                    ftp.mkd(arch_dir)
                    ftp.cwd(arch_dir)
                
                arch_name = f"gfs_conv_{RUN_DATE[:4]}_{RUN_DATE[4:6]}_{RUN_DATE[6:8]}_{RUN_HOUR}.csv"
                with open(path, "rb") as f:
                    ftp.storbinary(f"STOR {arch_name}", f)
                    print(f"📤 Wysłano na FTP (archiwum): {arch_name}")
                
                ftp.cwd("/stacja.meteo-krosno.pl/")
        
        ftp.quit()
        print("✅ Wszystkie pliki wysłane na FTP.")
    except Exception as e:
        print(f"❌ Błąd podczas wysyłania na FTP: {e}")

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    print(f"\n=== Start GFS Krosno KONWEKCJA {RUN_DATE}{RUN_HOUR}Z ===")
    
    start_time = datetime.utcnow()
    
    while True:
        elapsed_minutes = (datetime.utcnow() - start_time).total_seconds() / 60
        if elapsed_minutes > MAX_TOTAL_WAIT_MINUTES:
            print(f"⏰ Przekroczono maksymalny czas oczekiwania ({MAX_TOTAL_WAIT_MINUTES} min). Kończę.")
            break
        
        print(f"\n🔄 === PRÓBA POBIERANIA ===  (minuta {elapsed_minutes:.0f}/{MAX_TOTAL_WAIT_MINUTES})")
        
        downloaded, missing = download_missing_gribs_parallel(FORECAST_HOURS)
        print(f"   Pobrano łącznie: {len(downloaded)} plików | Brakuje jeszcze: {len(missing)} godzin")
        
        # Przetwarzamy to, co się pobrało
        df = process_local_gribs(FORECAST_HOURS)
        files = save_outputs(df)
        upload_to_ftp(files)
        
        if not missing:
            print("\n✅ Wszystkie pliki pobrane – pełna prognoza konwekcyjna gotowa!")
            break
        
        print(f"   ⏳ Czekam {RETRY_INTERVAL_SECONDS//60} minut na kolejne pliki NOMADS...\n")
        sleep(RETRY_INTERVAL_SECONDS)
    
    print("\n🏁 Skrypt zakończony.\n")
