#!/usr/bin/env python3
# gfs_krosno_conv.py - Poprawiona wersja (bazuje na działającym gfs.py)
import os
import requests
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from time import sleep
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------
# CONFIG
# -----------------------
OUTPUT_DIR = "gfs_krosno_conv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_LAT = 50.0
BOTTOM_LAT = 49.4
LEFT_LON = 21.3
RIGHT_LON = 22.01
KROSNO_LAT = 49.69
KROSNO_LON = 21.77

RETRY_INTERVAL_SECONDS = 10 * 60
MAX_TOTAL_WAIT_MINUTES = 90

# -----------------------
# LOGIKA RUNU
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
FORECAST_HOURS = list(range(0, 384, 3))

# -----------------------
# FILTR POBIERANIA
# -----------------------
STATIC_MIDDLE = (
    "&lev_2_m_above_ground=on"
    "&lev_10_m_above_ground=on"
    "&lev_850_mb=on"
    "&lev_700_mb=on"
    "&lev_500_mb=on"
    "&lev_surface=on"
    "&lev_entire_atmosphere_%28considered_as_a_single_layer%29=on"
    "&lev_mean_sea_level=on"
    "&var_TMP=on"
    "&var_HGT=on"
    "&var_UGRD=on"
    "&var_VGRD=on"
    "&var_CAPE=on"
    "&var_CIN=on"
    "&var_LFTX=on"
    "&var_PWAT=on"
    "&subregion=on"
    f"&toplat={TOP_LAT}"
    f"&bottomlat={BOTTOM_LAT}"
    f"&leftlon={LEFT_LON}"
    f"&rightlon={RIGHT_LON}"
)

def build_url(file_name):
    url = f"{BASE_URL}?file={file_name}&dir=/{CYCLE_DIR}{STATIC_MIDDLE}"
    return url.replace("suubregion", "subregion")

# -----------------------
# POMOCNICZE
# -----------------------
def try_open_by_filter(file_path, filter_by_keys):
    try:
        return xr.open_dataset(file_path, engine="cfgrib", 
                             backend_kwargs={"filter_by_keys": filter_by_keys, "indexpath": ""})
    except Exception:
        return None


def safe_get_point(ds, possible_names):
    if ds is None:
        return np.nan
    for name in possible_names:
        if name in ds.data_vars:
            try:
                val = ds[name].sel(latitude=KROSNO_LAT, longitude=KROSNO_LON, method="nearest")
                return float(np.squeeze(np.array(val)))
            except:
                continue
    return np.nan


# -----------------------
# PRZETWARZANIE
# -----------------------
def process_local_gribs(forecast_hours):
    rows = []
    for fh in forecast_hours:
        path = os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")
        if not os.path.exists(path):
            continue

        try:
            # Kluczowe datasety
            ds_sfc_inst = try_open_by_filter(path, {"typeOfLevel": "surface", "stepType": "instant"})
            ds_2m       = try_open_by_filter(path, {"typeOfLevel": "heightAboveGround", "level": 2})
            ds_10m      = try_open_by_filter(path, {"typeOfLevel": "heightAboveGround", "level": 10})
            ds_pwat     = try_open_by_filter(path, {"typeOfLevel": "atmosphereSingleLayer"})
            ds_700      = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa", "level": 700})
            ds_500      = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa", "level": 500})

            # Podstawowe parametry
            t2m  = safe_get_point(ds_2m,  ['t2m', '2t', 'TMP']) - 273.15
            cape = safe_get_point(ds_sfc_inst, ['cape', 'CAPE'])
            cin  = safe_get_point(ds_sfc_inst, ['cin', 'CIN'])
            li   = safe_get_point(ds_sfc_inst, ['lftx', 'LFTX'])
            pwat = safe_get_point(ds_pwat, ['pwat', 'PWAT'])

            # Poziomy dla gradientu i uskoków
            t700 = safe_get_point(ds_700, ['t', 'TMP']) - 273.15
            t500 = safe_get_point(ds_500, ['t', 'TMP']) - 273.15
            h700 = safe_get_point(ds_700, ['gh', 'HGT'])
            h500 = safe_get_point(ds_500, ['gh', 'HGT'])

            u10  = safe_get_point(ds_10m, ['u10', '10u', 'UGRD'])
            v10  = safe_get_point(ds_10m, ['v10', '10v', 'VGRD'])
            u500 = safe_get_point(ds_500, ['u', 'UGRD'])
            v500 = safe_get_point(ds_500, ['v', 'VGRD'])

            # Obliczenia
            dls = np.sqrt((u500 - u10)**2 + (v500 - v10)**2) if not np.isnan(u500) and not np.isnan(u10) else np.nan
            lr_700_500 = (t700 - t500) / ((h500 - h700)/1000) if all(not np.isnan(x) for x in [t700,t500,h700,h500]) else np.nan

            # Wysokość 0°C
            zero_deg_h = np.nan
            if not np.isnan(t2m) and not np.isnan(h700) and not np.isnan(t700):
                if t700 <= 0:
                    zero_deg_h = t2m * h700 / (t2m - t700)
                elif not np.isnan(t500) and not np.isnan(h500):
                    zero_deg_h = h700 + t700 * (h500 - h700) / (t700 - t500)

            # LCL (przybliżenie)
            d2m = safe_get_point(ds_2m, ['d2m', '2d', 'DPT']) - 273.15
            lcl = 125 * (t2m - d2m) if not np.isnan(t2m) and not np.isnan(d2m) else np.nan

            prob = calc_storm_prob(cape, cin, li, dls, pwat)
            hail = estimate_hail_size(cape, lr_700_500, dls)

            rows.append({
                "Czas": datetime.strptime(RUN_DATE + RUN_HOUR, "%Y%m%d%H") + timedelta(hours=fh),
                "T+": fh,
                "T2M [°C]": round(t2m, 1),
                "CAPE [J/kg]": int(round(cape, 0)) if not np.isnan(cape) else 0,
                "CIN [J/kg]": int(round(cin, 0)) if not np.isnan(cin) else 0,
                "LI [°C]": round(li, 1),
                "DLS 0-6km [m/s]": round(dls, 1),
                "LR 700-500 [C/km]": round(lr_700_500, 1),
                "0°C Height [m]": round(zero_deg_h, 0),
                "PWAT [mm]": round(pwat, 1),
                "LCL [m]": round(lcl, 0),
                "Prob Burzy [%]": prob,
                "Grad [cm]": hail
            })

        except Exception as e:
            print(f"Błąd przetwarzania f{fh:03d}: {e}")
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"Przetworzono {len(df)} rekordów | Przykładowe CAPE: {df['CAPE [J/kg]'].iloc[0]}")
    return df


# Algorytmy (bez zmian)
def calc_storm_prob(cape, cin, li, dls, pwat):
    if np.isnan(cape) or cape < 50: return 0.0
    score = (cape / 1500.0) * 40.0
    if not np.isnan(cin):
        if cin > -20: score += 20
        elif cin < -100: score -= 30
    if not np.isnan(dls) and dls > 15: score += 20
    if not np.isnan(pwat) and pwat > 30: score += 15
    if not np.isnan(li) and li < -4: score += 10
    return float(np.clip(np.round(score, 0), 0, 100))


def estimate_hail_size(cape, lr, dls):
    if np.isnan(cape) or cape < 400: return 0.0
    hail = (cape / 1000.0) * (lr / 6.5 if not np.isnan(lr) else 1.0)
    if not np.isnan(dls) and dls > 20: hail *= 1.3
    return float(np.round(np.clip(hail, 0, 8), 1))


# -----------------------
# POBIERANIE + ZAPIS + FTP (bez zmian z poprzedniej wersji)
# -----------------------
def download_missing_gribs_parallel(forecast_hours):
    pending = [fh for fh in forecast_hours 
               if not os.path.exists(os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")) 
               or os.path.getsize(os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")) < 40000]

    if not pending:
        return

    print(f" → Pobieranie {len(pending)} plików...")

    def fetch_single(fh):
        fstr = f"{fh:03d}"
        grib_filename = f"gfs.t{RUN_HOUR}z.pgrb2.0p25.f{fstr}"
        local_path = os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fstr}.grib2")
        url = build_url(grib_filename)
        try:
            r = requests.get(url, headers=HEADERS, timeout=90)
            if r.status_code == 200 and b"GRIB" in r.content[:10]:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                print(f" ✓ f{fh:03d} OK")
                return True
        except:
            pass
        return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(fetch_single, pending)


def save_outputs(df):
    if df.empty: return []
    csv_path = os.path.join(OUTPUT_DIR, "gfs-conv.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')

    xlsx_path = os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z.xlsx")
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Burze')
        # formatowanie...
        workbook = writer.book
        ws = writer.sheets['Burze']
        fmt_red = workbook.add_format({'bg_color': '#FF3333', 'font_color': 'white'})
        ws.conditional_format('D2:D200', {'type': 'cell', 'criteria': '>=', 'value': 1000, 'format': fmt_red})
        ws.conditional_format('M2:M200', {'type': 'cell', 'criteria': '>=', 'value': 1.5, 'format': fmt_red})

    return [csv_path, xlsx_path]


def upload_to_ftp(files):
    load_dotenv()
    host = os.getenv("FTP_HOST")
    user = os.getenv("FTP_USER")
    pswd = os.getenv("FTP_PASS")
    if not all([host, user, pswd]): return
    try:
        from ftplib import FTP
        ftp = FTP(host, user, pswd, timeout=30)
        ftp.cwd("/stacja.meteo-krosno.pl/")
        for p in files:
            target = "gfs-conv.csv" if p.endswith('.csv') else os.path.basename(p)
            with open(p, "rb") as f:
                ftp.storbinary(f"STOR {target}", f)
            if p.endswith('.csv'):
                try: ftp.cwd("/stacja.meteo-krosno.pl/archiv_conv")
                except: 
                    ftp.mkd("/stacja.meteo-krosno.pl/archiv_conv")
                    ftp.cwd("/stacja.meteo-krosno.pl/archiv_conv")
                arch_name = f"gfs_conv_{RUN_DATE}_{RUN_HOUR}.csv"
                with open(p, "rb") as f:
                    ftp.storbinary(f"STOR {arch_name}", f)
                ftp.cwd("/stacja.meteo-krosno.pl/")
        ftp.quit()
        print("✅ FTP OK")
    except Exception as e:
        print(f"❌ FTP Error: {e}")


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    print(f"\n=== GFS CONVECTION MODULE {RUN_DATE}{RUN_HOUR}Z ===")
    start_time = datetime.utcnow()

    while True:
        elapsed = (datetime.utcnow() - start_time).total_seconds() / 60
        if elapsed > MAX_TOTAL_WAIT_MINUTES: break

        download_missing_gribs_parallel(FORECAST_HOURS)
        df = process_local_gribs(FORECAST_HOURS)

        if not df.empty:
            files = save_outputs(df)
            upload_to_ftp(files)

        missing = [fh for fh in FORECAST_HOURS if not os.path.exists(os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2"))]
        if not missing:
            print("Wszystko pobrane i przetworzone.")
            break

        print(f"⏳ Czekam 10 min na brakujące pliki...")
        sleep(RETRY_INTERVAL_SECONDS)
