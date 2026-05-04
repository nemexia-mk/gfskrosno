#!/usr/bin/env python3
# gfs_conv_krosno.py
# Wersja konwekcyjna / burzowa (Storm Chasing)
# Pobiera GFS, generuje parametry burzowe (CAPE, CIN, LCL, SRH, DLS, LLS, STP, SCP)
# Wylicza zaawansowane prawdopodobieństwo burzy i zapisuje gfs-conv.csv + Excel.

import os
import sys
import requests
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from time import sleep
from dotenv import load_dotenv
from ftplib import FTP, error_perm
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
FORECAST_HOURS = list(range(0, 123, 3)) # Skracamy do 120h - najbardziej sensowne dla konwekcji

RETRY_INTERVAL_SECONDS = 10 * 60
MAX_TOTAL_WAIT_MINUTES = 90

# Static NOMADS filter - Zmodyfikowany pod konwekcję
STATIC_MIDDLE = (
    "&lev_2_m_above_ground=on"
    "&lev_10_m_above_ground=on"
    "&lev_500_mb=on"
    "&lev_850_mb=on"
    "&lev_surface=on"
    "&lev_entire_atmosphere_%28considered_as_a_single_layer%29=on"
    "&lev_1000-0_m_above_ground=on"
    "&lev_3000-0_m_above_ground=on"
    "&var_TMP=on"
    "&var_DPT=on"
    "&var_UGRD=on"
    "&var_VGRD=on"
    "&var_CAPE=on"
    "&var_CIN=on"
    "&var_LFTX=on"
    "&var_HLCY=on"
    "&var_PWAT=on"
    "&var_PRATE=on"
    "&var_GUST=on"
    "&subregion=on"
    f"&toplat={TOP_LAT}"
    f"&bottomlat={BOTTOM_LAT}"
    f"&leftlon={LEFT_LON}"
    f"&rightlon={RIGHT_LON}"
)

# -----------------------
# HELPERS & ALGORITHMS
# -----------------------
def build_url(file_name):
    url = f"{BASE_URL}?file={file_name}&dir=/{CYCLE_DIR}{STATIC_MIDDLE}"
    return url.replace("suubregion", "subregion").replace("lev_entire_atmoosphere", "lev_entire_atmosphere")

SHORTNAMES = {
    "t2m": ["t2m", "2t", "tmp2m", "tmp"],
    "d2m": ["d2m", "dew2m", "dpt"],
    "u": ["u", "ugrd"],
    "v": ["v", "vgrd"],
    "u10": ["ugrd", "u10"],
    "v10": ["vgrd", "v10"],
    "cape": ["cape", "sbcape"],
    "cin": ["cin"],
    "lftx": ["lftx"],
    "pwat": ["pwat"],
    "hlcy": ["hlcy"],
    "prate": ["prate"],
    "gust": ["gust"]
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

def convert_and_round(val, name):
    if val is None or np.isnan(val): return np.nan
    if name in ("t2m","d2m"): return float(np.round(val - 273.15, 1))
    if name == "prate": return float(np.round(val * 3600.0 * 3.0, 1))
    if name in ("cape", "cin", "hlcy"): return float(np.round(val, 0))
    if name == "pwat": return float(np.round(val, 1)) # mm
    return float(np.round(val, 2))

def lcl_height_m(t_c, td_c):
    if np.isnan(t_c) or np.isnan(td_c): return np.nan
    diff = max(t_c - td_c, 0.0)
    return float(np.round(125.0 * diff, 1))

def calc_wind_shear(u_top, v_top, u_bot, v_bot):
    if any(np.isnan(x) for x in [u_top, v_top, u_bot, v_bot]): return np.nan
    diff_u = u_top - u_bot
    diff_v = v_top - v_bot
    return float(np.round(np.sqrt(diff_u**2 + diff_v**2), 1))

# --- KOMPLEKSOWY ALGORYTM PRAWDOPODOBIEŃSTWA BURZY ---
def calculate_storm_probability(cape, cin, li, dls, pwat, prate):
    if np.isnan(cape) or cape < 50:
        return 0
    
    prob = 0.0
    # Term 1: Termodynamika (CAPE) - max 40 punktów
    prob += min(cape / 2000.0, 1.0) * 40.0
    
    # Term 2: Wspomaganie (Lifted Index) - max 20 punktów
    if not np.isnan(li) and li < 0:
        prob += min(abs(li) / 6.0, 1.0) * 20.0
        
    # Term 3: Kinematyka / Uskoki (DLS) - max 15 punktów
    if not np.isnan(dls):
        prob += min(dls / 25.0, 1.0) * 15.0
        
    # Term 4: Wilgotność (PWAT) - max 15 punktów
    if not np.isnan(pwat):
        prob += min(pwat / 40.0, 1.0) * 15.0
        
    # Term 5: Obecność konwekcji (Opad modelem) - bonus 10 punktów
    if not np.isnan(prate) and prate > 0:
        prob += 10.0
        
    # PENALTY: CIN (Inhibicja) - odejmujemy punkty jeśli warstwa hamująca jest silna
    if not np.isnan(cin) and cin != 0:
        cin_val = abs(cin)
        penalty = min(cin_val / 100.0, 1.0) * 60.0
        prob -= penalty
        
    return int(max(0, min(100, prob)))

def calc_stp(cape, lcl, srh1, lls):
    """Significant Tornado Parameter (Effective)"""
    if any(np.isnan(x) for x in [cape, lcl, srh1, lls]): return np.nan
    if cape < 100: return 0.0
    
    cape_term = cape / 1500.0
    lcl_term = max(0.0, min(1.0, (2000.0 - lcl) / 1000.0))
    srh_term = srh1 / 150.0
    lls_term = max(0.0, min(1.5, lls / 20.0))
    
    stp = cape_term * lcl_term * srh_term * lls_term
    return float(np.round(stp, 2))

def calc_scp(cape, srh3, dls):
    """Supercell Composite Parameter"""
    if any(np.isnan(x) for x in [cape, srh3, dls]): return np.nan
    if cape < 100: return 0.0
    
    mu_cape = cape / 1000.0
    srh_term = srh3 / 50.0
    shear_term = max(0.0, min(1.5, dls / 20.0))
    
    scp = mu_cape * srh_term * shear_term
    return float(np.round(scp, 2))

def handle_404_and_exit():
    print("❌ Błąd 404 dla f000 - przerywam skrypt.")
    sys.exit(0)

# -----------------------
# PARALLEL DOWNLOAD
# -----------------------
def download_missing_gribs_parallel(forecast_hours):
    pending = []
    downloaded = []
    for fh in forecast_hours:
        local_path = os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")
        if os.path.exists(local_path) and os.path.getsize(local_path) > 50000:
            downloaded.append(local_path)
            continue
        pending.append(fh)

    if not pending:
        return downloaded, []

    print(f"   → Równoległe pobieranie {len(pending)} plików (T+{min(pending)}–{max(pending)})")

    def fetch_single(fh):
        fstr = f"{fh:03d}"
        grib_filename = f"gfs.t{RUN_HOUR}z.pgrb2.0p25.f{fstr}"
        local_path = os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fstr}.grib2")
        url = build_url(grib_filename)

        try:
            r = requests.get(url, headers=HEADERS, timeout=90)
            if r.status_code == 404: return fh, None, "404"
            if r.status_code != 200 or b"GRIB" not in r.content[:10]: return fh, None, f"HTTP {r.status_code}"
            with open(local_path, "wb") as f:
                f.write(r.content)
            return fh, local_path, None
        except Exception as e:
            return fh, None, str(e)

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_fh = {executor.submit(fetch_single, fh): fh for fh in pending}
        for future in as_completed(future_to_fh):
            fh, path, error = future.result()
            if path: downloaded.append(path)
            if error == "404" and fh == 0: handle_404_and_exit()

    still_missing = [fh for fh in forecast_hours if not (os.path.exists(os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")) and os.path.getsize(os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")) > 50000)]
    return downloaded, still_missing

# -----------------------
# PROCESSING LOGIC
# -----------------------
def process_local_gribs(forecast_hours):
    rows = []
    for fh in forecast_hours:
        local_path = os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")
        if not os.path.exists(local_path): continue
        
        ds_surf_inst = try_open_by_filter(local_path, {"typeOfLevel": "surface", "stepType": "instant"})
        ds_surf_acc = try_open_by_filter(local_path, {"typeOfLevel": "surface", "stepType": "accum"})
        ds_2m = try_open_by_filter(local_path, {"typeOfLevel": "heightAboveGround", "level": 2})
        ds_10m = try_open_by_filter(local_path, {"typeOfLevel": "heightAboveGround", "level": 10})
        ds_850 = try_open_by_filter(local_path, {"typeOfLevel": "isobaricInhPa", "level": 850})
        ds_500 = try_open_by_filter(local_path, {"typeOfLevel": "isobaricInhPa", "level": 500})
        ds_pwat = try_open_by_filter(local_path, {"typeOfLevel": "atmosphereSingleLayer"})
        
        # Helicity w GFS występuje na warstwach (zazwyczaj indeksowanie za pomocą parametrów top/bottom level w cfgrib)
        ds_srh1 = try_open_by_filter(local_path, {"typeOfLevel": "heightAboveGroundLayer", "topLevel": 1000})
        ds_srh3 = try_open_by_filter(local_path, {"typeOfLevel": "heightAboveGroundLayer", "topLevel": 3000})
        
        def get_v(ds, key): return safe_get_point(ds, SHORTNAMES.get(key, [key]))

        try:
            t2m = convert_and_round(get_v(ds_2m, "t2m"), "t2m")
            d2m = convert_and_round(get_v(ds_2m, "d2m"), "d2m")
            cape = convert_and_round(get_v(ds_surf_inst, "cape"), "cape")
            cin = convert_and_round(get_v(ds_surf_inst, "cin"), "cin")
            li = convert_and_round(get_v(ds_surf_inst, "lftx"), "lftx")
            pwat = convert_and_round(get_v(ds_pwat, "pwat"), "pwat")
            
            prate = convert_and_round(get_v([ds_surf_acc, ds_surf_inst], "prate"), "prate")
            
            # Wind kinematics
            u10, v10 = get_v(ds_10m, "u10"), get_v(ds_10m, "v10")
            u850, v850 = get_v(ds_850, "u"), get_v(ds_850, "v")
            u500, v500 = get_v(ds_500, "u"), get_v(ds_500, "v")
            
            wspd10 = np.nan if np.isnan(u10) else convert_and_round(np.sqrt(u10**2 + v10**2), "wspd")
            gust = convert_and_round(get_v([ds_surf_inst, ds_10m], "gust"), "gust")
            
            dls = calc_wind_shear(u500, v500, u10, v10)  # 0-6 km Shear (m/s)
            lls = calc_wind_shear(u850, v850, u10, v10)  # 0-1 km Shear (m/s)
            
            srh1 = convert_and_round(get_v(ds_srh1, "hlcy"), "hlcy")
            srh3 = convert_and_round(get_v(ds_srh3, "hlcy"), "hlcy")
            # Fallback jeśli NOMADS zwraca HLcy w innej strukturze kluczy:
            if np.isnan(srh3): srh3 = convert_and_round(get_v(ds_surf_inst, "hlcy"), "hlcy") 
            
            lcl = lcl_height_m(t2m, d2m)
            
            prob = calculate_storm_probability(cape, cin, li, dls, pwat, prate)
            stp = calc_stp(cape, lcl, srh1 if not np.isnan(srh1) else 0, lls)
            scp = calc_scp(cape, srh3 if not np.isnan(srh3) else 0, dls)

            run_dt = datetime.strptime(RUN_DATE + RUN_HOUR, "%Y%m%d%H")
            valid_time = run_dt + timedelta(hours=fh)
            
            rows.append({
                "Czas": valid_time,
                "T+ (h)": fh,
                "T2M [°C]": t2m,
                "Td [°C]": d2m,
                "PWAT [mm]": pwat,
                "RRR [mm/3h]": prate,
                "Wiatr [m/s]": wspd10,
                "Porywy [m/s]": gust,
                "CAPE [J/kg]": cape,
                "CIN [J/kg]": abs(cin) if not np.isnan(cin) else np.nan,
                "LI [°C]": li,
                "LCL [m]": lcl,
                "DLS 0-6km [m/s]": dls,
                "LLS 0-1km [m/s]": lls,
                "SRH 0-1km": srh1,
                "SRH 0-3km": srh3,
                "SCP": scp,
                "STP": stp,
                "Prawd. Burzy [%]": prob
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
# SAVE & EXPORT
# -----------------------
def save_outputs(df):
    if df.empty: return []
    
    # 1. Zapis CSV
    csv_path = os.path.join(OUTPUT_DIR, f"krosno_gfs_conv_{RUN_DATE}_{RUN_HOUR}z.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print("✅ CSV konwekcyjny zapisany:", csv_path)
    
    # 2. Zapis Excel z kolorowaniem
    xlsx_path = os.path.join(OUTPUT_DIR, f"krosno_gfs_conv_{RUN_DATE}_{RUN_HOUR}z.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="konwekcja", index=False)
        workbook = writer.book
        ws = writer.sheets["konwekcja"]
        
        # Style
        border_fmt = workbook.add_format({'border': 1, 'align': 'center'})
        ws.conditional_format(f'A1:{chr(65 + len(df.columns) - 1)}{len(df) + 1}', {'type': 'no_blanks', 'format': border_fmt})
        
        # Color Scales
        def apply_color_scale(col_name, min_val, mid_val, max_val, min_color, mid_color, max_color):
            if col_name in df.columns:
                col_idx = df.columns.get_loc(col_name)
                rng = f"{chr(65+col_idx)}2:{chr(65+col_idx)}{len(df)+1}"
                ws.conditional_format(rng, {
                    'type': '3_color_scale',
                    'min_value': min_val, 'mid_value': mid_val, 'max_value': max_val,
                    'min_type': 'num', 'mid_type': 'num', 'max_type': 'num',
                    'min_color': min_color, 'mid_color': mid_color, 'max_color': max_color
                })

        apply_color_scale("CAPE [J/kg]", 0, 1000, 2500, "#FFFFFF", "#FFFF00", "#FF0000") # Biały -> Żółty -> Czerwony
        apply_color_scale("Prawd. Burzy [%]", 0, 40, 80, "#FFFFFF", "#FFFF00", "#FF0000") 
        apply_color_scale("STP", 0.0, 1.0, 3.0, "#FFFFFF", "#FFB6C1", "#800080") # Biały -> Różowy -> Fioletowy
        apply_color_scale("SCP", 0.0, 2.0, 8.0, "#FFFFFF", "#ADD8E6", "#00008B") # Biały -> Jasnoniebieski -> Ciemnoniebieski
        apply_color_scale("LI [°C]", -6, -2, 0, "#FF0000", "#FFFF00", "#FFFFFF") # Czerwony dla < -6
        apply_color_scale("DLS 0-6km [m/s]", 10, 15, 25, "#FFFFFF", "#D3D3D3", "#8A2BE2") # Ścinanie
        
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max() if len(df)>0 else 0, len(col)) + 2
            ws.set_column(i, i, max_len)
            
    print("✅ Excel konwekcyjny zapisany:", xlsx_path)
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
        print("⚠️ Brak danych FTP – pomijam wysyłkę.")
        return
    try:
        ftp = FTP(host, user, passwd, timeout=30)
        ftp.cwd("/stacja.meteo-krosno.pl/")
        
        for path in files_to_send:
            if not os.path.exists(path): continue
            
            if path.endswith('.csv'):
                with open(path, "rb") as f:
                    ftp.storbinary("STOR gfs-conv.csv", f)
                    print("📤 Wysłano na FTP (nadpisano): gfs-conv.csv")
                
                arch_dir = "/stacja.meteo-krosno.pl/archiv"
                try: ftp.cwd(arch_dir)
                except error_perm:
                    ftp.mkd(arch_dir)
                    ftp.cwd(arch_dir)
                
                arch_name = f"gfs_conv_{RUN_DATE}_{RUN_HOUR}.csv"
                with open(path, "rb") as f:
                    ftp.storbinary(f"STOR {arch_name}", f)
                    print(f"📤 Wysłano do archiwum: {arch_name}")
                
                ftp.cwd("/stacja.meteo-krosno.pl/")
            else:
                fname = os.path.basename(path)
                with open(path, "rb") as f:
                    ftp.storbinary(f"STOR {fname}", f)
                    print(f"📤 Wysłano na FTP: {fname}")
        ftp.quit()
        print("✅ Wszystkie pliki wysłane.")
    except Exception as e:
        print(f"❌ Błąd FTP: {e}")

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    print(f"\n=== Start GFS CONV Krosno {RUN_DATE}{RUN_HOUR}Z ===")
    start_time = datetime.utcnow()
    
    while True:
        elapsed_minutes = (datetime.utcnow() - start_time).total_seconds() / 60
        if elapsed_minutes > MAX_TOTAL_WAIT_MINUTES:
            print("⏰ Koniec czasu oczekiwania.")
            break
            
        print(f"\n🔄 === PRÓBA POBIERANIA === ({elapsed_minutes:.0f} min)")
        downloaded, missing = download_missing_gribs_parallel(FORECAST_HOURS)
        print(f"   Pobrano: {len(downloaded)} | Brakuje: {len(missing)}")
        
        df = process_local_gribs(FORECAST_HOURS)
        files = save_outputs(df)
        upload_to_ftp(files)
        
        if not missing:
            print("\n✅ Pełna prognoza burzowa gotowa!")
            break
            
        print(f"   ⏳ Czekam {RETRY_INTERVAL_SECONDS//60} minut...")
        sleep(RETRY_INTERVAL_SECONDS)
