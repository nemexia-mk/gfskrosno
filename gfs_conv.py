#!/usr/bin/env python3
# gfs_conv_v17_ultimate.py - KROSNO (Poprawki: 280m n.p.m., Tatry, Derecho, Storm Mode, Bugfix 2KB)
import os
import requests
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from time import sleep
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# Moduł Asynchroniczny
try:
    import asyncio
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    print("[INFO] Biblioteka aiohttp brakująca. Użyję wolniejszego pobierania sekwencyjnego.", flush=True)

try:
    import metpy.calc as mpcalc
    from metpy.units import units
    METPY_AVAILABLE = True
except ImportError:
    METPY_AVAILABLE = False
    print("[INFO] Biblioteka MetPy nie jest zainstalowana. Używam niezawodnych wzorów awaryjnych.", flush=True)

OUTPUT_DIR = "gfs_krosno_conv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_LAT = 50.0
BOTTOM_LAT = 49.4
LEFT_LON = 21.3
RIGHT_LON = 22.01
KROSNO_LAT = 49.69
KROSNO_LON = 21.77
KROSNO_ELEVATION_M = 280.0
RETRY_INTERVAL_SECONDS = 120
MAX_TOTAL_WAIT_MINUTES = 120

# ==================== LOGIKA CZASU ====================
now = datetime.utcnow()
current_time = now.time()

if time(3, 30) <= current_time < time(9, 30):
    RUN_HOUR = "00"
    RUN_DATE = now.strftime("%Y%m%d")
elif time(9, 30) <= current_time < time(15, 30):
    RUN_HOUR = "06"
    RUN_DATE = now.strftime("%Y%m%d")
elif time(15, 30) <= current_time < time(21, 30):
    RUN_HOUR = "12"
    RUN_DATE = now.strftime("%Y%m%d")
else:
    RUN_HOUR = "18"
    if current_time >= time(21, 30):
        RUN_DATE = now.strftime("%Y%m%d")
    else: 
        RUN_DATE = (now - timedelta(days=1)).strftime("%Y%m%d")

CYCLE_DIR = f"gfs.{RUN_DATE}/{RUN_HOUR}/atmos"
BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

FORECAST_HOURS = list(range(0, 121, 1)) + list(range(123, 385, 3))

STATIC_MIDDLE = (
    "&lev_2_m_above_ground=on&lev_10_m_above_ground=on"
    "&lev_850_mb=on&lev_700_mb=on&lev_500_mb=on&lev_925_mb=on&lev_1000_mb=on"
    "&lev_surface=on&lev_entire_atmosphere_%28considered_as_a_single_layer%29=on"
    "&lev_entire_atmosphere=on" 
    "&var_TMP=on&var_HGT=on&var_UGRD=on&var_VGRD=on&var_CAPE=on&var_CIN=on"
    "&var_LFTX=on&var_PWAT=on&var_HLCY=on&var_DPT=on&var_RH=on&var_SPFH=on&var_VVEL=on&var_APCP=on&var_PRATE=on"
    "&var_PRES=on"
    "&subregion=on"
    f"&toplat={TOP_LAT}&bottomlat={BOTTOM_LAT}&leftlon={LEFT_LON}&rightlon={RIGHT_LON}"
)

def build_url(file_name):
    return f"{BASE_URL}?file={file_name}&dir=/{CYCLE_DIR}{STATIC_MIDDLE}"

def try_open_by_filter(file_path, filter_by_keys):
    try:
        return xr.open_dataset(file_path, engine="cfgrib", backend_kwargs={"filter_by_keys": filter_by_keys, "indexpath": ""})
    except:
        return None

def safe_get_point(ds, possible_names):
    if ds is None: return np.nan
    for name in possible_names:
        if name in ds.data_vars:
            try:
                val = ds[name].sel(latitude=KROSNO_LAT, longitude=KROSNO_LON, method="nearest")
                return float(np.squeeze(np.array(val)))
            except: continue
    return np.nan

# ==================== ALGORYTMY ====================
def calc_rh(t_c, td_c):
    if np.isnan(t_c) or np.isnan(td_c): return np.nan
    e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    return min(max((e / es) * 100.0, 0.0), 100.0)

def calc_tatry_visibility(t2m, td2m, rh700, pwat, wdir10m, wspd10m, prate):
    if not np.isnan(prate) and prate > 0.1: return "Brak (Opad)"
    rh2m = calc_rh(t2m, td2m)
    if np.isnan(rh2m) or np.isnan(pwat): return ""
    
    score = 100.0 - (rh2m - 30.0) * 1.5 - (pwat * 2.5)
    
    if not np.isnan(wdir10m) and not np.isnan(wspd10m):
        if 135 <= wdir10m <= 225 and wspd10m > 5.0:
            score += 30 # Silny Halny oczyszcza dolną troposferę
        elif 290 <= wdir10m <= 360:
            score += 15 # Masa polarna-morska / arktyczna z północnego zachodu
            
    if not np.isnan(rh700) and rh700 > 75: 
        score -= 40 # Chmury piętra średniego zasłaniające góry
        
    score = np.clip(score, 0, 100)
    
    if score < 20: desc = "Brak"
    elif score < 45: desc = "Słaba (~50km)"
    elif score < 75: desc = "Dobra (~100km)"
    else: desc = "Wybitna (Tatry!)"
    
    return f"{int(score)}% ({desc})"

def estimate_storm_motion(u10, v10, u500, v500):
    if any(np.isnan(x) for x in [u10, v10, u500, v500]): return np.nan, np.nan
    u_mean, v_mean = (u10 + u500) / 2.0, (v10 + v500) / 2.0
    du, dv = u500 - u10, v500 - v10
    shear_mag = np.hypot(du, dv)
    if shear_mag < 1.0: return u_mean, v_mean
    perp_u, perp_v = dv, -du
    scale = 7.5 / (np.hypot(perp_u, perp_v) + 1e-6)
    return u_mean + perp_u * scale, v_mean + perp_v * scale

def calc_srh_layer(u_b, v_b, u_t, v_t, u_s, v_s):
    if any(np.isnan(x) for x in [u_b, v_b, u_t, v_t, u_s, v_s]): return np.nan
    return float(np.round((u_b - u_s)*(v_t - v_b) - (v_b - v_s)*(u_t - u_b), 1))

def wind_direction(u, v):
    if np.isnan(u) or np.isnan(v): return np.nan
    return round((270 - np.rad2deg(np.arctan2(v, u))) % 360, 1)

def is_foehn_wind(wdir):
    if np.isnan(wdir): return False
    return 145.0 <= wdir <= 235.0

def calc_brn(cape, dls_06):
    if np.isnan(cape) or np.isnan(dls_06) or dls_06 < 1 or cape <= 0: return np.nan
    return float(np.round(cape / (0.5 * dls_06 ** 2), 1))

def calc_stp(cape, srh1, dls06, lcl):
    if any(np.isnan(x) for x in [cape, srh1, dls06, lcl]): return np.nan
    return float(round(max(0, min((cape/1500)*(srh1/150)*(dls06/20)*max(0, (2000-lcl)/1000), 8.0)), 2))

def supercell_rotation_type(srh3):
    if np.isnan(srh3): return ""
    if srh3 > 50: return "Prawoskrętna"
    elif srh3 < -30: return "Lewoskrętna"
    return ""

def calc_dcp(mucape, dcape, dls06, u10, v10, u850, v850, u700, v700, u500, v500):
    if any(np.isnan(x) for x in [mucape, dcape, dls06]): return np.nan
    if mucape < 100 or dcape < 100: return np.nan
    wsfc = np.hypot(u10, v10) if not np.isnan(u10) else 0
    w850 = np.hypot(u850, v850) if not np.isnan(u850) else wsfc
    w700 = np.hypot(u700, v700) if not np.isnan(u700) else w850
    w500 = np.hypot(u500, v500) if not np.isnan(u500) else w700
    mean_wind_knots = ((wsfc + w850 + w700 + w500) / 4.0) * 1.94384
    dls_knots = dls06 * 1.94384
    dcp = (dcape / 980.0) * (mucape / 2000.0) * (dls_knots / 40.0) * (mean_wind_knots / 32.0)
    return float(np.round(dcp, 2))

def calc_lightning_rate(cape, lcl, cin):
    if np.isnan(cape) or cape < 150: return np.nan
    if not np.isnan(cin) and cin < -150: return np.nan 
    rate = (cape / 600.0) ** 1.3
    if not np.isnan(lcl) and lcl < 2000:
        rate *= (2000 / max(lcl, 400)) 
    return float(np.round(min(rate, 120.0), 1))

def calc_storm_prob(cape, cin, li, dls06, dls01, srh3, srh1, pwat, lcl, lr, brn, foehn=False):
    if np.isnan(cape) or cape < 50: return np.nan
    score = min(cape / 1200.0, 1.0) * 35
    if not np.isnan(cin):
        if -50 < cin < 0: score += 15
        elif cin < -150: score -= 20
    if not np.isnan(li):
        if li < -4: score += 12
    if not np.isnan(dls06) and dls06 > 15: score += 10
    if not np.isnan(srh3) and srh3 > 100: score += 8
    if not np.isnan(pwat) and pwat > 25: score += 6
    val = float(np.clip(np.round(score, 0), 0, 100))
    return val if val > 0 else np.nan

def calc_supercell_risk(cape, dls06, srh3, brn, li, dls01, srh1):
    if np.isnan(cape) or cape < 200: return np.nan
    score = 0.0
    if cape > 800: score += 20
    if not np.isnan(dls06) and dls06 > 15: score += 25
    if not np.isnan(srh3) and srh3 > 100: score += 25
    val = float(np.clip(np.round(score, 0), 0, 100))
    return val if val > 0 else np.nan

def estimate_hail_size(cape, lr, dls):
    if np.isnan(cape) or cape < 300: return np.nan
    hail = (cape / 800.0) * (lr / 6.5 if not np.isnan(lr) else 1.0)
    if not np.isnan(dls) and dls > 15: hail *= (1.0 + (dls - 15) * 0.02) 
    if hail < 0.5: return np.nan
    return float(np.round(np.clip(hail, 0, 10), 1))

def calc_ship(mucape, mu_mixing_ratio, lr_700_500, t500, dls06):
    if any(np.isnan(x) for x in [mucape, mu_mixing_ratio, lr_700_500, t500, dls06]): return np.nan
    val = round(max(0, min((mucape * mu_mixing_ratio * lr_700_500 * abs(t500) * dls06) / 44000000, 5.0)), 2)
    return val if val > 0 else np.nan

def calc_ehi(cape, srh):
    val = round((cape * srh) / 160000, 2) if not np.isnan(cape) and not np.isnan(srh) and cape > 0 else np.nan
    return val if val > 0 else np.nan

def calc_full_stp(sbcape, lcl_h, srh01, dls06, sbcin):
    if any(np.isnan(x) for x in [sbcape, lcl_h, srh01, dls06, sbcin]): return np.nan
    val = round(max(0, min((min(sbcape/1500,1.5) * max(0,(2000-lcl_h)/1000) * min(srh01/150,1.5) * min(dls06/20,1.5) * max(0,(200+sbcin)/150)), 8.0)), 2)
    return val if val > 0 else np.nan

def calc_tornado_prob(stp, lcl, srh01, dls01):
    if any(np.isnan(x) for x in [stp, lcl, srh01, dls01]): return np.nan
    if stp <= 0.1: return np.nan
    score = stp * 15.0
    if lcl < 800: score += 20
    elif lcl > 1200: score -= 20
    if srh01 > 100: score += 15
    if dls01 > 10: score += 15
    val = float(np.clip(round(score, 0), 0, 100))
    return val if val > 0 else np.nan

def calc_wind_risk(dcape, rh700, dls06, cape):
    base_dcape = dcape if (not np.isnan(dcape)) else (cape * 0.35 if not np.isnan(cape) else 0.0)
    if base_dcape < 300: return np.nan
    score = min(base_dcape / 1100.0, 1.0) * 45
    if not np.isnan(rh700) and rh700 < 55: score += 20
    if not np.isnan(dls06) and dls06 > 15: score += 20
    val = float(np.clip(np.round(score, 0), 0, 100))
    return val if val > 0 else np.nan

def calc_derecho_prob(dcp, dls06):
    if np.isnan(dcp) or dcp < 0.2: return np.nan
    score = (dcp / 1.5) * 50.0 + (dls06 / 25.0) * 50.0
    val = float(np.clip(np.round(score, 0), 0, 100))
    return val if val > 0 else np.nan

def calc_heavy_rain_potential(pwat, rh850, rh700, vvel850, dls06, storm_speed, foehn=False, orographic=1.0):
    if np.isnan(pwat) or pwat < 20: return np.nan
    score = min(pwat/40,1)*35 + (min(rh850/90,1)*20 if not np.isnan(rh850) else 0)
    if not np.isnan(storm_speed):
        if storm_speed < 5.0: score += 20      
        elif storm_speed > 18.0: score -= 15    
    score *= orographic
    if foehn: score *= 0.6
    val = float(np.clip(round(score,0),0,100))
    return val if val > 0 else np.nan

def get_estofex_category(prob_ulewa, prob_grad, prob_db, prob_tornado, prob_derecho):
    p_u = prob_ulewa if not np.isnan(prob_ulewa) else 0.0
    p_g = prob_grad if not np.isnan(prob_grad) else 0.0
    p_w = prob_db if not np.isnan(prob_db) else 0.0
    p_t = prob_tornado if not np.isnan(prob_tornado) else 0.0
    p_d = prob_derecho if not np.isnan(prob_derecho) else 0.0
    
    max_prob = max(p_u, p_g, p_w, p_t, p_d)
    if max_prob < 15: return ""
    elif max_prob < 30: return "1/MRGL"
    elif max_prob < 50: return "2/SLGT"
    elif max_prob < 75: return "3/ENH"
    else: return "4/MDT"

def classify_storm_mode(cape, dls06, srh3, cin, lcl, prob):
    if np.isnan(prob) or prob < 10 or np.isnan(cape) or cape < 100: return "Brak"
    if cin < -150 and cape > 800: return "Elevated"
    if dls06 < 10: return "Pulse Storm / Pojedyncza"
    if dls06 > 25 and srh3 > 200 and cape > 800: return "Supercell (Wysokie Ryzyko)"
    if dls06 > 20 and srh3 > 150 and cape > 500: return "Supercell"
    if dls06 > 15 and srh3 > 100 and cape > 300: return "Marginal Supercell"
    if dls06 > 20 and srh3 < 100: return "QLCS / Bow Echo"
    if dls06 > 15 and srh3 < 100: return "Squall Line"
    if 10 <= dls06 <= 20 and cape > 500: return "Multicell Cluster"
    return "Zwykła / Multicell"

def calc_orographic_factor(wdir, wspd):
    if np.isnan(wdir) or np.isnan(wspd): return 1.0
    angle_diff = min(abs(wdir - 220), 360 - abs(wdir - 220))
    return round(min(1.0 + 0.6 * np.sin(np.radians(angle_diff)) * min(wspd/15, 1.2), 1.8), 2)

def lcl_height_m(t2m_c, td2m_c):
    if np.isnan(t2m_c) or np.isnan(td2m_c): return np.nan
    diff = t2m_c - td2m_c
    if diff < 0: diff = 0.0
    return float(np.round(125.0 * diff, 1))

# ==================== ASYNCHRONICZNE POBIERANIE (AIOHTTP) ====================
if ASYNC_AVAILABLE:
    async def fetch_single_async(session, fh, sem):
        grib_filename = f"gfs.t{RUN_HOUR}z.pgrb2.0p25.f{fh:03d}"
        local_path = os.path.join(OUTPUT_DIR, f"krosno_conv_v17_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")
        url = build_url(grib_filename)
        
        async with sem:
            for attempt in range(1, 6): 
                await asyncio.sleep(0.5) 
                try:
                    async with session.get(url, headers=HEADERS, timeout=60) as r:
                        if r.status == 200:
                            data = await r.read()
                            if b"GRIB" in data[:12]:
                                with open(local_path, "wb") as f: f.write(data)
                                print(f"  ✅ [SUKCES] f{fh:03d} (Próba {attempt})", flush=True)
                                return True
                        elif r.status == 403:
                            await asyncio.sleep(10 + attempt * 2)
                        else:
                            await asyncio.sleep(3)
                except Exception:
                    await asyncio.sleep(3)
            print(f"  🛑 [PORAŻKA OSTATECZNA] f{fh:03d}", flush=True)
            return False

    async def download_all_async(pending):
        sem = asyncio.Semaphore(8)
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_single_async(session, fh, sem) for fh in pending]
            await asyncio.gather(*tasks)

def download_missing_gribs(forecast_hours, global_attempt):
    # Bugfix: NOAA filter dla pojedynczych stacji potrafi ważyć od 4 KB. Próg obniżony do 2048 bajtów.
    pending = [fh for fh in forecast_hours if not os.path.exists(os.path.join(OUTPUT_DIR, f"krosno_conv_v17_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")) or os.path.getsize(os.path.join(OUTPUT_DIR, f"krosno_conv_v17_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")) < 2048]
    
    if not pending: 
        print("[DOWNLOAD] Wszystkie pliki obecne w lokalnym cache.", flush=True)
        return
    
    print(f"\n🔄 [CYKL {global_attempt}] Pobieranie brakujących plików ({len(pending)} sztuk)...", flush=True)

    if ASYNC_AVAILABLE:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(download_all_async(pending))
    else:
        from concurrent.futures import ThreadPoolExecutor
        def fetch_single(fh):
            grib_filename = f"gfs.t{RUN_HOUR}z.pgrb2.0p25.f{fh:03d}"
            local_path = os.path.join(OUTPUT_DIR, f"krosno_conv_v17_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")
            url = build_url(grib_filename)
            for attempt in range(1, 6):
                sleep(1.5)
                try:
                    r = requests.get(url, headers=HEADERS, timeout=120)
                    if r.status_code == 200 and b"GRIB" in r.content[:12]:
                        with open(local_path, "wb") as f: f.write(r.content)
                        print(f"  ✅ [SUKCES] f{fh:03d} (Próba {attempt})", flush=True)
                        return True
                    elif r.status_code == 403: sleep(10)
                    else: sleep(5)
                except: sleep(5)
            print(f"  🛑 [PORAŻKA OSTATECZNA] f{fh:03d}", flush=True)
            return False
        with ThreadPoolExecutor(max_workers=1) as ex: ex.map(fetch_single, pending)

# ==================== FORMATOWANIE WYNIKÓW ====================
def fmt(val, decimals=0, zero_as_nan=True):
    if val is None or pd.isna(val) or np.isnan(val): return ""
    if isinstance(val, str): return val
    if zero_as_nan and val == 0: return ""
    if decimals == 0: return int(round(val))
    return round(val, decimals)

# ==================== GŁÓWNE PRZETWARZANIE ====================
def process_local_gribs(forecast_hours):
    print("\n[ANALIZA] Rozpoczynam przetwarzanie plików GRIB...", flush=True)
    rows = []
    cumulative_rain = 0.0
    prev_fh = 0
    
    for fh in forecast_hours:
        path = os.path.join(OUTPUT_DIR, f"krosno_conv_v17_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")
        if not os.path.exists(path) or os.path.getsize(path) < 2048: 
            prev_fh = fh
            continue
            
        datasets = []
        try:
            print(f"  ▶ Analiza f{fh:03d}...", end=" ", flush=True)
            
            ds_sfc = try_open_by_filter(path, {"typeOfLevel": "surface", "stepType": "instant"})
            ds_avg = try_open_by_filter(path, {"typeOfLevel": "surface", "stepType": "avg"})
            ds_accum = try_open_by_filter(path, {"typeOfLevel": "surface", "stepType": "accum"})
            ds_prate = try_open_by_filter(path, {"shortName": "prate"})
            ds_tp = try_open_by_filter(path, {"shortName": "tp"})
            
            ds_2m = try_open_by_filter(path, {"typeOfLevel": "heightAboveGround", "level": 2})
            ds_10m = try_open_by_filter(path, {"typeOfLevel": "heightAboveGround", "level": 10})
            ds_pwat = try_open_by_filter(path, {"typeOfLevel": "atmosphereSingleLayer"})
            ds_isobaric = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa"})
            ds_hlcy = try_open_by_filter(path, {"shortName": "hlcy"})
            ds_500 = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa", "level": 500})
            ds_925 = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa", "level": 925})
            ds_850 = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa", "level": 850})
            ds_700 = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa", "level": 700})

            datasets.extend([ds_sfc, ds_avg, ds_accum, ds_prate, ds_tp, ds_2m, ds_10m, ds_pwat, ds_isobaric, ds_hlcy, ds_500, ds_925, ds_850, ds_700])

            valid_time = datetime.strptime(RUN_DATE + RUN_HOUR, "%Y%m%d%H") + timedelta(hours=fh)

            t2m = safe_get_point(ds_2m, ['t2m', '2t', 'TMP']) - 273.15
            td2m = safe_get_point(ds_2m, ['d2m', '2d', 'DPT']) - 273.15
            cape = safe_get_point(ds_sfc, ['cape', 'CAPE'])
            cin = safe_get_point(ds_sfc, ['cin', 'CIN'])
            li = safe_get_point(ds_sfc, ['lftx', 'LFTX'])
            pwat = safe_get_point(ds_pwat, ['pwat', 'PWAT'])
            p_sfc = safe_get_point(ds_sfc, ['sp', 'PRES']) 
            
            if np.isnan(p_sfc): p_sfc = 98000.0 # Zastępcze 980 hPa dla 280 m n.p.m.

            t850 = safe_get_point(ds_850, ['t', 'TMP']) - 273.15
            t700 = safe_get_point(ds_700, ['t', 'TMP']) - 273.15
            t500 = safe_get_point(ds_500, ['t', 'TMP']) - 273.15
            td850 = safe_get_point(ds_850, ['dpt', 'DPT']) - 273.15
            rh850 = safe_get_point(ds_850, ['r', 'RH'])
            rh700 = safe_get_point(ds_700, ['r', 'RH'])
            vvel850 = safe_get_point(ds_850, ['w', 'VVEL'])

            u10 = safe_get_point(ds_10m, ['u10', '10u', 'UGRD'])
            v10 = safe_get_point(ds_10m, ['v10', '10v', 'VGRD'])
            u500 = safe_get_point(ds_500, ['u', 'UGRD'])
            v500 = safe_get_point(ds_500, ['v', 'VGRD'])
            u925 = safe_get_point(ds_925, ['u', 'UGRD'])
            v925 = safe_get_point(ds_925, ['v', 'VGRD'])
            u700 = safe_get_point(ds_700, ['u', 'UGRD'])
            v700 = safe_get_point(ds_700, ['v', 'VGRD'])
            u850 = safe_get_point(ds_850, ['u', 'UGRD'])
            v850 = safe_get_point(ds_850, ['v', 'VGRD'])

            step_duration = fh - prev_fh if fh > 0 else 0
            prate = np.nan
            for ds_opad in [ds_avg, ds_prate, ds_sfc]:
                val = safe_get_point(ds_opad, ['prate', 'PRATE'])
                if not np.isnan(val):
                    prate = val; break
                    
            apcp = np.nan
            for ds_opad in [ds_tp, ds_accum]:
                val = safe_get_point(ds_opad, ['tp', 'apcp', 'APCP'])
                if not np.isnan(val):
                    apcp = val; break

            rain_hour = 0.0
            if not np.isnan(prate) and prate > 0:
                rain_hour = prate * 3600.0
            elif not np.isnan(apcp) and apcp > 0 and step_duration > 0:
                rain_hour = apcp / step_duration

            if fh > 0:
                cumulative_rain += (rain_hour * step_duration)

            wdir = wind_direction(u10, v10)
            wdir850 = wind_direction(u850, v850)
            wspd10m = np.hypot(u10, v10) if not np.isnan(u10) else np.nan
            foehn = is_foehn_wind(wdir850)
            orog_factor = calc_orographic_factor(wdir, wspd10m)

            dls06 = np.hypot(u500 - u10, v500 - v10) if not any(np.isnan([u500, u10])) else np.nan
            dls01 = np.hypot(u925 - u10, v925 - v10) if not any(np.isnan([u925, u10])) else np.nan
            lr_700_500 = (t700 - t500) / 2.0 if not any(np.isnan([t700, t500])) else np.nan
            brn = calc_brn(cape, dls06)
            lcl = lcl_height_m(t2m, td2m)

            u_storm, v_storm = estimate_storm_motion(u10, v10, u500, v500)
            storm_speed = np.hypot(u_storm, v_storm) if not (np.isnan(u_storm) or np.isnan(v_storm)) else np.nan
            
            srh_01 = calc_srh_layer(u10, v10, u925, v925, u_storm, v_storm)
            srh3_manual = calc_srh_layer(u10, v10, u700, v700, u_storm, v_storm) * 1.3
            srh3 = safe_get_point(ds_hlcy, ['hlcy', 'HLCY'])
            if np.isnan(srh3): srh3 = srh3_manual

            mucape = cape * 1.1 if not np.isnan(cape) else np.nan
            dcape = cape * 0.35 if not np.isnan(cape) else np.nan
            
            mu_mr = 10.0
            if not np.isnan(td850):
                e_vp = 6.112 * np.exp((17.67 * td850) / (td850 + 243.5))
                mu_mr = 622.0 * e_vp / (1000.0 - e_vp)

            ship = calc_ship(mucape, mu_mr, lr_700_500, t500, dls06)
            ehi = calc_ehi(cape, srh3)
            stp_full = calc_full_stp(cape, lcl, srh_01, dls06, cin)
            
            # === POPRAWKA METPY === Zastrzyk danych z powierzchni Krosna (280m n.p.m.)
            if METPY_AVAILABLE and ds_isobaric is not None:
                try:
                    ds_point = ds_isobaric.sel(latitude=KROSNO_LAT, longitude=KROSNO_LON, method="nearest")
                    level_name = 'level' if 'level' in ds_point.coords else 'isobaricInhPa'
                    levels = ds_point[level_name].values.astype(float)
                    
                    # Filtrujemy tylko poziomy ciśnienia znajdujące się NAD Krosnem
                    p_sfc_hpa = p_sfc / 100.0
                    idx = levels < p_sfc_hpa
                    
                    p_prof = np.insert(levels[idx], 0, p_sfc_hpa) * units.hPa
                    t_vals = (ds_point['t'] if 't' in ds_point.data_vars else ds_point['TMP']).sel({level_name: levels[idx]}).values
                    t_prof = np.insert(t_vals, 0, t2m + 273.15) * units.K
                    
                    td_vals = (ds_point['dpt'] if 'dpt' in ds_point.data_vars else ds_point['DPT']).sel({level_name: levels[idx]}).values
                    td_prof = np.insert(td_vals, 0, td2m + 273.15) * units.K
                    
                    u_vals = (ds_point['u'] if 'u' in ds_point.data_vars else ds_point['UGRD']).sel({level_name: levels[idx]}).values
                    u_prof = np.insert(u_vals, 0, u10) * units('m/s')
                    
                    v_vals = (ds_point['v'] if 'v' in ds_point.data_vars else ds_point['VGRD']).sel({level_name: levels[idx]}).values
                    v_prof = np.insert(v_vals, 0, v10) * units('m/s')
                    
                    hgt_vals = (ds_point['gh'] if 'gh' in ds_point.data_vars else ds_point['HGT']).sel({level_name: levels[idx]}).values
                    hgt_prof = np.insert(hgt_vals, 0, KROSNO_ELEVATION_M) * units.meter

                    sbcape, sbcin = mpcalc.surface_based_cape_cin(p_prof, t_prof, td_prof)
                    mucape_val, _ = mpcalc.most_unstable_cape_cin(p_prof, t_prof, td_prof)
                    mucape = float(mucape_val.magnitude)
                    
                    lcl_p, _ = mpcalc.lcl(p_prof[0], t_prof[0], td_prof[0])
                    lcl_h = float(mpcalc.pressure_to_height_std(lcl_p).to('m').magnitude)
                    
                    u_storm2, v_storm2, _ = mpcalc.bunkers_storm_motion(p_prof, u_prof, v_prof, hgt_prof)
                    srh01_val = mpcalc.storm_relative_helicity(hgt_prof, u_prof, v_prof, depth=1*units.km, storm_u=u_storm2, storm_v=v_storm2)[0]
                    srh03_val = mpcalc.storm_relative_helicity(hgt_prof, u_prof, v_prof, depth=3*units.km, storm_u=u_storm2, storm_v=v_storm2)[0]
                    
                    dcape_val = mpcalc.downdraft_cape(p_prof, t_prof, td_prof)
                    dcape = float(dcape_val.magnitude)
                    
                    mu_mr_metpy = float(mpcalc.mixing_ratio_from_specific_humidity(mpcalc.specific_humidity_from_dewpoint(p_prof[0], td_prof[0])).magnitude * 1000)
                    
                    srh3 = float(srh03_val.magnitude) if not np.isnan(float(srh03_val.magnitude)) else srh3
                    ship = calc_ship(mucape, mu_mr_metpy, lr_700_500, t500, dls06)
                    ehi = calc_ehi(cape, srh3)
                    stp_full = calc_full_stp(float(sbcape.magnitude), lcl_h, float(srh01_val.magnitude), dls06, float(sbcin.magnitude))
                except Exception as e: pass

            stp_old = calc_stp(cape, srh_01, dls06, lcl)
            prob_old = calc_storm_prob(cape, cin, li, dls06, dls01, srh3, srh_01, pwat, lcl, lr_700_500, brn, foehn)
            prob_sc = calc_supercell_risk(cape, dls06, srh3, brn, li, dls01, srh_01)
            prob_ulewa = calc_heavy_rain_potential(pwat, rh850, rh700, vvel850, dls06, storm_speed, foehn, orog_factor)
            prob_db = calc_wind_risk(dcape, rh700, dls06, cape) 
            
            base_stp = stp_full if not np.isnan(stp_full) else 0.0
            prob_tornado = calc_tornado_prob(base_stp, lcl, srh_01, dls01)
            
            base_cape = cape if not np.isnan(cape) else 0.0
            base_ship = ship if not np.isnan(ship) else 0.0
            prob_grad = min(max(0, base_ship * 30 + (base_cape / 1500 * 25)), 100)
            if prob_grad <= 0: prob_grad = np.nan

            dcp = calc_dcp(mucape, dcape, dls06, u10, v10, u850, v850, u700, v700, u500, v500)
            prob_derecho = calc_derecho_prob(dcp, dls06)
            
            # Wiatr Halny już sztucznie nie ucina tornad i derecho, bo operują one często na innych warstwach
            if foehn:
                if not np.isnan(prob_old): prob_old *= 0.65

            if np.isnan(prob_old) or prob_old <= 20:
                prob_sc = np.nan
                prob_tornado = np.nan
                prob_grad = np.nan
                prob_ulewa = np.nan

            if rain_hour < 2.0: prob_ulewa = np.nan

            rot_type = ""
            if not np.isnan(prob_sc) and prob_sc > 20:
                rot_type = supercell_rotation_type(srh3)
                
            hail = estimate_hail_size(cape, lr_700_500, dls06)
            lightning = calc_lightning_rate(cape, lcl, cin)
            
            prob_temp = min(base_cape / 1200 * 40 + (srh3 / 200 * 20 if not np.isnan(srh3) else 0), 100)
            storm_mode = classify_storm_mode(cape, dls06, srh3, cin, lcl, prob_temp)

            estofex_category = get_estofex_category(prob_ulewa, prob_grad, prob_db, prob_tornado, prob_derecho)
            orog_display = f"+{int(round((orog_factor - 1.0) * 100))}%" if orog_factor >= 1.05 else ""
            dcp_display = fmt(dcp, 2, True) if (not np.isnan(dcp) and dcp >= 0.2) else ""
            
            storm_speed_kmh = np.round(storm_speed * 3.6, 1) if not np.isnan(storm_speed) else np.nan
            storm_speed_display = fmt(storm_speed_kmh, 1, False) if (not np.isnan(prob_old) and prob_old >= 10) else ""

            tatry_vis = calc_tatry_visibility(t2m, td2m, rh700, pwat, wdir, wspd10m, rain_hour)

            rows.append({
                "Czas": valid_time,
                "T+": fh,
                "T2M [°C]": fmt(t2m, 1, False),
                "CAPE [J/kg]": fmt(cape, 0, True),
                "MUCAPE [J/kg]": fmt(mucape, 0, True),
                "DCAPE [J/kg]": fmt(dcape, 0, True),
                "CIN [J/kg]": fmt(cin, 0, False),
                "DLS 0-6km [m/s]": fmt(dls06, 1, True),
                "DLS 0-1km [m/s]": fmt(dls01, 1, True),
                "SRH 0-3km": fmt(srh3, 0, True),
                "SRH 0-1km approx": fmt(srh_01, 1, True),
                "BRN": fmt(brn, 1, True),
                "STP (stary)": fmt(stp_old, 2, True),
                "STP (pełny MetPy)": fmt(stp_full, 2, True),
                "SHIP": fmt(ship, 2, True),
                "EHI": fmt(ehi, 2, True),
                "PWAT [mm]": fmt(pwat, 1, True),
                "LCL [m]": fmt(lcl, 0, False),
                "Opad [mm/h]": fmt(rain_hour, 1, False),
                "Kumulacyjny opad [mm]": fmt(cumulative_rain, 1, False),
                "Prob Burzy [%]": fmt(prob_old, 0, True),
                "Prob SC [%]": fmt(prob_sc, 0, True),
                "Prob Tornado [%]": fmt(prob_tornado, 0, True),
                "Prob Grad [%]": fmt(prob_grad, 0, True),
                "Prob DB (Wiatr) [%]": fmt(prob_db, 0, True),
                "Prob Derecho [%]": fmt(prob_derecho, 0, True),
                "Prob Ulewa [%]": fmt(prob_ulewa, 0, True),
                "Storm Mode": storm_mode,
                "Rotacja": rot_type,
                "Prędkość Burzy [km/h]": storm_speed_display,
                "Grad [cm]": fmt(hail, 1, True) if not np.isnan(prob_grad) else "", 
                "DCP (Derecho)": dcp_display,
                "Błyski [1/min]": fmt(lightning, 1, True) if not np.isnan(prob_old) and prob_old > 25 else "",
                "Halny": "TAK" if foehn else "",
                "Orografia": orog_display,
                "Poziom (ESTOFEX)": estofex_category,
                "Widzialność Tatr": tatry_vis
            })
            prev_fh = fh
            print("OK", flush=True)
        except Exception as e:
            print(f"Błąd f{fh:03d}: {e}", flush=True)
            prev_fh = fh
            continue
        finally:
            for ds in datasets:
                if ds is not None: ds.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"[ANALIZA ZAKOŃCZONA] Przetworzono {len(df)} rekordów", flush=True)
    return df

def save_outputs(df):
    if df.empty: return []
    csv_path = os.path.join(OUTPUT_DIR, "gfs-conv.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8', na_rep='')
    
    xlsx_path = os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z.xlsx")
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Burze_v3', na_rep='')
        ws = writer.sheets['Burze_v3']
        red = writer.book.add_format({'bg_color': '#FF3333', 'font_color': 'white'})
        
        # Lepsza metoda alokowania kolumn w XlsxWriter bez limitów
        for col_idx, col_name in enumerate(df.columns):
            if col_name in ["CAPE [J/kg]", "SHIP", "STP (stary)", "DCP (Derecho)", "Prob Burzy [%]", "Prob SC [%]", "Prob DB (Wiatr) [%]", "Prob Derecho [%]"]:
                
                thresh = 50
                if col_name == "CAPE [J/kg]": thresh = 1000
                elif col_name in ["SHIP", "STP (stary)", "DCP (Derecho)"]: thresh = 1.0
                elif col_name == "Prob Burzy [%]": thresh = 70
                elif col_name == "Prob Derecho [%]": thresh = 30
                
                from xlsxwriter.utility import xl_col_to_name
                letter = xl_col_to_name(col_idx)
                r_col = f"{letter}2:{letter}400"
                ws.conditional_format(r_col, {'type': 'cell', 'criteria': '>=', 'value': thresh, 'format': red})
            
    return [csv_path, xlsx_path]

def upload_to_ftp(files):
    load_dotenv()
    host, user, pswd = os.getenv("FTP_HOST"), os.getenv("FTP_USER"), os.getenv("FTP_PASS")
    if not all([host, user, pswd]): return
    try:
        print("\n[FTP] Łączenie...", flush=True)
        from ftplib import FTP, error_perm
        ftp = FTP(host, user, pswd, timeout=40)
        ftp.cwd("/stacja.meteo-krosno.pl/")
        
        for p in files:
            if p.endswith('.csv'):
                with open(p, "rb") as f:
                    ftp.storbinary("STOR gfs-conv.csv", f)
                    print("  📤 Wysłano na FTP (nadpisano): gfs-conv.csv", flush=True)
                
                arch_dir = "/stacja.meteo-krosno.pl/archiv_conv"
                try: ftp.cwd(arch_dir)
                except error_perm:
                    ftp.mkd(arch_dir)
                    ftp.cwd(arch_dir)
                
                arch_name = f"gfs_conv_tab_{RUN_DATE[:4]}_{RUN_DATE[4:6]}_{RUN_DATE[6:8]}_{RUN_HOUR}.csv"
                with open(p, "rb") as f:
                    ftp.storbinary(f"STOR {arch_name}", f)
                    print(f"  📤 Wysłano na FTP (archiwum): {arch_name}", flush=True)
                
                ftp.cwd("/stacja.meteo-krosno.pl/")
            else:
                target = os.path.basename(p)
                with open(p, "rb") as f:
                    ftp.storbinary(f"STOR {target}", f)
                    print(f"  📤 Wysłano na FTP: {target}", flush=True)
                    
        ftp.quit()
        print("[FTP] ✅ Wszystkie pliki wysłane na serwer", flush=True)
    except Exception as e:
        print(f"[FTP ERROR] ❌ Błąd: {e}", flush=True)

if __name__ == "__main__":
    print(f"\n==========================================")
    print(f"🚀 START: GFS CONVECTION v17 ULTIMATE {RUN_DATE}{RUN_HOUR}Z")
    print(f"==========================================\n", flush=True)
    
    start_time = datetime.utcnow()
    global_attempt = 1
    
    while True:
        elapsed = (datetime.utcnow() - start_time).total_seconds() / 60
        if elapsed > MAX_TOTAL_WAIT_MINUTES:
            print(f"\n[TIMEOUT] Zakończono wymuszonym limitem {MAX_TOTAL_WAIT_MINUTES} minut.", flush=True)
            break
            
        download_missing_gribs(FORECAST_HOURS, global_attempt)
        df = process_local_gribs(FORECAST_HOURS)
        
        if not df.empty:
            files = save_outputs(df)
            upload_to_ftp(files)
            
        # Zaktualizowany sprawdzian braku plików - uwzględnia poprawiony próg 2048 bajtów
        missing = [fh for fh in FORECAST_HOURS if not os.path.exists(os.path.join(OUTPUT_DIR, f"krosno_conv_v17_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")) or os.path.getsize(os.path.join(OUTPUT_DIR, f"krosno_conv_v17_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")) < 2048]
        
        if not missing:
            print("\n🎉 [SUKCES] Wszystkie prognozy zostały pomyślnie przetworzone i wysłane (T+384)! Skrypt kończy pracę.", flush=True)
            break
            
        print(f"\n⏳ Niekompletne dane. Czekam 2 minuty na kolejne pliki NOAA...", flush=True)
        sleep(RETRY_INTERVAL_SECONDS)
        global_attempt += 1
