#!/usr/bin/env python3
# gfs_conv_v9_ultimate.py - WERSJA Z ZAAWANSOWANYM ALGORYTMEM RYZYKA I OPADÓW (POPRAWIONA)
import os
import requests
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from time import sleep
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

try:
    import metpy.calc as mpcalc
    from metpy.units import units
    METPY_AVAILABLE = True
except ImportError:
    METPY_AVAILABLE = False

OUTPUT_DIR = "gfs_krosno_conv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_LAT = 50.0
BOTTOM_LAT = 49.4
LEFT_LON = 21.3
RIGHT_LON = 22.01
KROSNO_LAT = 49.69
KROSNO_LON = 21.77
RETRY_INTERVAL_SECONDS = 600
MAX_TOTAL_WAIT_MINUTES = 90

now = datetime.utcnow()
current_time = now.time()
if current_time >= time(20, 0) or current_time < time(3, 0):
    RUN_HOUR = "18"
    RUN_DATE = now.strftime("%Y%m%d") if current_time >= time(22, 0) else (now - timedelta(days=1)).strftime("%Y%m%d")
elif time(3, 0) <= current_time < time(8, 30):
    RUN_HOUR = "00"
    RUN_DATE = now.strftime("%Y%m%d")
elif time(8, 30) <= current_time < time(14, 30):
    RUN_HOUR = "06"
    RUN_DATE = now.strftime("%Y%m%d")
else:
    RUN_HOUR = "12"
    RUN_DATE = now.strftime("%Y%m%d")

CYCLE_DIR = f"gfs.{RUN_DATE}/{RUN_HOUR}/atmos"
BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
FORECAST_HOURS = list(range(0, 384, 3))

STATIC_MIDDLE = (
    "&lev_2_m_above_ground=on&lev_10_m_above_ground=on"
    "&lev_850_mb=on&lev_700_mb=on&lev_500_mb=on&lev_925_mb=on&lev_1000_mb=on"
    "&lev_surface=on&lev_entire_atmosphere_%28considered_as_a_single_layer%29=on"
    "&lev_3000-0_m_above_ground=on&lev_1000-0_m_above_ground=on"
    "&lev_180-0_mb_above_ground=on"
    "&var_TMP=on&var_HGT=on&var_UGRD=on&var_VGRD=on&var_CAPE=on&var_CIN=on"
    "&var_LFTX=on&var_PWAT=on&var_HLCY=on&var_DPT=on&var_RH=on&var_SPFH=on&var_VVEL=on"
    "&var_APCP=on&var_PRATE=on"
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

# ==================== OBLICZENIA I ALGORYTMY ZAGROŻEŃ ====================
def estimate_storm_motion(u10, v10, u500, v500):
    if any(np.isnan(x) for x in [u10, v10, u500, v500]):
        return np.nan, np.nan
    u_mean = (u10 + u500) / 2.0
    v_mean = (v10 + v500) / 2.0
    du = u500 - u10
    dv = v500 - v10
    shear_mag = np.hypot(du, dv)
    if shear_mag < 1.0:
        return u_mean, v_mean
    perp_u = dv
    perp_v = -du
    scale = 7.5 / (np.hypot(perp_u, perp_v) + 1e-6)
    return u_mean + perp_u * scale, v_mean + perp_v * scale

def calc_srh_manual(u_bottom, v_bottom, u_top, v_top, u_storm, v_storm):
    if any(np.isnan(x) for x in [u_bottom, v_bottom, u_top, v_top, u_storm, v_storm]):
        return np.nan
    return float(np.round((u_bottom - u_storm) * (v_top - v_bottom) - (v_bottom - v_storm) * (u_top - u_bottom), 1))

def wind_direction(u, v):
    if np.isnan(u) or np.isnan(v):
        return np.nan
    return round((270 - np.rad2deg(np.arctan2(v, u))) % 360, 1)

def is_foehn_wind(wdir):
    if np.isnan(wdir):
        return False
    return 157.5 <= wdir <= 247.5

def calc_orographic_factor(wdir, wspd):
    if np.isnan(wdir) or np.isnan(wspd):
        return 1.0
    angle_diff = min(abs(wdir - 220), 360 - abs(wdir - 220))
    return round(min(1.0 + 0.6 * np.sin(np.radians(angle_diff)) * min(wspd/15, 1.2), 1.8), 2)

def robust_lcl(t2m_c, td2m_c):
    if np.isnan(t2m_c) or np.isnan(td2m_c):
        return 1000.0
    diff = max(0.0, t2m_c - td2m_c)
    return float(np.clip(round(125.0 * diff, 0), 0, 4000))

# MOCNO ZAAWANSOWANE PRAWDOPODOBIEŃSTWO BURZY
def calc_advanced_storm_prob(cape, cin, li, dls06, srh3, pwat, lcl, td2m, t2m, rh850, vvel850, foehn):
    if np.isnan(cape) or cape < 50 or np.isnan(t2m):
        return 0.0
    
    score = min(cape / 1500.0, 1.0) * 45  # Baza z termodynamiki (max 45%)
    
    # Czynniki Wilgotnościowe (do 20%)
    if not np.isnan(td2m):
        if td2m >= 18: score += 20
        elif td2m >= 15: score += 15
        elif td2m >= 12: score += 8
    if not np.isnan(pwat):
        if pwat > 35: score += 10
        elif pwat < 15: score -= 15
        
    # Czynniki Kinematyczne i Orkanizacji (do 15%)
    if not np.isnan(dls06):
        if 12 <= dls06 <= 25: score += 10
        elif dls06 > 25: score += 15
    if not np.isnan(srh3) and srh3 > 150:
        score += 5
        
    # Czynniki Wyzwalające LIFT (do 20%)
    if not np.isnan(cin):
        if cin < -200: score -= 35  # Tzw. CAP zbyt mocny
        elif -50 <= cin <= 0: score += 10
    if not np.isnan(li) and li < -3:
        score += 10
    if not np.isnan(vvel850) and vvel850 < -0.2:
        score += 10 # Wspomaganie wznoszenia (wymuszanie zjawisk)
        
    # Czynniki blokujące
    if foehn:
        score *= 0.5 # Wiatr halny wysusza profil
    if t2m < 10 and cape < 200:
        score -= 20
        
    return float(np.clip(np.round(score, 0), 0, 100))

# ZAAWANSOWANE RYZYKO SUPERKOMÓREK I TORNAD (NAPRAWIONE)
def calc_supercell_and_tornado(cape, dls06, srh1, srh3, lcl, stp, foehn):
    risk_sc = 5.0
    risk_tor = 0.0
    if np.isnan(cape) or cape < 150 or np.isnan(dls06) or dls06 < 10:
        return 0.0, 0.0, "Neutralna" # TUTAJ BYŁ BŁĄD: zwracało 4 wartości zamiast 3
        
    # Superkomórka
    if cape > 500: risk_sc += 15
    if cape > 1500: risk_sc += 20
    if dls06 > 15: risk_sc += 20
    if dls06 > 22: risk_sc += 20
    if not np.isnan(srh3):
        if srh3 > 150: risk_sc += 20
        if srh3 > 250: risk_sc += 15
    
    # Tornado
    if stp and not np.isnan(stp) and stp > 0.1:
        risk_tor += stp * 30
    if not np.isnan(srh1) and srh1 > 100:
        risk_tor += 20
    if not np.isnan(lcl) and lcl < 1000:
        risk_tor += 20
    if dls06 > 20 and cape > 800:
        risk_tor += 10
        
    if foehn:
        risk_sc *= 0.6
        risk_tor *= 0.3
        
    risk_sc = np.clip(np.round(risk_sc, 0), 0, 100)
    risk_tor = np.clip(np.round(risk_tor, 0), 0, 100)
    
    rot_type = "Neutralna"
    if srh3 and srh3 > 100: rot_type = "Prawoskrętna"
    elif srh3 and srh3 < -50: rot_type = "Lewoskrętna"
    
    return float(risk_sc), float(risk_tor), rot_type

def classify_storm_mode_advanced(cape, dls06, srh3, cin, prob):
    if prob < 15 or np.isnan(cape) or cape < 100: return ""
    if cin < -150 and cape > 800: return "Elevated"
    if dls06 > 20 and srh3 > 150: return "Supercell"
    if dls06 > 18 and srh3 < 100: return "QLCS / Bow Echo"
    if 10 <= dls06 <= 18: return "Multicell"
    return "Pulse / Zwykła"

def assess_main_threats(prob, sc_risk, rain_prob, grad_prob, tor_prob, wind_prob):
    if prob < 20:
        return "Brak / Znikome"
    
    threats = []
    if tor_prob > 15: threats.append("TORNADO")
    if rain_prob > 60: threats.append("Ulewa / Powódź Błyskawiczna")
    if grad_prob > 50: threats.append("Duży Grad (>2cm)")
    if wind_prob > 60 or sc_risk > 60: threats.append("Silny Wiatr (Nawałnica)")
    
    if not threats and prob > 30:
        return "Zwykła burza (Wyładowania)"
    
    return " + ".join(threats)

def estimate_hail_size(cape, lr, dls):
    if np.isnan(cape) or cape < 400: return 0.0
    hail = (cape / 1000.0) * (lr / 6.5 if not np.isnan(lr) else 1.0)
    if not np.isnan(dls) and dls > 18: hail *= 1.4
    return float(np.round(np.clip(hail, 0, 10), 1))

def calc_ship(mucape, mu_mr, lr_700_500, t500, dls06):
    if any(np.isnan(x) for x in [mucape, mu_mr, lr_700_500, t500, dls06]): return np.nan
    return round(max(0, min((mucape * mu_mr * lr_700_500 * abs(t500) * dls06) / 44000000, 8.0)), 2)

def calc_ehi(cape, srh):
    if np.isnan(cape) or np.isnan(srh) or cape <= 0: return 0.0
    return max(0.0, round((cape * srh) / 160000.0, 2))

def calc_full_stp(sbcape, lcl, srh01, dls06, cin):
    if any(np.isnan(x) for x in [sbcape, lcl, srh01, dls06]): return np.nan
    cin_val = max(0, (200 - abs(cin if not np.isnan(cin) else 0)) / 150)
    return round(max(0, min((min(sbcape/1500,1.5) * max(0,(2000-lcl)/1000) * min(srh01/150,1.5) * min(dls06/20,1.5) * cin_val), 10.0)), 2)


# ==================== GŁÓWNA LOGIKA GRIB ====================
def download_missing_gribs_parallel(forecast_hours):
    pending = [fh for fh in forecast_hours if not os.path.exists(os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")) or os.path.getsize(os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")) < 45000]
    if not pending: return
    print(f" → Pobieranie {len(pending)} plików...")

    def fetch_single(fh):
        grib_filename = f"gfs.t{RUN_HOUR}z.pgrb2.0p25.f{fh:03d}"
        local_path = os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")
        url = build_url(grib_filename)
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            if r.status_code == 200 and b"GRIB" in r.content[:12]:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                print(f" ✓ f{fh:03d}")
                return True
        except: pass
        return False

    with ThreadPoolExecutor(max_workers=10) as ex:
        ex.map(fetch_single, pending)

def process_local_gribs(forecast_hours):
    rows = []
    cumulative_rain = 0.0
    for fh in forecast_hours:
        path = os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")
        if not os.path.exists(path): continue
        try:
            print(f"  f{fh:03d}...", end=" ")
            
            ds_sfc = try_open_by_filter(path, {"typeOfLevel": "surface"})
            ds_2m = try_open_by_filter(path, {"typeOfLevel": "heightAboveGround", "level": 2})
            ds_10m = try_open_by_filter(path, {"typeOfLevel": "heightAboveGround", "level": 10})
            ds_pwat = try_open_by_filter(path, {"typeOfLevel": "atmosphereSingleLayer"})
            ds_isobaric = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa"})
            
            # Bezpośrednie filtry do zjawisk
            ds_hlcy = try_open_by_filter(path, {"shortName": "hlcy"})
            ds_cape_mu = try_open_by_filter(path, {"shortName": "cape", "typeOfLevel": "pressureFromGroundLayer"})
            
            # OPAD - Szukamy APCP (skumulowany), jak nie ma to PRATE (chwilowy)
            ds_accum = try_open_by_filter(path, {"stepType": "accum"})
            ds_tp = try_open_by_filter(path, {"shortName": "tp"})
            ds_prate = try_open_by_filter(path, {"shortName": "prate"})
            
            ds_500 = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa", "level": 500})
            ds_700 = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa", "level": 700})
            ds_850 = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa", "level": 850})
            ds_925 = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa", "level": 925})

            # Odczyt Podstawowy
            t2m = safe_get_point(ds_2m, ['t2m', '2t', 'TMP']) - 273.15
            td2m = safe_get_point(ds_2m, ['d2m', '2d', 'DPT']) - 273.15
            
            cape = safe_get_point(ds_sfc, ['cape', 'CAPE'])
            cin_val = safe_get_point(ds_sfc, ['cin', 'CIN'])
            cin = -abs(cin_val) if not np.isnan(cin_val) else 0.0
            li = safe_get_point(ds_sfc, ['lftx', 'LFTX'])
            pwat = safe_get_point(ds_pwat, ['pwat', 'PWAT'])

            t500 = safe_get_point(ds_500, ['t', 'TMP']) - 273.15
            t700 = safe_get_point(ds_700, ['t', 'TMP']) - 273.15
            rh850 = safe_get_point(ds_850, ['r', 'RH'])
            vvel850 = safe_get_point(ds_850, ['w', 'VVEL'])

            u10 = safe_get_point(ds_10m, ['u10', '10u', 'UGRD'])
            v10 = safe_get_point(ds_10m, ['v10', '10v', 'VGRD'])
            u500 = safe_get_point(ds_500, ['u', 'UGRD'])
            v500 = safe_get_point(ds_500, ['v', 'VGRD'])
            u700 = safe_get_point(ds_700, ['u', 'UGRD'])
            v700 = safe_get_point(ds_700, ['v', 'VGRD'])
            u925 = safe_get_point(ds_925, ['u', 'UGRD'])
            v925 = safe_get_point(ds_925, ['v', 'VGRD'])

            # ODCZYT OPADU I OBLICZANIE KUMULACJI
            apcp = safe_get_point(ds_accum, ['tp', 'apcp', 'APCP'])
            if np.isnan(apcp): apcp = safe_get_point(ds_tp, ['tp', 'apcp', 'APCP'])
            
            rain_3h = 0.0
            rain_hour = 0.0
            if not np.isnan(apcp) and apcp >= 0:
                rain_3h = apcp
                rain_hour = apcp / 3.0
            else:
                prate = safe_get_point(ds_prate, ['prate', 'PRATE'])
                if not np.isnan(prate) and prate > 0:
                    rain_hour = prate * 3600.0
                    rain_3h = rain_hour * 3.0
            
            if fh > 0: # Dla godz. 0 nie sumujemy opadu z przeszłości
                cumulative_rain += rain_3h

            # Wiatry i SRH Ręczne jako Fallback
            wdir = wind_direction(u10, v10)
            foehn = is_foehn_wind(wdir)
            orog_factor = calc_orographic_factor(wdir, np.hypot(u10, v10) if not np.isnan(u10) else np.nan)

            dls06 = np.hypot(u500 - u10, v500 - v10) if not any(np.isnan([u500, u10])) else np.nan
            dls01 = np.hypot(u925 - u10, v925 - v10) if not any(np.isnan([u925, u10])) else np.nan
            u_storm, v_storm = estimate_storm_motion(u10, v10, u500, v500)
            
            srh3_manual = calc_srh_manual(u10, v10, u700, v700, u_storm, v_storm) * 1.3 # 0-3km approx
            srh1_manual = calc_srh_manual(u10, v10, u925, v925, u_storm, v_storm)

            srh_val = safe_get_point(ds_hlcy, ['hlcy', 'HLCY'])
            srh3 = srh_val if not np.isnan(srh_val) else srh3_manual
            srh1 = srh1_manual

            # Indeksy
            mucape_val = safe_get_point(ds_cape_mu, ['cape', 'CAPE'])
            mucape = mucape_val if not np.isnan(mucape_val) else cape
            lcl = robust_lcl(t2m, td2m)
            lr_700_500 = (t700 - t500) / 2.0 if not any(np.isnan([t700, t500])) else np.nan

            ehi = calc_ehi(cape, srh3)
            stp_full = calc_full_stp(cape, lcl, srh1, dls06, cin)
            
            # Własnoręcznie liczymy wskaźnik mieszania dla SHIP
            mu_mr = 10.0
            if not np.isnan(td2m):
                e_vp = 6.112 * np.exp((17.67 * td2m) / (td2m + 243.5))
                mu_mr = 622.0 * e_vp / (1000.0 - e_vp)

            ship = calc_ship(mucape, mu_mr, lr_700_500, t500, dls06)
            hail = estimate_hail_size(cape, lr_700_500, dls06)

            # PRAWDOPODOBIEŃSTWA
            prob_storm = calc_advanced_storm_prob(cape, cin, li, dls06, srh3, pwat, lcl, td2m, t2m, rh850, vvel850, foehn)
            storm_mode = classify_storm_mode_advanced(cape, dls06, srh3, cin, prob_storm)
            
            sc_risk, prob_tor, rot_type = calc_supercell_and_tornado(cape, dls06, srh1, srh3, lcl, stp_full, foehn)
            
            prob_grad = min(max(0, (ship or 0) * 40 + (cape / 2000 * 30)), 100) if prob_storm > 20 else 0
            prob_wind = min(max(0, (dls06 / 25 * 50) + (cape / 1500 * 50)), 100) if prob_storm > 20 else 0
            
            # Ulewa
            hr_score = (min(pwat/40,1)*40 + min(rain_hour/5,1)*30) * orog_factor
            if not np.isnan(vvel850) and vvel850 < 0: hr_score += 20
            prob_rain = min(round(hr_score, 0), 100) if prob_storm > 10 else 0
            
            # Główna diagnoza tekstowa
            threats = assess_main_threats(prob_storm, sc_risk, prob_rain, prob_grad, prob_tor, prob_wind)

            rows.append({
                "Czas": datetime.strptime(RUN_DATE + RUN_HOUR, "%Y%m%d%H") + timedelta(hours=fh),
                "T+": fh,
                "T2M [°C]": round(t2m, 1),
                "CAPE [J/kg]": int(round(cape)) if not np.isnan(cape) else 0,
                "MUCAPE [J/kg]": int(round(mucape)) if not np.isnan(mucape) else "-",
                "CIN [J/kg]": int(round(cin)) if not np.isnan(cin) else 0,
                "DLS 0-6km [m/s]": round(dls06, 1) if not np.isnan(dls06) else "-",
                "DLS 0-1km [m/s]": round(dls01, 1) if not np.isnan(dls01) else "-",
                "SRH 0-3km": int(round(srh3)) if not np.isnan(srh3) else 0,
                "SRH 0-1km approx": round(srh1, 1) if not np.isnan(srh1) else "-",
                "STP": round(stp_full, 2) if not np.isnan(stp_full) else "-",
                "SHIP": round(ship, 2) if not np.isnan(ship) else "-",
                "EHI": round(ehi, 2) if not np.isnan(ehi) else "-",
                "PWAT [mm]": round(pwat, 1) if not np.isnan(pwat) else "-",
                "LCL [m]": int(round(lcl)) if not np.isnan(lcl) else "-",
                "Opad [mm/h]": round(rain_hour, 1),
                "Kumulacyjny opad [mm]": round(cumulative_rain, 1),
                "Prob Burzy [%]": prob_storm,
                "Prob Nawałnicy [%]": round(prob_wind, 0),
                "Prob Ulewy [%]": prob_rain,
                "Prob Gradu [%]": round(prob_grad, 0),
                "Prob Tornada [%]": round(prob_tor, 0),
                "Główne Zagrożenia": threats,
                "Storm Mode": storm_mode,
                "Rotacja": rot_type,
                "Grad [cm]": hail,
                "Halny": "TAK" if foehn else "NIE"
            })
            print("OK")
        except Exception as e:
            print(f"Błąd f{fh:03d}: {e}")
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\nPrzetworzono {len(df)} rekordów")
    return df

def save_outputs(df):
    if df.empty: return []
    csv_path = os.path.join(OUTPUT_DIR, "gfs-conv.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    xlsx_path = os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z.xlsx")
    
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Analiza_Burzowa')
        ws = writer.sheets['Analiza_Burzowa']
        ws.set_column('A:AD', 14) # Rozszerza kolumny dla wygody odczytu tekstów (np. Zagrożeń)
        
        red_bg = writer.book.add_format({'bg_color': '#FF3333', 'font_color': 'white'})
        
        # Formatowanie kolumn: CAPE (D), Prob Burzy (R), Ulewy (T) itd. Zależnie od indeksów
        ws.conditional_format('D2:D300', {'type': 'cell', 'criteria': '>=', 'value': 1000, 'format': red_bg})
        ws.conditional_format('K2:K300', {'type': 'cell', 'criteria': '>=', 'value': 1.0, 'format': red_bg}) # STP
    return [csv_path, xlsx_path]

def upload_to_ftp(files):
    load_dotenv()
    host, user, pswd = os.getenv("FTP_HOST"), os.getenv("FTP_USER"), os.getenv("FTP_PASS")
    if not all([host, user, pswd]): return
    try:
        from ftplib import FTP
        ftp = FTP(host, user, pswd, timeout=40)
        ftp.cwd("/stacja.meteo-krosno.pl/")
        for p in files:
            target = "gfs-conv.csv" if p.endswith('.csv') else os.path.basename(p)
            with open(p, "rb") as f:
                ftp.storbinary(f"STOR {target}", f)
        ftp.quit()
        print("✅ FTP OK")
    except Exception as e:
        print(f"❌ FTP Error: {e}")

if __name__ == "__main__":
    print(f"\n=== GFS CONVECTION v9 ULTIMATE {RUN_DATE}{RUN_HOUR}Z ===\n")
    start_time = datetime.utcnow()
    while True:
        elapsed = (datetime.utcnow() - start_time).total_seconds() / 60
        if elapsed > MAX_TOTAL_WAIT_MINUTES:
            break
        download_missing_gribs_parallel(FORECAST_HOURS)
        df = process_local_gribs(FORECAST_HOURS)
        if not df.empty:
            files = save_outputs(df)
            upload_to_ftp(files)
        missing = [fh for fh in FORECAST_HOURS if not os.path.exists(os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2"))]
        if not missing:
            print("✅ Wszystko przetworzone!")
            break
        print(f"⏳ Czekam 10 min...")
        sleep(RETRY_INTERVAL_SECONDS)
