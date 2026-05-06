#!/usr/bin/env python3
# gfs_krosno_conv.py - Wersja rozbudowana o SRH + dodatkowe ważne parametry konwekcyjne (DLS 0-1km, K-Index, Total Totals, BRN)
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
# FILTR POBIERANIA (zaktualizowany o HLCY, 925mb + DPT dla K-Index/TT)
# -----------------------
STATIC_MIDDLE = (
    "&lev_2_m_above_ground=on"
    "&lev_10_m_above_ground=on"
    "&lev_850_mb=on"
    "&lev_700_mb=on"
    "&lev_500_mb=on"
    "&lev_925_mb=on"
    "&lev_surface=on"
    "&lev_entire_atmosphere_%28considered_as_a_single_layer%29=on"
    "&lev_mean_sea_level=on"
    "&lev_3000-0_m_above_ground=on"
    "&lev_0-3_km_above_ground=on"
    "&var_TMP=on"
    "&var_HGT=on"
    "&var_UGRD=on"
    "&var_VGRD=on"
    "&var_CAPE=on"
    "&var_CIN=on"
    "&var_LFTX=on"
    "&var_PWAT=on"
    "&var_HLCY=on"
    "&var_DPT=on"
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


def safe_get_point_level(ds, level, possible_names):
    """Bezpieczne pobieranie wartości z poziomu izobarycznego (dla T i Td)"""
    if ds is None:
        return np.nan
    for name in possible_names:
        if name in ds.data_vars:
            try:
                val = ds[name].sel(level=level, latitude=KROSNO_LAT, longitude=KROSNO_LON, method="nearest")
                return float(np.squeeze(np.array(val)))
            except:
                continue
    return np.nan


# -----------------------
# FUNKCJE SRH + NOWE OBLICZENIA
# -----------------------
def estimate_storm_motion(u10, v10, u500, v500):
    """Proste oszacowanie ruchu burzy (uproszczony Bunkers)"""
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
    perp_mag = np.hypot(perp_u, perp_v)
    scale = 7.5 / (perp_mag + 1e-6)
    u_storm = u_mean + perp_u * scale
    v_storm = v_mean + perp_v * scale
    return u_storm, v_storm


def calc_srh_01(u10, v10, u925, v925, u_storm, v_storm):
    """SRH 0-1km approx"""
    if any(np.isnan(x) for x in [u10, v10, u925, v925, u_storm, v_storm]):
        return np.nan
    srh = (u10 - u_storm) * (v925 - v10) - (v10 - v_storm) * (u925 - u10)
    return float(np.round(srh, 1))


def wind_direction(u, v):
    """Oblicza kierunek wiatru w stopniach (0-360) z U i V (meteorologiczna konwencja)"""
    if np.isnan(u) or np.isnan(v):
        return np.nan
    direction = (270 - np.rad2deg(np.arctan2(v, u))) % 360
    return round(direction, 1)


def wind_compass(deg):
    """Zwraca kierunek w skali 16-ramiennej"""
    if np.isnan(deg):
        return "-"
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    idx = int((deg + 11.25) / 22.5) % 16
    return dirs[idx]


def is_foehn_wind(wdir):
    """Sprawdza czy wiatr jest z kierunku halnego (SSE-S-SW) dla Krosna"""
    if np.isnan(wdir):
        return False
    # SSE (157.5°) do SW (247.5°)
    return 157.5 <= wdir <= 247.5


# K-Index usunięty na prośbę użytkownika (DPT nie jest konsekwentnie dostępne)


def calc_total_totals(t850, td850, t500):
    """Total Totals Index"""
    if any(np.isnan(x) for x in [t850, td850, t500]):
        return np.nan
    return float(np.round((t850 - t500) + (td850 - t500), 1))


def calc_brn(cape, dls_06):
    """Bulk Richardson Number (przybliżenie)"""
    if np.isnan(cape) or np.isnan(dls_06) or dls_06 < 1 or cape <= 0:
        return np.nan
    brn = cape / (0.5 * dls_06 ** 2)
    return float(np.round(brn, 1))


def calc_stp(cape, srh1, dls06, lcl):
    """Significant Tornado Parameter (uproszczona wersja operacyjna)
    
    Wzór uproszczony:
    STP = (CAPE/1500) × (SRH 0-1km/150) × (DLS 0-6km/20) × ((2000 - LCL)/1000)
    
    Wartości > 1 wskazują na podwyższone ryzyko silnych tornad (EF2+).
    """
    if any(np.isnan(x) for x in [cape, srh1, dls06, lcl]):
        return np.nan
    cape_term = min(cape / 1500.0, 1.5)
    srh_term  = min(srh1 / 150.0, 1.5)
    shear_term = min(dls06 / 20.0, 1.5)
    lcl_term  = max(0, (2000 - lcl) / 1000.0)
    stp = cape_term * srh_term * shear_term * lcl_term
    return float(round(max(0, min(stp, 8.0)), 2))


def supercell_rotation_type(srh3, supercell_risk):
    """Zwraca typ rotacji tylko gdy ryzyko superkomórki > 20"""
    if np.isnan(srh3) or supercell_risk <= 20:
        return "-"
    if srh3 > 50:
        return "Prawoskrętna"
    elif srh3 < -30:
        return "Lewoskrętna"
    else:
        return "Neutralna"


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
            ds_925      = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa", "level": 925})
            ds_850      = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa", "level": 850})
            ds_isobaric = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa"})
            ds_hlcy     = try_open_by_filter(path, {"shortName": "hlcy"})

            # Podstawowe parametry
            t2m  = safe_get_point(ds_2m,  ['t2m', '2t', 'TMP']) - 273.15
            cape = safe_get_point(ds_sfc_inst, ['cape', 'CAPE'])
            cin  = safe_get_point(ds_sfc_inst, ['cin', 'CIN'])
            li   = safe_get_point(ds_sfc_inst, ['lftx', 'LFTX'])
            pwat = safe_get_point(ds_pwat, ['pwat', 'PWAT'])

            # Poziomy temperatury i wysokości (dla LR i 0°C) — stare, sprawdzone ds
            t700 = safe_get_point(ds_700, ['t', 'TMP']) - 273.15
            t500 = safe_get_point(ds_500, ['t', 'TMP']) - 273.15
            t850 = safe_get_point(ds_850, ['t', 'TMP']) - 273.15
            h700 = safe_get_point(ds_700, ['gh', 'HGT'])
            h500 = safe_get_point(ds_500, ['gh', 'HGT'])

            # Punkt rosy — używamy szerokiego ds_isobaric (najpewniejszy sposób na DPT)
            td700 = safe_get_point_level(ds_isobaric, 700, ['dpt', 'DPT', 'td', 'dewpoint', 'DEWPT']) - 273.15
            td500 = safe_get_point_level(ds_isobaric, 500, ['dpt', 'DPT', 'td', 'dewpoint', 'DEWPT']) - 273.15
            td850 = safe_get_point_level(ds_isobaric, 850, ['dpt', 'DPT', 'td', 'dewpoint', 'DEWPT']) - 273.15

            # Wiatry
            u10  = safe_get_point(ds_10m, ['u10', '10u', 'UGRD'])
            v10  = safe_get_point(ds_10m, ['v10', '10v', 'VGRD'])
            u500 = safe_get_point(ds_500, ['u', 'UGRD'])
            v500 = safe_get_point(ds_500, ['v', 'VGRD'])
            u925 = safe_get_point(ds_925, ['u', 'UGRD'])
            v925 = safe_get_point(ds_925, ['v', 'VGRD'])

            wdir = wind_direction(u10, v10)
            wdir_compass = wind_compass(wdir)
            foehn = is_foehn_wind(wdir)

            # Obliczenia shear i LR
            dls = np.sqrt((u500 - u10)**2 + (v500 - v10)**2) if not np.isnan(u500) and not np.isnan(u10) else np.nan
            dls_01 = np.sqrt((u925 - u10)**2 + (v925 - v10)**2) if not np.isnan(u925) and not np.isnan(u10) else np.nan
            lr_700_500 = (t700 - t500) / ((h500 - h700)/1000) if all(not np.isnan(x) for x in [t700,t500,h700,h500]) else np.nan

            # Wysokość 0°C
            zero_deg_h = np.nan
            if not np.isnan(t2m) and not np.isnan(h700) and not np.isnan(t700):
                if t700 <= 0:
                    zero_deg_h = t2m * h700 / (t2m - t700)
                elif not np.isnan(t500) and not np.isnan(h500):
                    zero_deg_h = h700 + t700 * (h500 - h700) / (t700 - t500)

            # LCL
            d2m = safe_get_point(ds_2m, ['d2m', '2d', 'DPT']) - 273.15
            lcl = 125 * (t2m - d2m) if not np.isnan(t2m) and not np.isnan(d2m) else np.nan

            # SRH
            srh_3km = safe_get_point(ds_hlcy, ['hlcy', 'HLCY'])
            u_storm, v_storm = estimate_storm_motion(u10, v10, u500, v500)
            srh_01  = calc_srh_01(u10, v10, u925, v925, u_storm, v_storm)

            brn = calc_brn(cape, dls)
            stp = calc_stp(cape, srh_01, dls, lcl)

            prob = calc_storm_prob(cape, cin, li, dls, dls_01, srh_3km, srh_01, pwat, lcl, lr_700_500, brn, foehn)
            supercell_risk = calc_supercell_risk(cape, dls, srh_3km, brn, li, dls_01, srh_01, foehn)
            rot_type = supercell_rotation_type(srh_3km, supercell_risk)
            hail = estimate_hail_size(cape, lr_700_500, dls)

            # Pokazuj ryzyko superkomórki tylko gdy jest szansa na burzę
            supercell_display = supercell_risk if prob > 0 else "-"

            rows.append({
                "Czas": datetime.strptime(RUN_DATE + RUN_HOUR, "%Y%m%d%H") + timedelta(hours=fh),
                "T+": fh,
                "T2M [°C]": round(t2m, 1),
                "CAPE [J/kg]": int(round(cape, 0)) if not np.isnan(cape) else 0,
                "CIN [J/kg]": int(round(cin, 0)) if not np.isnan(cin) else 0,
                "LI [°C]": round(li, 1),
                "DLS 0-6km [m/s]": round(dls, 1),
                "DLS 0-1km [m/s]": round(dls_01, 1),
                "SRH 0-3km [m²/s²]": int(round(srh_3km, 0)) if not np.isnan(srh_3km) else 0,
                "SRH 0-1km approx [m²/s²]": round(srh_01, 1) if not np.isnan(srh_01) else np.nan,
                "BRN": round(brn, 1) if not np.isnan(brn) else np.nan,
                "STP": stp,
                "LR 700-500 [C/km]": round(lr_700_500, 1),
                "0°C Height [m]": round(zero_deg_h, 0),
                "PWAT [mm]": round(pwat, 1),
                "LCL [m]": round(lcl, 0),
                "WDIR [°]": wdir,
                "Kierunek": wdir_compass,
                "Ryzyko Superkomórki [%]": supercell_display,
                "Prob Burzy [%]": prob,
                "Grad [cm]": hail,
                "STP": stp,
                "Rotacja supercelli": rot_type
            })

        except Exception as e:
            print(f"Błąd przetwarzania f{fh:03d}: {e}")
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"Przetworzono {len(df)} rekordów | CAPE: {df['CAPE [J/kg]'].iloc[0]} | Ryzyko Superkomórki: {df['Ryzyko Superkomórki [%]'].iloc[0]}")
    return df


# Algorytmy
def calc_storm_prob(cape, cin, li, dls06, dls01, srh3, srh1, pwat, lcl, lr, brn, foehn=False):
    """Ulepszona funkcja prawdopodobieństwa burzy — bierze pod uwagę znacznie więcej parametrów + wpływ halnego"""
    if np.isnan(cape) or cape < 50:
        return 0.0

    score = 0.0

    # CAPE (podstawa)
    score += min(cape / 1200.0, 1.0) * 35

    # CIN (zbyt silne hamowanie = minus)
    if not np.isnan(cin):
        if -50 < cin < 0:
            score += 15
        elif cin < -150:
            score -= 20

    # LI (ujemny = dobre)
    if not np.isnan(li):
        if li < -4:
            score += 12
        elif li < -2:
            score += 6

    # Shear 0-6km
    if not np.isnan(dls06):
        if dls06 > 25:
            score += 12
        elif dls06 > 15:
            score += 8

    # Shear 0-1km (ważny dla tornad)
    if not np.isnan(dls01):
        if dls01 > 12:
            score += 8
        elif dls01 > 8:
            score += 4

    # SRH 0-3km
    if not np.isnan(srh3):
        if srh3 > 200:
            score += 12
        elif srh3 > 100:
            score += 6

    # SRH 0-1km (ujemny mocno hamuje)
    if not np.isnan(srh1):
        if srh1 > 80:
            score += 10
        elif srh1 < -50:
            score -= 8

    # Wilgotność (PWAT)
    if not np.isnan(pwat):
        if pwat > 35:
            score += 8
        elif pwat > 25:
            score += 4

    # LCL (niski = lepsze dla tornad)
    if not np.isnan(lcl):
        if lcl < 800:
            score += 6
        elif lcl > 1800:
            score -= 5

    # Lapse rate 700-500
    if not np.isnan(lr):
        if lr > 7.5:
            score += 6
        elif lr > 6.5:
            score += 3

    # Indeksy klasyczne (jeśli dostępne)
    if not np.isnan(brn) and 10 < brn < 50:
        score += 6

    # === WPŁYW WIATRU HALNEGO (Krosno) ===
    if foehn:
        score *= 0.65   # Halny mocno osłabia burze (osuszanie + stabilizacja)

    return float(np.clip(np.round(score, 0), 0, 100))


def calc_supercell_risk(cape, dls06, srh3, brn, li, dls01, srh1, foehn=False):
    """Ryzyko superkomórki (0-100) — uwzględnia wpływ halnego w Krośnie"""
    if np.isnan(cape) or cape < 200:
        return 5.0

    score = 0.0

    # CAPE (bardziej hojne w średnim zakresie)
    if cape > 1500:
        score += 30
    elif cape > 800:
        score += 22
    elif cape > 400:
        score += 12

    # Deep layer shear
    if not np.isnan(dls06):
        if dls06 > 22:
            score += 18
        elif dls06 > 15:
            score += 10

    # SRH 0-3km — najważniejszy wskaźnik
    if not np.isnan(srh3):
        if srh3 > 200:
            score += 28
        elif srh3 > 120:
            score += 18
        elif srh3 > 70:
            score += 8

    # BRN w złotym zakresie
    if not np.isnan(brn):
        if 12 < brn < 48:
            score += 12
        elif brn < 8 or brn > 70:
            score -= 4

    # LI
    if not np.isnan(li) and li < -2.5:
        score += 6

    # Ujemny SRH 0-1km lekko obniża
    if not np.isnan(srh1) and srh1 < -50:
        score -= 6

    # DLS 0-1km
    if not np.isnan(dls01) and dls01 > 8:
        score += 4

    # === WPŁYW WIATRU HALNEGO (Krosno) ===
    # Halny z południa mocno osłabia burze poprzez osuszanie i stabilizację
    if foehn:
        score *= 0.55

    return float(np.clip(np.round(score, 0), 0, 100))


def estimate_hail_size(cape, lr, dls):
    if np.isnan(cape) or cape < 400: return 0.0
    hail = (cape / 1000.0) * (lr / 6.5 if not np.isnan(lr) else 1.0)
    if not np.isnan(dls) and dls > 20: hail *= 1.3
    return float(np.round(np.clip(hail, 0, 8), 1))


# -----------------------
# POBIERANIE + ZAPIS + FTP
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
        workbook = writer.book
        ws = writer.sheets['Burze']
        fmt_red = workbook.add_format({'bg_color': '#FF3333', 'font_color': 'white'})
        fmt_orange = workbook.add_format({'bg_color': '#FFA500', 'font_color': 'black'})
        fmt_green = workbook.add_format({'bg_color': '#90EE90', 'font_color': 'black'})

        # CAPE (kolumna D)
        ws.conditional_format('D2:D200', {'type': 'cell', 'criteria': '>=', 'value': 1000, 'format': fmt_red})
        # Grad (kolumna S)
        ws.conditional_format('S2:S200', {'type': 'cell', 'criteria': '>=', 'value': 1.5, 'format': fmt_red})

        # SRH 0-3km (kolumna I)
        ws.conditional_format('I2:I200', {'type': 'cell', 'criteria': '>=', 'value': 150, 'format': fmt_red})
        # BRN (kolumna L po usunięciu Total Totals) - zakres 10-45 korzystny dla supercell
        ws.conditional_format('L2:L200', {'type': 'cell', 'criteria': '>=', 'value': 10, 'format': fmt_green})
        ws.conditional_format('L2:L200', {'type': 'cell', 'criteria': '<=', 'value': 45, 'format': fmt_green})

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
    print(f"\n=== GFS CONVECTION MODULE {RUN_DATE}{RUN_HOUR}Z (rozbudowany) ===")
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
