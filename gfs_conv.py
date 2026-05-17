#!/usr/bin/env python3
# gfs_krosno_conv_v3.1.py - Czysta wersja bez ukrytych znaków (U+00A0)
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
elif time(8, 30) <= current_time < time(15, 30):
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
    "&var_TMP=on&var_HGT=on&var_UGRD=on&var_VGRD=on&var_CAPE=on&var_CIN=on"
    "&var_LFTX=on&var_PWAT=on&var_HLCY=on&var_DPT=on&var_RH=on&var_SPFH=on&var_VVEL=on"
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

def calc_srh_01(u10, v10, u925, v925, u_storm, v_storm):
    if any(np.isnan(x) for x in [u10, v10, u925, v925, u_storm, v_storm]):
        return np.nan
    return float(np.round((u10 - u_storm) * (v925 - v10) - (v10 - v_storm) * (u925 - u10), 1))

def wind_direction(u, v):
    if np.isnan(u) or np.isnan(v):
        return np.nan
    return round((270 - np.rad2deg(np.arctan2(v, u))) % 360, 1)

def is_foehn_wind(wdir):
    if np.isnan(wdir):
        return False
    return 157.5 <= wdir <= 247.5

def calc_brn(cape, dls_06):
    if np.isnan(cape) or np.isnan(dls_06) or dls_06 < 1 or cape <= 0:
        return np.nan
    return float(np.round(cape / (0.5 * dls_06 ** 2), 1))

def calc_stp(cape, srh1, dls06, lcl):
    if any(np.isnan(x) for x in [cape, srh1, dls06, lcl]):
        return np.nan
    return float(round(max(0, min((cape/1500) * (srh1/150) * (dls06/20) * max(0, (2000 - lcl)/1000), 8.0)), 2))

def supercell_rotation_type(srh3, supercell_risk=None):
    if np.isnan(srh3) or (supercell_risk is not None and supercell_risk <= 20):
        return "Neutralna"
    if srh3 > 50:
        return "Prawoskrętna"
    elif srh3 < -30:
        return "Lewoskrętna"
    return "Neutralna"

def calc_storm_prob(cape, cin, li, dls06, dls01, srh3, srh1, pwat, lcl, lr, brn, foehn=False):
    if np.isnan(cape) or cape < 50:
        return 0.0
    score = min(cape / 1200.0, 1.0) * 35
    if not np.isnan(cin):
        if -50 < cin < 0:
            score += 15
        elif cin < -150:
            score -= 20
    if not np.isnan(li):
        if li < -4:
            score += 12
        elif li < -2:
            score += 6
    if not np.isnan(dls06):
        if dls06 > 25:
            score += 12
        elif dls06 > 15:
            score += 8
    if not np.isnan(dls01):
        if dls01 > 12:
            score += 8
        elif dls01 > 8:
            score += 4
    if not np.isnan(srh3):
        if srh3 > 200:
            score += 12
        elif srh3 > 100:
            score += 6
    if not np.isnan(srh1):
        if srh1 > 80:
            score += 10
        elif srh1 < -50:
            score -= 8
    if not np.isnan(pwat):
        if pwat > 35:
            score += 8
        elif pwat > 25:
            score += 4
    if not np.isnan(lcl):
        if lcl < 800:
            score += 6
        elif lcl > 1800:
            score -= 5
    if not np.isnan(lr):
        if lr > 7.5:
            score += 6
        elif lr > 6.5:
            score += 3
    if not np.isnan(brn) and 10 < brn < 50:
        score += 6
    if foehn:
        score *= 0.65
    return float(np.clip(np.round(score, 0), 0, 100))

def calc_supercell_risk(cape, dls06, srh3, brn, li, dls01, srh1, foehn=False):
    if np.isnan(cape) or cape < 200:
        return 5.0
    score = 0.0
    if cape > 1500:
        score += 30
    elif cape > 800:
        score += 22
    elif cape > 400:
        score += 12
    if not np.isnan(dls06):
        if dls06 > 22:
            score += 18
        elif dls06 > 15:
            score += 10
    if not np.isnan(srh3):
        if srh3 > 200:
            score += 28
        elif srh3 > 120:
            score += 18
        elif srh3 > 70:
            score += 8
    if not np.isnan(brn):
        if 12 < brn < 48:
            score += 12
        elif brn < 8 or brn > 70:
            score -= 4
    if not np.isnan(li) and li < -2.5:
        score += 6
    if not np.isnan(srh1) and srh1 < -50:
        score -= 6
    if not np.isnan(dls01) and dls01 > 8:
        score += 4
    if foehn:
        score *= 0.55
    return float(np.clip(np.round(score, 0), 0, 100))

def estimate_hail_size(cape, lr, dls):
    if np.isnan(cape) or cape < 400:
        return 0.0
    hail = (cape / 1000.0) * (lr / 6.5 if not np.isnan(lr) else 1.0)
    if not np.isnan(dls) and dls > 20:
        hail *= 1.3
    return float(np.round(np.clip(hail, 0, 8), 1))

def extract_profile_for_metpy(ds_isobaric):
    if ds_isobaric is None or not METPY_AVAILABLE:
        return None
    try:
        ds_point = ds_isobaric.sel(latitude=KROSNO_LAT, longitude=KROSNO_LON, method="nearest")
        level_name = 'level' if 'level' in ds_point.coords else 'isobaricInhPa' if 'isobaricInhPa' in ds_point.coords else [k for k in ds_point.coords if 'hPa' in str(k).lower()][0]
        levels = ds_point[level_name].values.astype(float)
        idx = np.argsort(-levels)
        levels_sorted = levels[idx]
        p = levels_sorted * units.hPa
        t = (ds_point['t'] if 't' in ds_point.data_vars else ds_point['TMP']).sel({level_name: levels_sorted}).values * units.K
        td = ds_point['dpt'].sel({level_name: levels_sorted}).values * units.K if 'dpt' in ds_point.data_vars else t - 8 * units.K
        u = ds_point['u'].sel({level_name: levels_sorted}).values * units('m/s')
        v = ds_point['v'].sel({level_name: levels_sorted}).values * units('m/s')
        hgt = (ds_point['gh'] if 'gh' in ds_point.data_vars else ds_point['HGT']).sel({level_name: levels_sorted}).values * units.meter
        return p, t, td, u, v, hgt
    except:
        return None

def calc_ship(mucape, mu_mixing_ratio, lr_700_500, t500, dls06):
    if any(np.isnan(x) for x in [mucape, mu_mixing_ratio, lr_700_500, t500, dls06]):
        return np.nan
    return round(max(0, min((mucape * mu_mixing_ratio * lr_700_500 * abs(t500) * dls06) / 44000000, 5.0)), 2)

def calc_ehi(cape, srh):
    return round((cape * srh) / 160000, 2) if not np.isnan(cape) and not np.isnan(srh) and cape > 0 else 0.0

def calc_full_stp(sbcape, lcl_h, srh01, dls06, sbcin):
    if any(np.isnan(x) for x in [sbcape, lcl_h, srh01, dls06, sbcin]):
        return np.nan
    return round(max(0, min((min(sbcape/1500,1.5) * max(0,(2000-lcl_h)/1000) * min(srh01/150,1.5) * min(dls06/20,1.5) * max(0,(200+sbcin)/150)), 8.0)), 2)

def calc_heavy_rain_potential(pwat, rh850, rh700, vvel850, dls06, foehn=False, orographic=1.0):
    if np.isnan(pwat) or pwat < 20:
        return 0.0
    score = min(pwat/40,1)*35 + (min(rh850/90,1)*20 if not np.isnan(rh850) else 0) + (min(rh700/85,1)*15 if not np.isnan(rh700) else 0)
    if not np.isnan(vvel850) and vvel850 < 0:
        score += min(abs(vvel850)/0.8,1)*15
    if not np.isnan(dls06) and 10 < dls06 < 25:
        score += 10
    score *= orographic
    if foehn:
        score *= 0.6
    return float(np.clip(round(score,0),0,100))

def classify_storm_mode(cape, dls06, srh3, cin, lcl, prob):
    if prob < 15 or np.isnan(cape) or cape < 100:
        return "Brak / Słaba"
    if cin < -150 and cape > 800:
        return "Elevated"
    if dls06 < 10:
        return "Pulse / Zwykła"
    if dls06 > 25 and srh3 > 150 and cape > 800:
        return "Supercell (prawdopodobna)"
    if dls06 > 20 and srh3 < 80:
        return "QLCS / Squall Line"
    if 12 < dls06 < 22 and cape > 600:
        return "Multicell"
    return "Zwykła / Multicell"

def calc_orographic_factor(wdir, wspd):
    if np.isnan(wdir) or np.isnan(wspd):
        return 1.0
    angle_diff = min(abs(wdir - 220), 360 - abs(wdir - 220))
    return round(min(1.0 + 0.6 * np.sin(np.radians(angle_diff)) * min(wspd/15, 1.2), 1.8), 2)

def download_missing_gribs_parallel(forecast_hours):
    pending = [fh for fh in forecast_hours if not os.path.exists(os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")) or os.path.getsize(os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")) < 45000]
    if not pending:
        return
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
        except:
            pass
        return False

    with ThreadPoolExecutor(max_workers=10) as ex:
        ex.map(fetch_single, pending)

def process_local_gribs(forecast_hours):
    rows = []
    for fh in forecast_hours:
        path = os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")
        if not os.path.exists(path):
            continue
        try:
            print(f"  f{fh:03d}...", end=" ")
            ds_sfc = try_open_by_filter(path, {"typeOfLevel": "surface", "stepType": "instant"})
            ds_2m = try_open_by_filter(path, {"typeOfLevel": "heightAboveGround", "level": 2})
            ds_10m = try_open_by_filter(path, {"typeOfLevel": "heightAboveGround", "level": 10})
            ds_pwat = try_open_by_filter(path, {"typeOfLevel": "atmosphereSingleLayer"})
            ds_isobaric = try_open_by_filter(path, {"typeOfLevel": "isobaricInhPa"})
            ds_hlcy = try_open_by_filter(path, {"shortName": "hlcy"})

            t2m = safe_get_point(ds_2m, ['t2m', '2t', 'TMP']) - 273.15
            cape = safe_get_point(ds_sfc, ['cape', 'CAPE'])
            cin = safe_get_point(ds_sfc, ['cin', 'CIN'])
            li = safe_get_point(ds_sfc, ['lftx', 'LFTX'])
            pwat = safe_get_point(ds_pwat, ['pwat', 'PWAT'])
            srh3 = safe_get_point(ds_hlcy, ['hlcy', 'HLCY'])

            t850 = safe_get_point(ds_isobaric, ['t', 'TMP']) - 273.15 if 'level' in ds_isobaric.coords else np.nan
            t700 = safe_get_point(ds_isobaric, ['t', 'TMP']) - 273.15 if 'level' in ds_isobaric.coords else np.nan
            t500 = safe_get_point(ds_isobaric, ['t', 'TMP']) - 273.15 if 'level' in ds_isobaric.coords else np.nan
            td850 = safe_get_point(ds_isobaric, ['dpt', 'DPT']) - 273.15 if 'level' in ds_isobaric.coords else np.nan
            rh850 = safe_get_point(ds_isobaric, ['r', 'RH']) if 'level' in ds_isobaric.coords else np.nan
            rh700 = safe_get_point(ds_isobaric, ['r', 'RH']) if 'level' in ds_isobaric.coords else np.nan
            vvel850 = safe_get_point(ds_isobaric, ['w', 'VVEL']) if 'level' in ds_isobaric.coords else np.nan

            u10 = safe_get_point(ds_10m, ['u10', '10u', 'UGRD'])
            v10 = safe_get_point(ds_10m, ['v10', '10v', 'VGRD'])
            u500 = safe_get_point(ds_isobaric, ['u', 'UGRD']) if 'level' in ds_isobaric.coords else np.nan
            v500 = safe_get_point(ds_isobaric, ['v', 'VGRD']) if 'level' in ds_isobaric.coords else np.nan
            u925 = safe_get_point(ds_isobaric, ['u', 'UGRD']) if 'level' in ds_isobaric.coords else np.nan
            v925 = safe_get_point(ds_isobaric, ['v', 'VGRD']) if 'level' in ds_isobaric.coords else np.nan

            wdir = wind_direction(u10, v10)
            foehn = is_foehn_wind(wdir)
            orog_factor = calc_orographic_factor(wdir, np.hypot(u10, v10) if not np.isnan(u10) else np.nan)

            dls06 = np.hypot(u500 - u10, v500 - v10) if not any(np.isnan([u500, u10])) else np.nan
            dls01 = np.hypot(u925 - u10, v925 - v10) if not any(np.isnan([u925, u10])) else np.nan
            lr_700_500 = (t700 - t500) / 2.0 if not any(np.isnan([t700, t500])) else np.nan
            brn = calc_brn(cape, dls06)

            u_storm, v_storm = estimate_storm_motion(u10, v10, u500, v500)
            srh_01 = calc_srh_01(u10, v10, u925, v925, u_storm, v_storm)
            lcl = 125 * (t2m - (td850 if not np.isnan(td850) else t2m - 8)) if not np.isnan(t2m) else np.nan
            stp_old = calc_stp(cape, srh_01, dls06, lcl)
            prob_old = calc_storm_prob(cape, cin, li, dls06, dls01, srh3, srh_01, pwat, lcl, lr_700_500, brn, foehn)
            supercell_risk = calc_supercell_risk(cape, dls06, srh3, brn, li, dls01, srh_01, foehn)
            rot_type = supercell_rotation_type(srh3, supercell_risk)
            hail = estimate_hail_size(cape, lr_700_500, dls06)

            mucape = dcape = ship = ehi = stp_full = np.nan
            heavy_rain = 0.0
            storm_mode = "Brak danych"

            profile = extract_profile_for_metpy(ds_isobaric)
            if profile and METPY_AVAILABLE:
                p, t, td, u, v, hgt = profile
                try:
                    sbcape, sbcin = mpcalc.surface_based_cape_cin(p, t, td)
                    mucape_val, _ = mpcalc.most_unstable_cape_cin(p, t, td)
                    mucape = float(mucape_val.magnitude)
                    lcl_p, _ = mpcalc.lcl(p[0], t[0], td[0])
                    lcl_h = float(mpcalc.pressure_to_height_std(lcl_p).to('m').magnitude)
                    u_storm2, v_storm2, _ = mpcalc.bunkers_storm_motion(p, u, v, hgt)
                    srh01_val = mpcalc.storm_relative_helicity(hgt, u, v, depth=1*units.km, storm_u=u_storm2, storm_v=v_storm2)[0]
                    srh03_val = mpcalc.storm_relative_helicity(hgt, u, v, depth=3*units.km, storm_u=u_storm2, storm_v=v_storm2)[0]
                    dcape_val = mpcalc.downdraft_cape(p, t, td)
                    dcape = float(dcape_val.magnitude)
                    mu_mr = float(mpcalc.mixing_ratio_from_specific_humidity(mpcalc.specific_humidity_from_dewpoint(p[0], td[0])).magnitude * 1000)
                    ship = calc_ship(mucape, mu_mr, lr_700_500, t500, dls06)
                    ehi = calc_ehi(cape, float(srh03_val.magnitude))
                    stp_full = calc_full_stp(float(sbcape.magnitude), lcl_h, float(srh01_val.magnitude), dls06, float(sbcin.magnitude))
                except:
                    pass

            heavy_rain = calc_heavy_rain_potential(pwat, rh850, rh700, vvel850, dls06, foehn, orog_factor)
            prob_temp = min(cape / 1200 * 40 + (srh3 / 200 * 20 if not np.isnan(srh3) else 0), 100)
            storm_mode = classify_storm_mode(cape, dls06, srh3, cin, lcl, prob_temp)

            prob_tornado = min(max(0, (stp_full or 0) * 25 + (srh3 / 300 * 30 if not np.isnan(srh3) else 0)), 100)
            prob_grad = min(max(0, (ship or 0) * 35 + (cape / 2000 * 25)), 100)
            prob_ulewa = heavy_rain

            if foehn:
                prob_old = int(prob_old * 0.65)
                prob_tornado = int(prob_tornado * 0.5)
                prob_grad = int(prob_grad * 0.6)

            rows.append({
                "Czas": datetime.strptime(RUN_DATE + RUN_HOUR, "%Y%m%d%H") + timedelta(hours=fh),
                "T+": fh,
                "T2M [°C]": round(t2m, 1),
                "CAPE [J/kg]": int(round(cape)) if not np.isnan(cape) else 0,
                "MUCAPE [J/kg]": int(round(mucape)) if not np.isnan(mucape) else "-",
                "CIN [J/kg]": int(round(cin)) if not np.isnan(cin) else 0,
                "DCAPE [J/kg]": int(round(dcape)) if not np.isnan(dcape) else "-",
                "DLS 0-6km [m/s]": round(dls06, 1) if not np.isnan(dls06) else "-",
                "DLS 0-1km [m/s]": round(dls01, 1) if not np.isnan(dls01) else "-",
                "SRH 0-3km": int(round(srh3)) if not np.isnan(srh3) else 0,
                "SRH 0-1km approx": round(srh_01, 1) if not np.isnan(srh_01) else "-",
                "BRN": round(brn, 1) if not np.isnan(brn) else "-",
                "STP (stary)": stp_old,
                "STP (pełny MetPy)": stp_full if not np.isnan(stp_full) else "-",
                "SHIP": ship if not np.isnan(ship) else "-",
                "EHI": ehi if not np.isnan(ehi) else "-",
                "PWAT [mm]": round(pwat, 1) if not np.isnan(pwat) else "-",
                "LCL [m]": round(lcl, 0) if not np.isnan(lcl) else "-",
                "Heavy Rain Pot [%]": heavy_rain,
                "Prob Burzy [%]": prob_old,
                "Prob Tornado [%]": round(prob_tornado, 0),
                "Prob Grad [%]": round(prob_grad, 0),
                "Prob Ulewa [%]": prob_ulewa,
                "Storm Mode": storm_mode,
                "Rotacja": rot_type,
                "Grad [cm]": hail,
                "Halny": "TAK" if foehn else "NIE",
                "Orografia": orog_factor
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
    if df.empty:
        return []
    csv_path = os.path.join(OUTPUT_DIR, "gfs-conv.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    xlsx_path = os.path.join(OUTPUT_DIR, f"krosno_conv_{RUN_DATE}_{RUN_HOUR}z.xlsx")
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Burze_v2')
        ws = writer.sheets['Burze_v2']
        red = writer.book.add_format({'bg_color': '#FF3333', 'font_color': 'white'})
        ws.conditional_format('D2:D300', {'type': 'cell', 'criteria': '>=', 'value': 1000, 'format': red})
        ws.conditional_format('P2:P300', {'type': 'cell', 'criteria': '>=', 'value': 1.2, 'format': red})
        ws.conditional_format('R2:R300', {'type': 'cell', 'criteria': '>=', 'value': 1.0, 'format': red})
        ws.conditional_format('S2:S300', {'type': 'cell', 'criteria': '>=', 'value': 70, 'format': red})
    return [csv_path, xlsx_path]

def upload_to_ftp(files):
    load_dotenv()
    host, user, pswd = os.getenv("FTP_HOST"), os.getenv("FTP_USER"), os.getenv("FTP_PASS")
    if not all([host, user, pswd]):
        return
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
    print(f"\n=== GFS CONVECTION v3.1 {RUN_DATE}{RUN_HOUR}Z ===\n")
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
