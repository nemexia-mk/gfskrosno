#!/usr/bin/env python3
# gfs_krosno.py
# Pobiera GFS, generuje Excel + meteorogram 120h + daily PNG; retry co 10 min jeśli brakuje plików; wysyła wyniki na FTP (credentials z .env)

import os
import requests
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
from matplotlib.dates import DateFormatter
from time import sleep
from dotenv import load_dotenv
from ftplib import FTP, error_perm

# -----------------------
# CONFIG
# -----------------------
OUTPUT_DIR = "gfs_krosno_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Krosno
TOP_LAT = 50.0
BOTTOM_LAT = 49.4
LEFT_LON = 21.3
RIGHT_LON = 22.01
KROSNO_LAT = 49.69
KROSNO_LON = 21.77

# GFS cycle selection (dynamic based on UTC now)
now = datetime.utcnow()
RUN_DATE = now.strftime("%Y%m%d")

# time cutoffs (UTC) for selecting run hour (matches Twoja logika)
t_0120 = time(1, 20)
t_0520 = time(5, 20)
t_1120 = time(11, 20)
t_1720 = time(17, 20)

if t_0120 <= now.time() < t_0520:
    RUN_HOUR = "18"
elif t_0520 <= now.time() < t_1120:
    RUN_HOUR = "00"
elif t_1120 <= now.time() < t_1720:
    RUN_HOUR = "06"
else:
    RUN_HOUR = "12"

CYCLE_DIR = f"gfs.{RUN_DATE}/{RUN_HOUR}/atmos"

BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

FORECAST_HOURS = list(range(0, 385, 3))  # pełny zakres; wykres będzie ograniczony do 120h

# Retry policy
RETRY_INTERVAL_SECONDS = 10 * 60   # 10 minut
MAX_RETRIES = 12                    # domyślnie spróbuj 12 razy (2 godziny); ustaw None aby próbować w nieskończoność

# Static NOMADS filter
STATIC_MIDDLE = (
    "&lev_2_m_above_ground=on"
    "&lev_10_m_above_ground=on"
    "&lev_850_mb=on"
    "&lev_surface=on"
    "&lev_mean_sea_level=on"
    "&lev_entire_atmosphere_%28considered_as_a_single_layer%29=on"
    "&lev_low_cloud_layer=on"
    "&lev_middle_cloud_layer=on"
    "&lev_high_cloud_layer=on"
    "&var_TMP=on"
    "&var_TCDC=on"
    "&var_LCDC=on"
    "&var_MCDC=on"
    "&var_HCDC=on"
    "&var_RH=on"
    "&var_PRATE=on"
    "&var_APCP=on"
    "&var_GUST=on"
    "&var_UGRD=on"
    "&var_VGRD=on"
    "&var_PRES=on"
    "&var_PRMSL=on"
    "&var_DPT=on"
    "&var_DSWRF=on"
    "&var_USWRF=on"
    "&var_DLWRF=on"
    "&var_ULWRF=on"
    "&var_CAPE=on"
    "&var_LFTX=on"
    "&var_WEASD=on"
    "&var_VIS=on"
    "&subregion=on"
    f"&toplat={TOP_LAT}"
    f"&bottomlat={BOTTOM_LAT}"
    f"&leftlon={LEFT_LON}"
    f"&rightlon={RIGHT_LON}"
)

# -----------------------
# HELPERS (same jak wcześniej)
# -----------------------
def build_url(file_name):
    url = f"{BASE_URL}?file={file_name}&dir=/{CYCLE_DIR}{STATIC_MIDDLE}"
    return url.replace("suubregion", "subregion").replace("lev_entire_atmoosphere", "lev_entire_atmosphere")

SHORTNAMES = {
    "t2m": ["t2m", "2t", "tmp2m", "tmp"],
    "d2m": ["d2m", "dew2m", "dpt"],
    "r2": ["r2", "rh2", "rh"],
    "gust": ["gust"],
    "tcc": ["tcc", "tcdc"],
    "lcc": ["lcc", "lcdc"],
    "mcc": ["mcc", "mcdc"],
    "hcc": ["hcc", "hcdc"],
    "prate": ["prate"],
    "apcp": ["apcp", "tp"],
    "msl": ["msl", "pres", "prmsl", "sp"],
    "dpt": ["dpt", "d2m"],
    "t850": ["t", "tmp"],
    "weasd": ["weasd", "snow", "snowsurf", "sdwe"],
    "cape": ["cape", "sbcape"],
    "lftx": ["lftx"],
    "vis": ["vis"],
    "u10": ["ugrd", "u10"],
    "v10": ["vgrd", "v10"]
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
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if name in ("t2m","t850","d2m","tmax","tmin",):
        return float(np.round(val - 273.15, 1))
    if name in ("sp","msl","pres"):
        return float(np.round(val / 100.0, 1))
    if name in ("prate",):
        return float(np.round(val * 3600.0 * 3.0, 2))
    if name in ("apcp",):
        return float(np.round(val, 2))
    if name in ("tcc","lcc","mcc","hcc","r2"):
        v = float(val)
        if v > 1.5 and v < 100:
            v = v
        elif v > 100:
            v = v / 100.0
        return float(np.round(v, 1))
    if name in ("u10","v10","wind_ms","gust"):
        return float(np.round(val, 1))
    if name in ("vis",):
        return float(np.round(val / 1000, 1))
    if name in ("weasd",):
        return float(np.round(val * 10, 1))
    return float(np.round(val, 2))

def lcl_height_m(t_c, td_c):
    if np.isnan(t_c) or np.isnan(td_c):
        return np.nan
    diff = t_c - td_c
    if diff < 0:
        diff = 0.0
    return float(np.round(125.0 * diff, 1))

def detect_precip_type(prate, t2m_c, t850_c):
    if prate is None or (isinstance(prate, float) and np.isnan(prate)) or prate <= 0:
        return "Brak"
    if np.isnan(t2m_c):
        return "Brak"
    try:
        if t2m_c <= 0 and (not np.isnan(t850_c) and t850_c < 0):
            return "Śnieg"
        elif 0 < t2m_c < 2 and (not np.isnan(t850_c) and t850_c < 0):
            return "Deszcz ze śniegiem"
        elif t2m_c < 0 and (not np.isnan(t850_c) and t850_c > 0):
            return "Deszcz marznący"
        else:
            return "Deszcz"
    except Exception:
        return "Brak"

def storm_risk_category(cape, li):
    if np.isnan(cape) and np.isnan(li):
        return "Brak"
    cape_val = 0.0 if np.isnan(cape) else cape
    if cape_val < 100:
        cat = "Niskie"
    elif 100 <= cape_val <= 400:
        cat = "Niskie"
    elif 401 <= cape_val <= 1000:
        cat = "Średnie"
    elif 1001 <= cape_val <= 2000:
        cat = "Wysokie"
    else:
        cat = "Ekstremalne"
    if not np.isnan(li):
        if li <= -4:
            if cat == "Niskie": cat = "Średnie"
            elif cat == "Średnie": cat = "Wysokie"
            elif cat == "Wysokie": cat = "Ekstremalne"
        elif li <= -2:
            if cat == "Niskie": cat = "Średnie"
            elif cat == "Średnie": cat = "Wysokie"
            elif cat == "WysOKie": cat = "Ekstremalne"
    return cat

# color map for Excel - minimal set
PREC_TYPE_TO_COLOR = {
    "Deszcz": "#0FB00F",
    "Śnieg": "#ADD8E6",
    "Deszcz ze śniegiem": "#00FFBB",
    "Deszcz marznący": "#FFA500",
}

# -----------------------
# DOWNLOAD with RETRY logic
# -----------------------
def download_missing_gribs(forecast_hours, max_retries=MAX_RETRIES, retry_interval=RETRY_INTERVAL_SECONDS):
    """Pobieraj pliki GRIB; jeśli brakujące - próbuj ponownie co retry_interval sekund.
       Zwraca listę lokalnych plików które zostały pobrane (pełne ścieżki)."""
    pending = set(forecast_hours)
    downloaded_files = set()

    # If file already exists (from previous runs) assume ok and remove from pending (but we could validate)
    for fh in list(pending):
        local_path = os.path.join(OUTPUT_DIR, f"krosno_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            downloaded_files.add(local_path)
            pending.remove(fh)

    attempt = 0
    while pending and (max_retries is None or attempt < max_retries):
        attempt += 1
        print(f"\n>>> Pobieranie brakujących plików — próba {attempt}. Pozostało godzin: {sorted(pending)}")
        for fh in sorted(list(pending)):
            fstr = f"{fh:03d}"
            grib_filename = f"gfs.t{RUN_HOUR}z.pgrb2.0p25.f{fstr}"
            local_path = os.path.join(OUTPUT_DIR, f"krosno_{RUN_DATE}_{RUN_HOUR}z_f{fstr}.grib2")
            # If exists and appears valid, skip
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                downloaded_files.add(local_path)
                pending.remove(fh)
                continue

            url = build_url(grib_filename)
            try:
                r = requests.get(url, headers=HEADERS, timeout=90)
            except Exception as e:
                print(f" - Błąd requestu dla f{fstr}: {e}")
                continue

            if r.status_code != 200 or b"GRIB" not in r.content[:10]:
                print(f" - NOMADS zwrócił {r.status_code} (f{fstr}) — pomijam na teraz")
                continue

            try:
                with open(local_path, "wb") as f:
                    f.write(r.content)
                print(f" - Zapisano: {local_path} ({os.path.getsize(local_path)/1024:.1f} KB)")
                downloaded_files.add(local_path)
                pending.remove(fh)
            except Exception as e:
                print(f" - Nie udało się zapisać f{fstr}: {e}")

        if pending:
            print(f"Nie wszystkie pliki pobrane. Kolejna próba za {retry_interval//60} minut.")
            sleep(retry_interval)

    if pending:
        print("\nUwaga: nie udało się pobrać wszystkich prognoz w przydzielonym czasie. Brakujące godziny:", sorted(pending))
    else:
        print("\nWszystkie żądane pliki pobrane.")

    return downloaded_files, sorted(list(pending))

# -----------------------
# CORE: przetwarzanie lokalnych plików GRIB -> df (działa nawet gdy nie ma wszystkich godzin)
# -----------------------
def process_local_gribs(forecast_hours):
    rows = []
    for fh in forecast_hours:
        local_path = os.path.join(OUTPUT_DIR, f"krosno_{RUN_DATE}_{RUN_HOUR}z_f{fh:03d}.grib2")
        if not os.path.exists(local_path):
            # brak pliku - pomiń (będzie niepełna tabela)
            continue

        # Open datasets like in poprzednim skrypcie
        ds_surface_instant = try_open_by_filter(local_path, {"typeOfLevel": "surface", "stepType": "instant"})
        ds_surface_accum = try_open_by_filter(local_path, {"typeOfLevel": "surface", "stepType": "accum"})
        ds_2m = try_open_by_filter(local_path, {"typeOfLevel": "heightAboveGround", "level": 2})
        ds_10m = try_open_by_filter(local_path, {"typeOfLevel": "heightAboveGround", "level": 10})
        ds_msl = try_open_by_filter(local_path, {"typeOfLevel": "meanSea"})
        ds_low = try_open_by_filter(local_path, {"typeOfLevel": "lowCloudLayer"})
        ds_mid = try_open_by_filter(local_path, {"typeOfLevel": "middleCloudLayer"})
        ds_high = try_open_by_filter(local_path, {"typeOfLevel": "highCloudLayer"})
        ds_t850 = try_open_by_filter(local_path, {"typeOfLevel": "isobaricInhPa", "level": 850})

        def get_val(datasets, key):
            if not isinstance(datasets, list):
                datasets = [datasets]
            for ds in datasets:
                if ds is not None:
                    val = safe_get_point(ds, SHORTNAMES.get(key, [key]))
                    if not np.isnan(val):
                        return val
            return np.nan

        try:
            t2m = convert_and_round(get_val(ds_2m, "t2m"), "t2m")
            d2m = convert_and_round(get_val(ds_2m, "d2m"), "d2m")
            t850_val = convert_and_round(get_val(ds_t850, "t850"), "t850")
            sp = convert_and_round(get_val([ds_msl, ds_surface_instant], "msl"), "msl")
            tcc = convert_and_round(get_val([ds_high, ds_mid, ds_low, ds_t850], "tcc"), "tcc")
            lcc = convert_and_round(get_val(ds_low, "lcc"), "lcc")
            mcc = convert_and_round(get_val(ds_mid, "mcc"), "mcc")
            hcc = convert_and_round(get_val(ds_high, "hcc"), "hcc")
            prate_mm3h = convert_and_round(get_val([ds_surface_accum, ds_surface_instant], "prate"), "prate")
            snow_cm = convert_and_round(get_val([ds_surface_instant, ds_surface_accum], "weasd"), "weasd")
            cape = convert_and_round(get_val(ds_surface_instant, "cape"), "cape")
            lftx = convert_and_round(get_val(ds_surface_instant, "lftx"), "lftx")
            u10_val = get_val(ds_10m, "u10")
            v10_val = get_val(ds_10m, "v10")
            wind_ms = np.nan
            wind_dir = np.nan
            if not np.isnan(u10_val) and not np.isnan(v10_val):
                wind_ms = convert_and_round(np.sqrt(u10_val**2 + v10_val**2), "wind_ms")
                wind_dir = convert_and_round((np.degrees(np.arctan2(-u10_val, -v10_val)) + 360) % 360, "wind_dir")
            gust = convert_and_round(get_val([ds_surface_instant, ds_10m], "gust"), "gust")
            vis_km = convert_and_round(get_val([ds_surface_instant], "vis"), "vis")

            run_dt = datetime.strptime(RUN_DATE + RUN_HOUR, "%Y%m%d%H")
            valid_time = run_dt + timedelta(hours=fh)

            rows.append({
                "Czas": valid_time,
                "T+ (h)": fh,
                "T2M [°C]": t2m,
                "D2M [°C]": d2m,
                "T850 [°C]": t850_val,
                "MSLP [hPa]": sp,
                "CL [%]": lcc,
                "CM [%]": mcc,
                "CH [%]": hcc,
                "CC [%]": tcc,
                "RRR [mm/3h]": prate_mm3h,
                "SNOW [cm]": snow_cm,
                "WSPD [m/s]": wind_ms,
                "GUST [m/s]": gust,
                "WDIR [°]": wind_dir,
                "CAPE [J/kg]": cape,
                "LIFTED [°C]": lftx,
                "VIS [km]": vis_km
            })
        except Exception as e:
            print(f" - Błąd przetwarzania pliku {local_path}: {e}")
            continue

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame()

    # sort and compute daily stats
    df.sort_values("Czas", inplace=True)
    df.reset_index(drop=True, inplace=True)
    if not df.empty:
        df["Date"] = df["Czas"].dt.date
        daily_stats = df.groupby("Date").agg({
            "T2M [°C]": ["max","min"],
            "RRR [mm/3h]": "sum",
            "WSPD [m/s]": "mean",
            "MSLP [hPa]": "mean",
            "CAPE [J/kg]": "max",
            "LIFTED [°C]": "min",
            "VIS [km]": "min"
        })
        daily_stats.columns = ["Tmax","Tmin","Suma_opadu","Wsp_sred","Pres_sred","CAPE_max","LIFTED_min","VIS_min"]
        daily = daily_stats.reset_index()
        daily_temp_mean = df.groupby("Date")["T2M [°C]"].mean().reset_index(name="T_mean")
        daily_dew_mean = df.groupby("Date")["D2M [°C]"].mean().reset_index(name="Td_mean")
        daily = daily.merge(daily_temp_mean, on="Date", how="left").merge(daily_dew_mean, on="Date", how="left")
        daily["LCL_m"] = daily.apply(lambda r: lcl_height_m(r["T_mean"], r["Td_mean"]), axis=1)
        df["PrecType_step"] = df.apply(lambda r: detect_precip_type(r.get("RRR [mm/3h]", np.nan), r.get("T2M [°C]", np.nan), r.get("T850 [°C]", np.nan)), axis=1)
        prec_mode = df.groupby("Date")["PrecType_step"].agg(lambda x: x.mode().iat[0] if not x.mode().empty else "Brak").reset_index(name="PrecType")
        daily = daily.merge(prec_mode, on="Date", how="left")
        daily["StormRisk"] = daily.apply(lambda r: storm_risk_category(r["CAPE_max"], r["LIFTED_min"]), axis=1)
        daily["Date_str"] = daily["Date"].astype(str)
    else:
        daily = pd.DataFrame()

    df["LCL_m"] = df.apply(lambda r: lcl_height_m(r.get("T2M [°C]", np.nan), r.get("D2M [°C]", np.nan)) if not np.isnan(r.get("T2M [°C]", np.nan)) else np.nan, axis=1)
    df["PrecType"] = df.apply(lambda r: detect_precip_type(r.get("RRR [mm/3h]", np.nan), r.get("T2M [°C]", np.nan), r.get("T850 [°C]", np.nan)), axis=1)
    df["StormRisk"] = df.apply(lambda r: storm_risk_category(r.get("CAPE [J/kg]", np.nan), r.get("LIFTED [°C]", np.nan)), axis=1)

    return df, daily

# -----------------------
# SAVE to Excel + METEOROGRAM + DAILY PNG (re-using earlier code blocks)
# -----------------------
def save_outputs(df, daily):
    # Excel
    xlsx_path = os.path.join(OUTPUT_DIR, f"krosno_gfs_{RUN_DATE}_{RUN_HOUR}z.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="prognoza", index=False)
        daily.to_excel(writer, sheet_name="dzienna_prognoza", index=False)
        workbook = writer.book
        worksheet = writer.sheets["prognoza"]

        # minimal formatting (re-using many rules from Twoja wersja)
        border_fmt = workbook.add_format({'border': 1})
        if not df.empty:
            worksheet.conditional_format(f'A1:{chr(65 + len(df.columns) - 1)}{len(df) + 1}', {'type': 'no_blanks', 'format': border_fmt})

        # PrecType coloring
        if "PrecType" in df.columns:
            col_idx = df.columns.get_loc("PrecType")
            rng = f"{chr(65+col_idx)}2:{chr(65+col_idx)}{len(df)+1}"
            fmt_rain = workbook.add_format({'bg_color': PREC_TYPE_TO_COLOR.get("Deszcz", "#90EE90"), 'border': 1})
            fmt_snow = workbook.add_format({'bg_color': PREC_TYPE_TO_COLOR.get("Śnieg", "#ADD8E6"), 'border': 1})
            fmt_mix = workbook.add_format({'bg_color': PREC_TYPE_TO_COLOR.get("Deszcz ze śniegiem", "#00FFBB"), 'border': 1})
            fmt_freeze = workbook.add_format({'bg_color': PREC_TYPE_TO_COLOR.get("Deszcz marznący", "#FFA500"), 'border': 1})
            worksheet.conditional_format(rng, {'type': 'cell', 'criteria': 'equal to', 'value': '"Deszcz"', 'format': fmt_rain})
            worksheet.conditional_format(rng, {'type': 'cell', 'criteria': 'equal to', 'value': '"Śnieg"', 'format': fmt_snow})
            worksheet.conditional_format(rng, {'type': 'cell', 'criteria': 'equal to', 'value': '"Deszcz ze śniegiem"', 'format': fmt_mix})
            worksheet.conditional_format(rng, {'type': 'cell', 'criteria': 'equal to', 'value': '"Deszcz marznący"', 'format': fmt_freeze})

        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max() if len(df)>0 else 0, len(col)) + 2
            worksheet.set_column(i, i, max_len)

    print("\n✅ Excel zapisany:", xlsx_path)

        # Meteorogram 120h (if any data)
    df_plot = df[df["T+ (h)"] <= 120].copy() if not df.empty else pd.DataFrame()
    out_png = os.path.join(OUTPUT_DIR, f"meteorogram_krosno_120h.png")
    if not df_plot.empty:
        fig, axes = plt.subplots(7, 1, figsize=(13, 15), sharex=True)
        fig.subplots_adjust(hspace=0.3)
        time_axis = df_plot["Czas"]

        # 1️⃣ Temperatura i punkt rosy
        axes[0].plot(time_axis, df_plot["T2M [°C]"], color="#D62728", label="Temperatura")
        axes[0].plot(time_axis, df_plot["D2M [°C]"], color="#1F77B4", linestyle="--", label="Punkt rosy")
        axes[0].set_ylabel("°C")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[0].grid(True, ls=":")

        # 2️⃣ Opad słupkowo + suma
        axes[1].bar(time_axis, df_plot["RRR [mm/3h]"].fillna(0), width=0.08, color="#1F77B4", label="Opad [mm/3h]")
        axes[1].plot(time_axis, df_plot["RRR [mm/3h]"].fillna(0).cumsum(), color="#000080", linewidth=1, label="Suma opadów")
        axes[1].set_ylabel("mm/3h")
        axes[1].legend(loc="upper left", fontsize=8)
        axes[1].grid(True, ls=":")

        # 3️⃣ Ciśnienie
        axes[2].plot(time_axis, df_plot["MSLP [hPa]"], color="#000000")
        axes[2].set_ylabel("hPa")
        axes[2].grid(True, ls=":")

        # 4️⃣ Wiatr i porywy
        axes[3].plot(time_axis, df_plot["WSPD [m/s]"], color="#FF7F0E", label="Wiatr")
        axes[3].plot(time_axis, df_plot["GUST [m/s]"], color="#D62728", linestyle="--", label="Porywy")
        axes[3].set_ylabel("m/s")
        axes[3].legend(loc="upper left", fontsize=8)
        axes[3].grid(True, ls=":")

        # 5️⃣ Zachmurzenie (stacked)
        low = df_plot["CL [%]"].fillna(0)
        mid = df_plot["CM [%]"].fillna(0)
        high = df_plot["CH [%]"].fillna(0)
        axes[4].fill_between(time_axis, 0, low, color="#b0c4de", label="Niskie")
        axes[4].fill_between(time_axis, low, low + mid, color="#778899", label="Średnie")
        axes[4].fill_between(time_axis, low + mid, low + mid + high, color="#2f4f4f", label="Wysokie")
        axes[4].set_ylabel("Chmury [%]")
        axes[4].set_ylim(0, 100)
        axes[4].legend(loc="upper left", fontsize=8)
        axes[4].grid(True, ls=":")

        # 6️⃣ CAPE + Lifted Index
        axes[5].plot(time_axis, df_plot["CAPE [J/kg]"].fillna(0), color="#8A2BE2", label="CAPE")
        axes[5].plot(time_axis, df_plot["LIFTED [°C]"], color="#2ca02c", linestyle="--", label="Lifted")
        axes[5].set_ylabel("CAPE / LI")
        axes[5].legend(loc="upper left", fontsize=8)
        axes[5].grid(True, ls=":")

        # 7️⃣ Śnieg + widzialność
        axes[6].bar(time_axis, df_plot["SNOW [cm]"].fillna(0), width=0.08, color="#87CEFA", label="Śnieg [cm]")
        ax7b = axes[6].twinx()
        ax7b.plot(time_axis, df_plot["VIS [km]"].fillna(np.nan), color="#8B4513", linewidth=1, label="Widzialność [km]")
        axes[6].set_ylabel("cm")
        ax7b.set_ylabel("km")
        axes[6].legend(loc="upper left", fontsize=8)
        ax7b.legend(loc="upper right", fontsize=8)
        axes[6].grid(True, ls=":")

        date_fmt = DateFormatter("%d.%m\n%H UTC")
        axes[-1].xaxis.set_major_formatter(date_fmt)
        plt.suptitle(f"GFS Krosno – Meteorogram 120h ({RUN_DATE}{RUN_HOUR}Z)", fontsize=14, weight="bold")
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print("✅ Meteorogram zapisany:", out_png)
    else:
        print("⚠️ Brak danych do meteorogramu.")

    # -----------------------
    # DAILY PNG SUMMARY (tabela z ikonami)
    # -----------------------
    if not daily.empty:
        display_df = daily[["Date_str", "Tmax", "Tmin", "Suma_opadu", "PrecType",
                            "VIS_min", "StormRisk", "LCL_m", "Wsp_sred", "Pres_sred"]].copy()
        display_df.columns = ["Data", "Tmax", "Tmin", "Suma_opad", "Typ_opadu",
                              "Vis_min_km", "Ryzyko_burzy", "LCL_m", "W_sred", "P_sred"]

        fig2, ax2 = plt.subplots(figsize=(12, max(2, 0.7 * len(display_df) + 1)))
        ax2.axis('off')
        ax2.set_title(f"Prognoza dzienna - Krosno (pierwsze {len(display_df)} dni) - GFS {RUN_DATE}{RUN_HOUR}Z",
                      fontsize=12, weight="bold")

        cell_text = []
        cell_colors = []
        for _, row in display_df.iterrows():
            prec = row["Typ_opadu"]
            if prec == "Deszcz":
                icon = "🌧️"
                bg = PREC_TYPE_TO_COLOR.get("Deszcz", "#90EE90")
            elif prec == "Śnieg":
                icon = "❄️"
                bg = PREC_TYPE_TO_COLOR.get("Śnieg", "#ADD8E6")
            elif prec == "Deszcz marznący":
                icon = "🧊"
                bg = PREC_TYPE_TO_COLOR.get("Deszcz marznący", "#FFA500")
            else:
                icon = ""
                bg = "#FFFFFF"

            risk = row["Ryzyko_burzy"]
            risk_map = {"Brak": "#FFFFFF", "Niskie": "#FFFF99", "Średnie": "#FFD700",
                        "Wysokie": "#FF8C00", "Ekstremalne": "#FF4500"}
            risk_color = risk_map.get(risk, "#FFFFFF")

            text_row = [
                row["Data"],
                f"{row['Tmax']:.1f}°C" if not np.isnan(row['Tmax']) else "-",
                f"{row['Tmin']:.1f}°C" if not np.isnan(row['Tmin']) else "-",
                f"{row['Suma_opad']:.1f} mm" if not np.isnan(row['Suma_opad']) else "-",
                f"{icon} {prec}",
                f"{row['Vis_min_km']:.1f} km" if not np.isnan(row['Vis_min_km']) else "-",
                risk,
                f"{row['LCL_m']:.0f} m" if not np.isnan(row['LCL_m']) else "-",
                f"{row['W_sred']:.1f} m/s" if not np.isnan(row['W_sred']) else "-",
                f"{row['P_sred']:.1f} hPa" if not np.isnan(row['P_sred']) else "-"
            ]
            cell_text.append(text_row)
            row_colors = ["#FFFFFF"] * len(text_row)
            row_colors[4] = bg
            row_colors[6] = risk_color
            cell_colors.append(row_colors)

        cols = ["Data", "Tmax", "Tmin", "Opad", "Typ opadu", "Widzialność",
                "Ryzyko burzy", "LCL", "Wiatr śr.", "Ciśnienie śr."]
        table = ax2.table(cellText=cell_text, colLabels=cols,
                          cellColours=cell_colors, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)

        out_daily = os.path.join(OUTPUT_DIR, "daily_summary.png")
        plt.savefig(out_daily, dpi=220, bbox_inches="tight")
        plt.close(fig2)
        print("✅ Daily summary PNG zapisany:", out_daily)
    else:
        print("⚠️ Brak danych dziennych do tabeli PNG.")

    return [xlsx_path, out_png, os.path.join(OUTPUT_DIR, "daily_summary.png")]


# -----------------------
# FTP UPLOAD
# -----------------------
def upload_to_ftp(files_to_send):
    """Wysyła pliki przez FTP, dane logowania z .env"""
    load_dotenv()
    host = os.getenv("FTP_HOST")
    user = os.getenv("FTP_USER")
    passwd = os.getenv("FTP_PASS")
    if not host or not user or not passwd:
        print("⚠️ Brak danych FTP w pliku .env – pomijam wysyłkę.")
        return
    try:
        ftp = FTP(host, user, passwd, timeout=30)
        ftp.cwd("/")  # główny katalog
        for path in files_to_send:
            if not os.path.exists(path):
                continue
            fname = os.path.basename(path)
            with open(path, "rb") as f:
                ftp.storbinary(f"STOR {fname}", f)
                print(f"📤 Wysłano na FTP: {fname}")
        ftp.quit()
        print("✅ Wszystkie pliki wysłane na FTP.")
    except error_perm as e:
        print("❌ Błąd FTP (uprawnienia):", e)
    except Exception as e:
        print("❌ Błąd podczas wysyłania na FTP:", e)


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    print(f"\n=== Start GFS Krosno {RUN_DATE}{RUN_HOUR}Z ===")
    downloaded, missing = download_missing_gribs(FORECAST_HOURS)
    df, daily = process_local_gribs(FORECAST_HOURS)
    files = save_outputs(df, daily)
    upload_to_ftp(files)
    print("\n🏁 Gotowe.\n")

