#!/usr/bin/env python3
# ecmwf_krosno_smart.py
# Wersja: Smart Run Detection + Natywne Tmax/Tmin
# Zmieniona kolejność kolumn: T2M -> Tmax -> Tmin -> D2M...

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time as dt_time
from matplotlib.dates import DateFormatter
from dotenv import load_dotenv
from ftplib import FTP, error_perm

# -----------------------
# CONFIG
# -----------------------
OUTPUT_DIR = "ecmwf_krosno_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KROSNO_LAT = 49.69
KROSNO_LON = 21.77

# -----------------------
# LOGIKA WYBORU RUNU (06:30 / 18:30 UTC)
# -----------------------
def get_run_info():
    now_utc = datetime.utcnow()
    current_time = now_utc.time()
    
    cutoff_morning = dt_time(6, 30)
    cutoff_evening = dt_time(18, 30)
    
    if current_time < cutoff_morning:
        run_date = (now_utc - timedelta(days=1)).strftime("%Y%m%d")
        run_hour = "12"
    elif current_time < cutoff_evening:
        run_date = now_utc.strftime("%Y%m%d")
        run_hour = "00"
    else:
        run_date = now_utc.strftime("%Y%m%d")
        run_hour = "12"
        
    return run_date, run_hour, now_utc

RUN_DATE_STR, RUN_HOUR_STR, NOW_UTC = get_run_info()
RUN_LABEL = f"{RUN_DATE_STR}_{RUN_HOUR_STR}"

print(f"🕒 Czas UTC: {NOW_UTC.strftime('%H:%M')}")
print(f"🎯 Zidentyfikowany RUN modelu: {RUN_LABEL}Z")

# -----------------------
# OPEN-METEO API SETUP
# -----------------------
URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m", "dew_point_2m", "pressure_msl", 
    "precipitation", "snowfall", "weather_code",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "cape", "lifted_index", "visibility", "temperature_850hPa"
]

# Dodano parametry dzienne
DAILY_VARS = ["temperature_2m_max", "temperature_2m_min"]

PARAMS = {
    "latitude": KROSNO_LAT,
    "longitude": KROSNO_LON,
    "hourly": ",".join(HOURLY_VARS),
    "daily": ",".join(DAILY_VARS),  # Pobieranie natywnych max/min
    "models": "ecmwf_ifs025",
    "timezone": "UTC",
    "wind_speed_unit": "ms",
    "forecast_days": 10
}

COLUMN_MAPPING = {
    "time": "Czas",
    "temperature_2m": "T2M [°C]",
    "dew_point_2m": "D2M [°C]",
    "temperature_850hPa": "T850 [°C]",
    "pressure_msl": "MSLP [hPa]",
    "cloud_cover_low": "CL [%]",
    "cloud_cover_mid": "CM [%]",
    "cloud_cover_high": "CH [%]",
    "cloud_cover": "CC [%]",
    "precipitation": "RRR [mm]",
    "snowfall": "SNOW [cm]",
    "wind_speed_10m": "WSPD [m/s]",
    "wind_direction_10m": "WDIR [°]",
    "wind_gusts_10m": "GUST [m/s]",
    "cape": "CAPE [J/kg]",
    "lifted_index": "LIFTED [°C]",
    "visibility": "VIS [m]"
}

# -----------------------
# HELPERS
# -----------------------
def lcl_height_m(t_c, td_c):
    if pd.isna(t_c) or pd.isna(td_c): return np.nan
    diff = t_c - td_c
    return float(np.round(125.0 * (diff if diff > 0 else 0), 1))

def detect_precip_type(prate, t2m_c, t850_c):
    if pd.isna(prate) or prate <= 0: return "Brak"
    if pd.isna(t2m_c): return "Brak"
    if t2m_c <= 0: return "Śnieg"
    elif 0 < t2m_c < 2:
        if not pd.isna(t850_c) and t850_c < -2: return "Śnieg"
        return "Deszcz ze śniegiem"
    elif t2m_c < 0 and (not pd.isna(t850_c) and t850_c > 0): return "Deszcz marznący"
    else: return "Deszcz"

def storm_risk_category(cape, li):
    if pd.isna(cape) and pd.isna(li): return "Brak"
    cape_val = 0.0 if pd.isna(cape) else cape
    if cape_val < 100: cat = "Niskie"
    elif cape_val <= 400: cat = "Niskie"
    elif cape_val <= 1000: cat = "Średnie"
    elif cape_val <= 2000: cat = "Wysokie"
    else: cat = "Ekstremalne"
    if not pd.isna(li) and li <= -4 and cat == "Niskie": cat = "Średnie"
    return cat

PREC_TYPE_TO_COLOR = {
    "Deszcz": "#0FB00F", "Śnieg": "#ADD8E6",
    "Deszcz ze śniegiem": "#00FFBB", "Deszcz marznący": "#FFA500",
}

# -----------------------
# DATA FETCHING
# -----------------------
def fetch_ecmwf_data():
    print(f"📡 Pobieranie danych ECMWF dla Runu: {RUN_LABEL}Z ...")
    try:
        r = requests.get(URL, params=PARAMS, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        # 1. Dane godzinowe
        hourly_data = data.get('hourly', {})
        if not hourly_data:
            print("❌ API zwróciło pusty obiekt 'hourly'.")
            return pd.DataFrame()
        
        df = pd.DataFrame(hourly_data)
        
        # 2. Dane dzienne (Tmax, Tmin)
        daily_data = data.get('daily', {})
        if daily_data:
            df_daily = pd.DataFrame(daily_data)
            # Konwersja czasu na samą datę dla złączenia
            df_daily["time"] = pd.to_datetime(df_daily["time"]).dt.date
            df_daily.rename(columns={
                "time": "Date",
                "temperature_2m_max": "Tmax [°C]",
                "temperature_2m_min": "Tmin [°C]"
            }, inplace=True)
        else:
            df_daily = pd.DataFrame()

        # Konwersja danych godzinowych
        for col in df.columns:
            if col == "time": continue
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.rename(columns=COLUMN_MAPPING, inplace=True)
        df["Czas"] = pd.to_datetime(df["Czas"])
        
        # Dodanie kolumny Date do głównego DF, aby połączyć z daily
        df["Date"] = df["Czas"].dt.date
        
        # 3. Złączenie (Merge) - dodanie Tmax i Tmin do każdego wiersza
        if not df_daily.empty:
            df = df.merge(df_daily[["Date", "Tmax [°C]", "Tmin [°C]"]], on="Date", how="left")
        else:
            df["Tmax [°C]"] = np.nan
            df["Tmin [°C]"] = np.nan

        # Oblicz T+
        first_time = df["Czas"].iloc[0]
        df["T+ (h)"] = ((df["Czas"] - first_time).dt.total_seconds() / 3600).astype(int)

        if "VIS [m]" in df.columns:
            df["VIS [km]"] = (df["VIS [m]"] / 1000).round(1)
            df.drop(columns=["VIS [m]"], inplace=True)
        
        # Zaokrąglenia
        cols_round_1 = ["T2M [°C]", "D2M [°C]", "T850 [°C]", "MSLP [hPa]", "RRR [mm]", "GUST [m/s]", "WSPD [m/s]", "Tmax [°C]", "Tmin [°C]"]
        for c in cols_round_1:
            if c in df.columns: df[c] = df[c].round(1)
        
        cols_round_0 = ["CC [%]", "CL [%]", "CM [%]", "CH [%]", "CAPE [J/kg]"]
        for c in cols_round_0:
            if c in df.columns: df[c] = df[c].round(0)

        df["LCL_m"] = df.apply(lambda r: lcl_height_m(r.get("T2M [°C]"), r.get("D2M [°C]")), axis=1)
        df["PrecType"] = df.apply(lambda r: detect_precip_type(r.get("RRR [mm]"), r.get("T2M [°C]"), r.get("T850 [°C]")), axis=1)
        df["StormRisk"] = df.apply(lambda r: storm_risk_category(r.get("CAPE [J/kg]"), r.get("LIFTED [°C]")), axis=1)

        # 4. Ustalenie kolejności kolumn
        # Lista pożądana: T2M, Tmax, Tmin, D2M, T850...
        desired_order = [
            "Czas", "T+ (h)", 
            "T2M [°C]", "Tmax [°C]", "Tmin [°C]", "D2M [°C]", "T850 [°C]",
            "MSLP [hPa]",
            "RRR [mm]", "SNOW [cm]", "PrecType",
            "WSPD [m/s]", "GUST [m/s]", "WDIR [°]",
            "CC [%]", "CL [%]", "CM [%]", "CH [%]",
            "CAPE [J/kg]", "LIFTED [°C]", "StormRisk",
            "VIS [km]", "LCL_m", "Date"
        ]
        
        # Filtrujemy tylko te kolumny, które faktycznie istnieją w DF
        final_cols = [c for c in desired_order if c in df.columns]
        
        # Dodajemy ewentualne pozostałe, których nie ma w liście desired_order (dla bezpieczeństwa)
        remaining = [c for c in df.columns if c not in final_cols]
        
        df = df[final_cols + remaining]

        return df

    except Exception as e:
        print(f"❌ Błąd pobierania: {e}")
        return pd.DataFrame()

def process_daily(df):
    if df.empty: return pd.DataFrame()
    
    # Grupowanie
    daily = df.groupby("Date").agg({
        "T2M [°C]": ["mean"],       # Średnia z godzinowych
        "Tmax [°C]": ["first"],     # Bierzemy natywną wartość (jest taka sama dla całego dnia w wierszach)
        "Tmin [°C]": ["first"],     # Bierzemy natywną wartość
        "RRR [mm]": "sum",
        "WSPD [m/s]": "mean",
        "MSLP [hPa]": "mean",
        "CAPE [J/kg]": "max",
        "VIS [km]": "min"
    }).reset_index()
    
    # Spłaszczenie MultiIndex
    daily.columns = ["Date", "T_mean", "Tmax", "Tmin", "Suma_opadu", "Wsp_sred", "Pres_sred", "CAPE_max", "VIS_min"]
    
    if "LIFTED [°C]" in df.columns:
        daily["LIFTED_min"] = df.groupby("Date")["LIFTED [°C]"].min().values
    else: daily["LIFTED_min"] = np.nan

    td_mean = df.groupby("Date")["D2M [°C]"].mean().reset_index(name="Td_mean")
    daily = daily.merge(td_mean, on="Date", how="left")
    
    daily["LCL_m"] = daily.apply(lambda r: lcl_height_m(r["T_mean"], r["Td_mean"]), axis=1)
    daily["StormRisk"] = daily.apply(lambda r: storm_risk_category(r["CAPE_max"], r["LIFTED_min"]), axis=1)
    
    prec_mode = df.groupby("Date")["PrecType"].agg(lambda x: x.mode().iat[0] if not x.mode().empty else "Brak").reset_index(name="PrecType")
    daily = daily.merge(prec_mode, on="Date", how="left")
    daily["Date_str"] = daily["Date"].astype(str)
    
    for c in ["Tmax", "Tmin", "Suma_opadu", "Wsp_sred", "Pres_sred", "T_mean", "Td_mean", "LCL_m"]:
        if c in daily.columns: daily[c] = daily[c].round(1)
        
    return daily

# -----------------------
# SAVE OUTPUTS (Używa RUN_LABEL)
# -----------------------
def save_outputs(df, daily):
    if df.empty:
        print("⚠️ Pusty DataFrame, pomijam zapis.")
        return []

    # Usuwamy kolumnę techniczną 'Date' z głównego pliku, żeby nie śmieciła,
    # ale robimy to na kopii do zapisu, bo 'Date' jest potrzebna w process_daily (choć tutaj już po procesowaniu)
    df_save = df.drop(columns=["Date"], errors='ignore')

    filename_base = f"krosno_ecmwf_{RUN_LABEL}"
    xlsx_path = os.path.join(OUTPUT_DIR, f"{filename_base}.xlsx")
    
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df_save.to_excel(writer, sheet_name="prognoza", index=False)
        daily.to_excel(writer, sheet_name="dzienna_prognoza", index=False)
        workbook = writer.book
        worksheet = writer.sheets["prognoza"]
        
        if "PrecType" in df_save.columns:
            col_idx = df_save.columns.get_loc("PrecType")
            rng = f"{chr(65+col_idx)}2:{chr(65+col_idx)}{len(df_save)+1}"
            for ptype, color in PREC_TYPE_TO_COLOR.items():
                fmt = workbook.add_format({'bg_color': color, 'border': 1})
                worksheet.conditional_format(rng, {'type': 'cell', 'criteria': 'equal to', 'value': f'"{ptype}"', 'format': fmt})
        
        for i, col in enumerate(df_save.columns):
            worksheet.set_column(i, i, 12)

    print(f"\n✅ Excel ECMWF zapisany: {xlsx_path}")

    csv_path = os.path.join(OUTPUT_DIR, f"{filename_base}.csv")
    df_save.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ CSV ECMWF zapisany: {csv_path}")

    # Meteorogram
    df_plot = df[df["T+ (h)"] <= 120].copy()
    out_png = os.path.join(OUTPUT_DIR, f"meteorogram_ecmwf_krosno_120h.png")
    
    if not df_plot.empty:
        fig, axes = plt.subplots(7, 1, figsize=(13, 15), sharex=True)
        fig.subplots_adjust(hspace=0.3)
        time_axis = df_plot["Czas"]
        
        axes[0].plot(time_axis, df_plot["T2M [°C]"], color="#D62728", label="T2M")
        axes[0].plot(time_axis, df_plot["D2M [°C]"], color="#1F77B4", ls="--", label="D2M")
        # Można opcjonalnie dorysować Tmax/Tmin jako punkty, ale zostawmy klasyczny wygląd
        axes[0].set_ylabel("°C"); axes[0].legend(loc="upper left"); axes[0].grid(True, ls=":")
        
        axes[1].bar(time_axis, df_plot["RRR [mm]"].fillna(0), width=0.04, color="#1F77B4", label="Opad/h")
        axes[1].set_ylabel("mm"); axes[1].legend(loc="upper left"); axes[1].grid(True, ls=":")
        
        axes[2].plot(time_axis, df_plot["MSLP [hPa]"], color="black")
        axes[2].set_ylabel("hPa"); axes[2].grid(True, ls=":")
        
        axes[3].plot(time_axis, df_plot["WSPD [m/s]"], color="#FF7F0E", label="Wiatr")
        axes[3].plot(time_axis, df_plot["GUST [m/s]"], color="#D62728", ls="--", label="Porywy")
        axes[3].set_ylabel("m/s"); axes[3].legend(loc="upper left"); axes[3].grid(True, ls=":")
        
        axes[4].stackplot(time_axis, df_plot["CL [%]"].fillna(0), df_plot["CM [%]"].fillna(0), df_plot["CH [%]"].fillna(0),
                          labels=['Low','Mid','High'], colors=['#b0c4de','#778899','#2f4f4f'], alpha=0.7)
        axes[4].set_ylim(0, 100); axes[4].set_ylabel("[%]"); axes[4].legend(loc="upper left"); axes[4].grid(True, ls=":")
        
        axes[5].plot(time_axis, df_plot["CAPE [J/kg]"].fillna(0), color="purple", label="CAPE")
        axes[5].set_ylabel("J/kg"); axes[5].legend(loc="upper left"); axes[5].grid(True, ls=":")
        
        axes[6].bar(time_axis, df_plot["SNOW [cm]"].fillna(0), width=0.04, color="cyan", label="Śnieg")
        ax7b = axes[6].twinx()
        ax7b.plot(time_axis, df_plot["VIS [km]"], color="brown", label="Vis")
        axes[6].set_ylabel("cm"); ax7b.set_ylabel("km")
        axes[6].legend(loc="upper left"); axes[6].grid(True, ls=":")

        axes[-1].xaxis.set_major_formatter(DateFormatter("%d.%m\n%H"))
        
        plt.suptitle(f"ECMWF 0.25° Krosno | RUN: {RUN_LABEL}Z", weight="bold", fontsize=14)
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Meteorogram zapisany: {out_png}")
        
        return [xlsx_path, csv_path, out_png]
    return [xlsx_path, csv_path]

# -----------------------
# FTP
# -----------------------
def upload_to_ftp(files):
    load_dotenv()
    host = os.getenv("FTP_HOST")
    user = os.getenv("FTP_USER")
    passwd = os.getenv("FTP_PASS")
    if not all([host, user, passwd]): return

    try:
        ftp = FTP(host, user, passwd, timeout=30)
        ftp.cwd("/stacja.meteo-krosno.pl/")
        for path in files:
            if not os.path.exists(path): continue
            fname = os.path.basename(path)
            with open(path, "rb") as f:
                if path.endswith('.csv'):
                    ftp.storbinary("STOR ecmwf-tab.csv", f)
                    
                    # Archiwizacja
                    arch_dir = "/stacja.meteo-krosno.pl/archiv"
                    try: ftp.cwd(arch_dir)
                    except error_perm: 
                        ftp.mkd(arch_dir); ftp.cwd(arch_dir)
                    
                    f.seek(0)
                    ftp.storbinary(f"STOR {fname}", f)
                    print(f"📤 Archiwum: {fname}")
                    ftp.cwd("/stacja.meteo-krosno.pl/")
                else:
                    ftp.storbinary(f"STOR {fname}", f)
        ftp.quit()
        print("✅ FTP Upload OK.")
    except Exception as e:
        print(f"❌ FTP Błąd: {e}")

if __name__ == "__main__":
    df = fetch_ecmwf_data()
    daily = process_daily(df)
    files = save_outputs(df, daily)
    upload_to_ftp(files)
    print("🏁 Gotowe.")
