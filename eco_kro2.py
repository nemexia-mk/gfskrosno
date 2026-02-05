import requests
import time
import pandas as pd
import os
import ftplib
from datetime import datetime, timedelta, timezone

# ==================== KONFIGURACJA ====================
APP_KEY = "CCC4084F1EA68A5D31DF36CD567E9C40"
API_KEY = "9ec822a2-8fd2-44dc-902c-2d574bd8850f"

STATIONS = {
    "Turaszowka": "E8:DB:84:99:BF:2B",   # → eco_krosno.csv
    "Lesniowka":  "E8:68:E7:12:8B:9B"    # → eco_lesniowka.csv
}

OUTPUT_FILES = {
    "Turaszowka": "eco_krosno.csv",
    "Lesniowka":  "eco_lesniowka.csv"
}

# Wszystkie parametry
FIELDS = (
    "outdoor.temperature,outdoor.feels_like,outdoor.app_temp,outdoor.dew_point,outdoor.vpd,outdoor.humidity,"
    "indoor.temperature,indoor.humidity,indoor.dew_point,indoor.feels_like,indoor.app_tempin,"
    "solar_and_uvi.solar,solar_and_uvi.uvi,"
    "rainfall.rain_rate,rainfall.daily,rainfall.event,rainfall.1_hour,rainfall.24_hours,rainfall.weekly,rainfall.monthly,rainfall.yearly,"
    "wind.wind_speed,wind.wind_gust,wind.wind_direction,wind.10_minute_average_wind_direction,"
    "pressure.relative,pressure.absolute,"
    "temp_and_humidity_ch4.temperature,temp_and_humidity_ch4.humidity,"
    "temp_ch1.temperature,temp_ch2.temperature,temp_ch3.temperature,"
    "ch_lds1.air_ch1,ch_lds1.depth_ch1,ch_lds1.ldsheat_ch1,"
    "battery.outdoor_t_rh_sensor,battery.wind_sensor,battery.rainfall_sensor,"
    "battery.temp_humidity_sensor_ch4,battery.temperature_sensor_ch1,battery.temperature_sensor_ch2,"
    "battery.temperature_sensor_ch3,battery.ldsbatt_1"
)

# Konwersje z obsługą '-'
def safe_float(v):
    if v is None or v == '-': return None
    try: return float(v)
    except: return None

def safe_int(v):
    if v is None or v == '-': return None
    try: return int(v)
    except: return None

def f_to_c(f): 
    val = safe_float(f)
    return round((val - 32) * 5/9, 1) if val is not None else None

def mph_to_kmh(m): 
    val = safe_float(m)
    return round(val * 1.60934, 1) if val is not None else None

def in_to_mm(i): 
    val = safe_float(i)
    return round(val * 25.4, 1) if val is not None else None

def inHg_to_hPa(p): 
    val = safe_float(p)
    return round(val * 33.8639, 1) if val is not None else None

# ==================== FUNKCJA POBIERANIA ====================
def fetch_history(mac, station_name, start_dt, end_dt):
    params = {
        "application_key": APP_KEY,
        "api_key": API_KEY,
        "mac": mac,
        "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "call_back": FIELDS
    }

    print(f"[{station_name}] {params['start_date']} → {params['end_date']}")

    try:
        r = requests.get("https://api.ecowitt.net/api/v3/device/history", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if data.get("code") != 0:
            print(f"  Błąd API: {data.get('msg')}")
            return None

        hist = data.get("data", {})
        if not hist:
            print("  Brak danych")
            return None

        temp_data = hist.get("outdoor", {}).get("temperature", {}).get("list", {})
        timestamps = sorted([int(ts) for ts in temp_data.keys() if ts.isdigit()])

        if not timestamps:
            print("  Brak rekordów w tym zakresie")
            return None

        rows = []
        for ts in timestamps:
            str_ts = str(ts)
            row = {"timestamp_utc": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(), "station": station_name}

            def g(group, sensor): 
                return hist.get(group, {}).get(sensor, {}).get("list", {}).get(str_ts)

            # outdoor
            row["out_temp_c"] = f_to_c(g("outdoor", "temperature"))
            row["out_hum_pct"] = safe_float(g("outdoor", "humidity"))
            row["feels_like_c"] = f_to_c(g("outdoor", "feels_like"))
            row["app_temp_c"] = f_to_c(g("outdoor", "app_temp"))
            row["dew_point_c"] = f_to_c(g("outdoor", "dew_point"))
            row["vpd_inHg"] = safe_float(g("outdoor", "vpd"))

            # indoor
            row["indoor_temp_c"] = f_to_c(g("indoor", "temperature"))
            row["indoor_hum_pct"] = safe_float(g("indoor", "humidity"))
            row["indoor_dew_c"] = f_to_c(g("indoor", "dew_point"))
            row["indoor_feels_c"] = f_to_c(g("indoor", "feels_like"))
            row["indoor_app_c"] = f_to_c(g("indoor", "app_tempin"))

            # solar, rain, wind, pressure, ch4, temp_ch, lds, battery – wszystkie na końcu
            row["solar_wm2"] = safe_float(g("solar_and_uvi", "solar"))
            row["uvi"] = safe_int(g("solar_and_uvi", "uvi"))
            row["rain_rate_mmh"] = in_to_mm(g("rainfall", "rain_rate"))
            row["rain_daily_mm"] = in_to_mm(g("rainfall", "daily"))
            row["rain_event_mm"] = in_to_mm(g("rainfall", "event"))
            row["rain_1h_mm"] = in_to_mm(g("rainfall", "1_hour"))
            row["rain_24h_mm"] = in_to_mm(g("rainfall", "24_hours"))
            row["rain_weekly_mm"] = in_to_mm(g("rainfall", "weekly"))
            row["rain_monthly_mm"] = in_to_mm(g("rainfall", "monthly"))
            row["rain_yearly_mm"] = in_to_mm(g("rainfall", "yearly"))

            row["wind_kmh"] = mph_to_kmh(g("wind", "wind_speed"))
            row["gust_kmh"] = mph_to_kmh(g("wind", "wind_gust"))
            row["wind_dir_deg"] = safe_float(g("wind", "wind_direction"))
            row["wind_10min_avg"] = safe_float(g("wind", "10_minute_average_wind_direction"))

            row["pressure_rel_hpa"] = inHg_to_hPa(g("pressure", "relative"))
            row["pressure_abs_hpa"] = inHg_to_hPa(g("pressure", "absolute"))

            row["ch4_temp_c"] = f_to_c(g("temp_and_humidity_ch4", "temperature"))
            row["ch4_hum_pct"] = safe_float(g("temp_and_humidity_ch4", "humidity"))

            row["ch1_temp_c"] = f_to_c(g("temp_ch1", "temperature"))
            row["ch2_temp_c"] = f_to_c(g("temp_ch2", "temperature"))
            row["ch3_temp_c"] = f_to_c(g("temp_ch3", "temperature"))

            row["lds_air_ft"] = safe_float(g("ch_lds1", "air_ch1"))
            row["lds_depth_ft"] = safe_float(g("ch_lds1", "depth_ch1"))
            row["lds_heat"] = safe_int(g("ch_lds1", "ldsheat_ch1"))

            row["batt_out_trh"] = safe_int(g("battery", "outdoor_t_rh_sensor"))
            row["batt_wind_v"] = safe_float(g("battery", "wind_sensor"))
            row["batt_rain_v"] = safe_float(g("battery", "rainfall_sensor"))
            row["batt_ch4_th"] = safe_int(g("battery", "temp_humidity_sensor_ch4"))
            row["batt_ch1_v"] = safe_float(g("battery", "temperature_sensor_ch1"))
            row["batt_ch2_v"] = safe_float(g("battery", "temperature_sensor_ch2"))
            row["batt_ch3_v"] = safe_float(g("battery", "temperature_sensor_ch3"))
            row["batt_lds_v"] = safe_float(g("battery", "ldsbatt_1"))

            rows.append(row)

        if rows:
            print(f"  → pobrano {len(rows)} rekordów")
            return pd.DataFrame(rows)
        return None

    except Exception as e:
        print(f"  Błąd: {e}")
        return None

# ==================== UPLOAD FTP ====================
def upload_to_ftp(local_file):
    try:
        host = os.getenv("FTP_HOST")
        user = os.getenv("FTP_USER")
        passwd = os.getenv("FTP_PASS")
        if not all([host, user, passwd]):
            print(f"  Brak danych FTP dla {local_file}")
            return

        ftp = ftplib.FTP(host)
        ftp.login(user, passwd)
        with open(local_file, 'rb') as f:
            ftp.storbinary(f"STOR {os.path.basename(local_file)}", f)
        ftp.quit()
        print(f"  ✓ Wysłano na FTP: {os.path.basename(local_file)}")
    except Exception as e:
        print(f"  FTP error: {e}")

# ==================== GŁÓWNA LOGIKA ====================
# Ostatnia pełna doba (wczoraj 00:00 → dziś 00:00)
utc_now = datetime.now(timezone.utc)
end_dt = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
start_dt = end_dt - timedelta(days=1)

print(f"Pobieram ostatnią pełną dobę: {start_dt.date()} 00:00 → {end_dt.date()} 00:00\n")

for station_name, mac in STATIONS.items():
    csv_file = OUTPUT_FILES[station_name]
    df_chunk = fetch_history(mac, station_name, start_dt, end_dt)
    
    if df_chunk is not None and not df_chunk.empty:
        # Logika append bez mieszania starych rekordów
        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
            df_old = pd.read_csv(csv_file)
            existing_cols = df_old.columns.tolist()
            new_cols = [c for c in df_chunk.columns if c not in existing_cols]
            df_chunk = df_chunk[existing_cols + new_cols]
            df_chunk.to_csv(csv_file, mode='a', header=False, index=False)
            print(f"→ Dołączono {len(df_chunk)} rekordów do {csv_file}")
        else:
            df_chunk.to_csv(csv_file, index=False)
            print(f"→ Utworzono {csv_file} z {len(df_chunk)} rekordami")

        upload_to_ftp(csv_file)
    else:
        print(f"→ Brak nowych danych dla {station_name}")

print("\n=== ZAKOŃCZONO ===")
