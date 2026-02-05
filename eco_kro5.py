import requests
import time
import pandas as pd
import os
import ftplib
from datetime import datetime, timedelta, timezone

# ────────────────────────────────────────────────
# KONFIGURACJA
# ────────────────────────────────────────────────
APP_KEY = "CCC4084F1EA68A5D31DF36CD567E9C40"
API_KEY = "9ec822a2-8fd2-44dc-902c-2d574bd8850f"
STATIONS = {
    "Turaszowka": "E8:DB:84:99:BF:2B", # eco_krosno.csv
    "Lesniowka": "E8:68:E7:12:8B:9B" # eco_lesniowka.csv
}
# Pliki lokalne (append + nowe kolumny)
LOCAL_FILES = {
    "Turaszowka": "eco_krosno.csv",
    "Lesniowka": "eco_lesniowka.csv"
}
# Remote na FTP (nazwy plików na serwerze)
REMOTE_FILES = {
    "Turaszowka": "eco_krosno.csv",
    "Lesniowka": "eco_lesniowka.csv"
}
# Wszystkie parametry + temp_ch1/2/3 na końcu
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
    "battery.outdoor_t_rh_sensor,battery.wind_sensor,battery.rainfall_sensor,battery.temp_humidity_sensor_ch4,battery.temperature_sensor_ch1,battery.temperature_sensor_ch2,battery.temperature_sensor_ch3,battery.ldsbatt_1"
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
    f = safe_float(f)
    return round((f - 32) * 5/9, 1) if f is not None else None
def mph_to_kmh(m):
    m = safe_float(m)
    return round(m * 1.60934, 1) if m is not None else None
def in_to_mm(i):
    i = safe_float(i)
    return round(i * 25.4, 1) if i is not None else None
def inHg_to_hPa(p):
    p = safe_float(p)
    return round(p * 33.8639, 1) if p is not None else None
# ────────────────────────────────────────────────
# FUNKCJA POBIERANIA (ostatnia pełna doba)
# ────────────────────────────────────────────────
def fetch_last_full_day(mac, station_name):
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    start_dt = yesterday
    end_dt = today
    params = {
        "application_key": APP_KEY,
        "api_key": API_KEY,
        "mac": mac,
        "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "call_back": FIELDS
    }
    print(f"[{station_name}] Ostatnia pełna doba: {params['start_date']} → {params['end_date']}")
    try:
        r = requests.get("https://api.ecowitt.net/api/v3/device/history", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != 0:
            print(f" Błąd API: {data.get('msg')}")
            return None
        hist = data.get("data", {})
        if not hist:
            print(" Brak danych")
            return None
        # Timestampy z dict list
        temp_list = hist.get("outdoor", {}).get("temperature", {}).get("list", {})
        timestamps = sorted([int(k) for k in temp_list.keys() if k.isdigit()])
        if not timestamps:
            print(" Brak rekordów")
            return None
        rows = []
        for ts in timestamps:
            str_ts = str(ts)
            row = {"timestamp_utc": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(), "station": station_name}
            def get_val(g, s):
                return hist.get(g, {}).get(s, {}).get("list", {}).get(str_ts)
            # outdoor / indoor / solar / rain / wind / pressure / ch4 / battery (jak wcześniej)
            row["out_temp_c"] = f_to_c(get_val("outdoor", "temperature"))
            row["out_hum_pct"] = safe_float(get_val("outdoor", "humidity"))
            row["feels_like_c"] = f_to_c(get_val("outdoor", "feels_like"))
            row["app_temp_c"] = f_to_c(get_val("outdoor", "app_temp"))
            row["dew_point_c"] = f_to_c(get_val("outdoor", "dew_point"))
            row["vpd"] = safe_float(get_val("outdoor", "vpd"))
            row["indoor_temp_c"] = f_to_c(get_val("indoor", "temperature"))
            row["indoor_hum_pct"] = safe_float(get_val("indoor", "humidity"))
            row["indoor_dew_c"] = f_to_c(get_val("indoor", "dew_point"))
            row["indoor_feels_c"] = f_to_c(get_val("indoor", "feels_like"))
            row["indoor_app_c"] = f_to_c(get_val("indoor", "app_tempin"))
            row["solar_wm2"] = safe_float(get_val("solar_and_uvi", "solar"))
            row["uvi"] = safe_int(get_val("solar_and_uvi", "uvi"))
            row["rain_rate_mmh"] = in_to_mm(get_val("rainfall", "rain_rate"))
            row["rain_daily_mm"] = in_to_mm(get_val("rainfall", "daily"))
            row["rain_event_mm"] = in_to_mm(get_val("rainfall", "event"))
            row["rain_1h_mm"] = in_to_mm(get_val("rainfall", "1_hour"))
            row["rain_24h_mm"] = in_to_mm(get_val("rainfall", "24_hours"))
            row["rain_weekly_mm"] = in_to_mm(get_val("rainfall", "weekly"))
            row["rain_monthly_mm"] = in_to_mm(get_val("rainfall", "monthly"))
            row["rain_yearly_mm"] = in_to_mm(get_val("rainfall", "yearly"))
            row["wind_kmh"] = mph_to_kmh(get_val("wind", "wind_speed"))
            row["gust_kmh"] = mph_to_kmh(get_val("wind", "wind_gust"))
            row["wind_dir"] = safe_float(get_val("wind", "wind_direction"))
            row["wind_10min_dir"] = safe_float(get_val("wind", "10_minute_average_wind_direction"))
            row["pressure_rel_hpa"] = inHg_to_hPa(get_val("pressure", "relative"))
            row["pressure_abs_hpa"] = inHg_to_hPa(get_val("pressure", "absolute"))
            row["ch4_temp_c"] = f_to_c(get_val("temp_and_humidity_ch4", "temperature"))
            row["ch4_hum"] = safe_float(get_val("temp_and_humidity_ch4", "humidity"))
            row["batt_out"] = safe_int(get_val("battery", "outdoor_t_rh_sensor"))
            row["batt_wind"] = safe_float(get_val("battery", "wind_sensor"))
            row["batt_rain"] = safe_float(get_val("battery", "rainfall_sensor"))
            row["batt_ch4"] = safe_int(get_val("battery", "temp_humidity_sensor_ch4"))
            row["batt_ch1"] = safe_float(get_val("battery", "temperature_sensor_ch1"))
            row["batt_ch2"] = safe_float(get_val("battery", "temperature_sensor_ch2"))
            row["batt_ch3"] = safe_float(get_val("battery", "temperature_sensor_ch3"))
            row["batt_lds"] = safe_float(get_val("battery", "ldsbatt_1"))
            # NOWE CZUJNIKI NA KOŃCU (temp_ch1/2/3, ch_lds1)
            row["ch1_temp_c"] = f_to_c(get_val("temp_ch1", "temperature"))
            row["ch2_temp_c"] = f_to_c(get_val("temp_ch2", "temperature"))
            row["ch3_temp_c"] = f_to_c(get_val("temp_ch3", "temperature"))
            row["lds_air_ft"] = safe_float(get_val("ch_lds1", "air_ch1"))
            row["lds_depth_ft"] = safe_float(get_val("ch_lds1", "depth_ch1"))
            row["lds_heat"] = safe_int(get_val("ch_lds1", "ldsheat_ch1"))
            rows.append(row)
        if rows:
            print(f" → {len(rows)} rekordów")
            return pd.DataFrame(rows)
        return None
    except Exception as e:
        print(f" Błąd: {e}")
        return None
# ────────────────────────────────────────────────
# UPLOAD FTP (z tworzeniem folderu jeśli nie istnieje)
# ────────────────────────────────────────────────
def upload_ftp(local_file, remote_file):
    try:
        host = os.getenv("FTP_HOST")
        user = os.getenv("FTP_USER")
        passwd = os.getenv("FTP_PASS")
        if not all([host, user, passwd]):
            print(" Brak FTP env vars – pomijam upload")
            return
        ftp = ftplib.FTP(host)
        ftp.login(user, passwd)
        try:
            ftp.cwd(FTP_DIR)
        except error_perm as e:
            if '550' in str(e):
                # Folder nie istnieje – utwórz
                try:
                    ftp.mkd(FTP_DIR)
                    ftp.cwd(FTP_DIR)
                    print(f" Utworzono folder na FTP: {FTP_DIR}")
                except Exception as mk_e:
                    print(f" Błąd tworzenia folderu: {mk_e}")
                    ftp.quit()
                    return
            else:
                print(f" Błąd cwd: {e}")
                ftp.quit()
                return
        with open(local_file, 'rb') as f:
            ftp.storbinary(f"STOR {remote_file}", f)
        ftp.quit()
        print(f" Upload OK: {remote_file}")
    except Exception as e:
        print(f" FTP błąd: {e}")
# ────────────────────────────────────────────────
# GŁÓWNA LOGIKA
# ────────────────────────────────────────────────
for station_name, mac in STATIONS.items():
    df_new = fetch_last_full_day(mac, station_name)
    if df_new is None or df_new.empty:
        continue
    csv_file = LOCAL_FILES[station_name]
    # Merge z istniejącym CSV (nowe kolumny na końcu)
    if os.path.exists(csv_file):
        try:
            df_old = pd.read_csv(csv_file)
            # Upewnij się, że timestamp_utc jest datetime
            df_old['timestamp_utc'] = pd.to_datetime(df_old['timestamp_utc'])
            df_new['timestamp_utc'] = pd.to_datetime(df_new['timestamp_utc'])
            # Concat i dedup po timestamp + station
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=['timestamp_utc', 'station'], keep='last', inplace=True)
            df_combined.sort_values('timestamp_utc', inplace=True)
        except Exception as e:
            print(f" Błąd merge: {e} – nadpisuję")
            df_combined = df_new
    else:
        df_combined = df_new
    # Zapisz (pełny plik z nowymi kolumnami)
    df_combined.to_csv(csv_file, index=False)
    print(f"Zapisano {len(df_combined)} wierszy do {csv_file}")
    # Upload na FTP
    upload_ftp(csv_file, REMOTE_FILES[station_name])
print("\nKoniec – ostatnia pełna doba pobrana i wysłana.")
