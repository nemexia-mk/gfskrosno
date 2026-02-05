import requests
import time
import pandas as pd
import os
from datetime import datetime, timedelta, timezone

# ────────────────────────────────────────────────
# KONFIGURACJA
# ────────────────────────────────────────────────

APP_KEY = "CCC4084F1EA68A5D31DF36CD567E9C40"
API_KEY = "9ec822a2-8fd2-44dc-902c-2d574bd8850f"

STATIONS = {
    "Turaszowka": "E8:DB:84:99:BF:2B",
    "Lesniowka":  "E8:68:E7:12:8B:9B"
}

OUTPUT_FILES = {
    "Turaszowka": "ecowitt_history_Turaszowka_2026-01-01_to_2026-02-04.csv",
    "Lesniowka": "ecowitt_history_Lesniowka_2026-01-01_to_2026-02-04.csv"
}

# Wszystkie parametry z przykładu real_time
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

# Konwersje z obsługą '-' i None
def safe_float(v):
    if v is None or v == '-':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def safe_int(v):
    if v is None or v == '-':
        return None
    try:
        return int(v)
    except ValueError:
        return None

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
# FUNKCJA POBIERANIA
# ────────────────────────────────────────────────

def fetch_history(mac, station_name, start_dt, end_dt):
    params = {
        "application_key": APP_KEY,
        "api_key": API_KEY,
        "mac": mac,
        "start_date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date":   end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "call_back": FIELDS
    }

    url = "https://api.ecowitt.net/api/v3/device/history"

    print(f"[{station_name}] {params['start_date']} → {params['end_date']}")

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if data.get("code") != 0:
            print(f"  Błąd API: {data.get('msg')}")
            return None

        hist = data.get("data", {})
        if not hist:
            print("  Brak danych w odpowiedzi")
            return None

        # Parsowanie – struktura dict {str_ts: str_value}
        rows = []

        # Bierzemy timestampy z jednego pola (np. outdoor.temperature.list.keys())
        temp_data = hist.get("outdoor", {}).get("temperature", {}).get("list", {})
        timestamps = sorted([int(ts) for ts in temp_data.keys() if ts.isdigit()])

        if not timestamps:
            print("  Brak danych w listach – pusty zakres")
            return None

        for ts in timestamps:
            str_ts = str(ts)
            row = {
                "timestamp_utc": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "station": station_name,
            }

            def get_val(group, sensor):
                return hist.get(group, {}).get(sensor, {}).get("list", {}).get(str_ts)

            # outdoor
            row["out_temp_c"]       = f_to_c(get_val("outdoor", "temperature"))
            row["out_hum_pct"]      = safe_float(get_val("outdoor", "humidity"))
            row["feels_like_c"]     = f_to_c(get_val("outdoor", "feels_like"))
            row["app_temp_c"]       = f_to_c(get_val("outdoor", "app_temp"))
            row["dew_point_c"]      = f_to_c(get_val("outdoor", "dew_point"))
            row["vpd_inHg"]         = safe_float(get_val("outdoor", "vpd"))

            # indoor
            row["indoor_temp_c"]    = f_to_c(get_val("indoor", "temperature"))
            row["indoor_hum_pct"]   = safe_float(get_val("indoor", "humidity"))
            row["indoor_dew_c"]     = f_to_c(get_val("indoor", "dew_point"))
            row["indoor_feels_c"]   = f_to_c(get_val("indoor", "feels_like"))
            row["indoor_app_temp_c"] = f_to_c(get_val("indoor", "app_tempin"))

            # solar_and_uvi
            row["solar_wm2"]        = safe_float(get_val("solar_and_uvi", "solar"))
            row["uvi"]              = safe_int(get_val("solar_and_uvi", "uvi"))

            # rainfall
            row["rain_rate_mmh"]    = in_to_mm(get_val("rainfall", "rain_rate"))
            row["rain_daily_mm"]    = in_to_mm(get_val("rainfall", "daily"))
            row["rain_event_mm"]    = in_to_mm(get_val("rainfall", "event"))
            row["rain_1h_mm"]       = in_to_mm(get_val("rainfall", "1_hour"))
            row["rain_24h_mm"]      = in_to_mm(get_val("rainfall", "24_hours"))
            row["rain_weekly_mm"]   = in_to_mm(get_val("rainfall", "weekly"))
            row["rain_monthly_mm"]  = in_to_mm(get_val("rainfall", "monthly"))
            row["rain_yearly_mm"]   = in_to_mm(get_val("rainfall", "yearly"))

            # wind
            row["wind_kmh"]         = mph_to_kmh(get_val("wind", "wind_speed"))
            row["gust_kmh"]         = mph_to_kmh(get_val("wind", "wind_gust"))
            row["wind_dir_deg"]     = safe_float(get_val("wind", "wind_direction"))
            row["wind_10min_avg_dir"] = safe_float(get_val("wind", "10_minute_average_wind_direction"))

            # pressure
            row["pressure_rel_hpa"] = inHg_to_hPa(get_val("pressure", "relative"))
            row["pressure_abs_hpa"] = inHg_to_hPa(get_val("pressure", "absolute"))

            # temp_and_humidity_ch4
            row["ch4_temp_c"]       = f_to_c(get_val("temp_and_humidity_ch4", "temperature"))
            row["ch4_hum_pct"]      = safe_float(get_val("temp_and_humidity_ch4", "humidity"))

            # temp_ch1/2/3
            row["ch1_temp_c"]       = f_to_c(get_val("temp_ch1", "temperature"))
            row["ch2_temp_c"]       = f_to_c(get_val("temp_ch2", "temperature"))
            row["ch3_temp_c"]       = f_to_c(get_val("temp_ch3", "temperature"))

            # ch_lds1
            row["lds_air_ft"]       = safe_float(get_val("ch_lds1", "air_ch1"))
            row["lds_depth_ft"]     = safe_float(get_val("ch_lds1", "depth_ch1"))
            row["lds_heat"]         = safe_int(get_val("ch_lds1", "ldsheat_ch1"))

            # battery
            row["batt_out_trh"]     = safe_int(get_val("battery", "outdoor_t_rh_sensor"))
            row["batt_wind_v"]      = safe_float(get_val("battery", "wind_sensor"))
            row["batt_rain_v"]      = safe_float(get_val("battery", "rainfall_sensor"))
            row["batt_ch4_th"]      = safe_int(get_val("battery", "temp_humidity_sensor_ch4"))
            row["batt_ch1_temp_v"]  = safe_float(get_val("battery", "temperature_sensor_ch1"))
            row["batt_ch2_temp_v"]  = safe_float(get_val("battery", "temperature_sensor_ch2"))
            row["batt_ch3_temp_v"]  = safe_float(get_val("battery", "temperature_sensor_ch3"))
            row["batt_lds_v"]       = safe_float(get_val("battery", "ldsbatt_1"))

            rows.append(row)

        if rows:
            print(f"  → pobrano {len(rows)} rekordów")
            return pd.DataFrame(rows)
        else:
            print("  Brak wierszy danych")
            return None

    except Exception as e:
        print(f"  Błąd: {e}")
        return None


# ────────────────────────────────────────────────
# GŁÓWNA LOGIKA
# ────────────────────────────────────────────────

START_DATE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_DATE   = datetime(2026, 2, 4, 23, 59, 59, tzinfo=timezone.utc)

DAYS_PER_REQUEST = 1

all_data = {name: [] for name in STATIONS}

current_start = START_DATE
while current_start < END_DATE:
    current_end = min(current_start + timedelta(days=DAYS_PER_REQUEST), END_DATE)

    for station_name, mac in STATIONS.items():
        df_chunk = fetch_history(mac, station_name, current_start, current_end)
        if df_chunk is not None and not df_chunk.empty:
            all_data[station_name].append(df_chunk)

    current_start = current_end + timedelta(seconds=1)
    time.sleep(5)

# ────────────────────────────────────────────────
# ZAPIS DO OSOBNYCH PLIKÓW
# ────────────────────────────────────────────────

for station_name, chunks in all_data.items():
    if chunks:
        final_df = pd.concat(chunks, ignore_index=True)
        final_df.sort_values("timestamp_utc", inplace=True)

        csv_file = OUTPUT_FILES[station_name]

        if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
            final_df.to_csv(csv_file, mode='a', header=False, index=False)
            print(f"\nDołączono {len(final_df)} rekordów do {csv_file}")
        else:
            final_df.to_csv(csv_file, index=False)
            print(f"\nZapisano {len(final_df)} rekordów do {csv_file}")

        print(f"Liczba wierszy dla {station_name}: {len(final_df)}")
    else:
        print(f"\nBrak danych dla {station_name}")

print("\nKoniec.")