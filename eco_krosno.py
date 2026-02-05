import requests
import pandas as pd
import os
from ftplib import FTP, error_perm
from datetime import datetime, timedelta, timezone
import io

# ────────────────────────────────────────────────
# KONFIGURACJA
# ────────────────────────────────────────────────
APP_KEY = "CCC4084F1EA68A5D31DF36CD567E9C40"
API_KEY = "9ec822a2-8fd2-44dc-902c-2d574bd8850f"

# Konfiguracja FTP
FTP_HOST_CFG = os.getenv("FTP_HOST", "twoj_host_ftp.pl") 
FTP_USER_CFG = os.getenv("FTP_USER", "uzytkownik")
FTP_PASS_CFG = os.getenv("FTP_PASS", "haslo")
FTP_DIR = "stacja.meteo-krosno.pl/"  # Ścieżka na serwerze

STATIONS = {
    "Turaszowka": "E8:DB:84:99:BF:2B", # eco_krosno.csv
    "Lesniowka": "E8:68:E7:12:8B:9B"   # eco_lesniowka.csv
}

# Nazwy plików
FILES_MAP = {
    "Turaszowka": "eco_krosno.csv",
    "Lesniowka": "eco_lesniowka.csv"
}

# Pola API
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

# ────────────────────────────────────────────────
# FUNKCJE POMOCNICZE
# ────────────────────────────────────────────────
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
# 1. POBIERANIE Z FTP
# ────────────────────────────────────────────────
def download_ftp_file(remote_filename, local_filename):
    print(f"--- [FTP DOWNLOAD] Próba pobrania: {remote_filename} ---")
    try:
        host = FTP_HOST_CFG
        if not host or "twoj_host" in host:
            print(" [INFO] Brak konfiguracji FTP. Używam tylko pliku lokalnego.")
            return False

        ftp = FTP(host)
        ftp.login(FTP_USER_CFG, FTP_PASS_CFG)
        try:
            ftp.cwd(FTP_DIR)
        except:
            pass # Próbujemy w głównym

        with open(local_filename, 'wb') as f:
            ftp.retrbinary(f"RETR {remote_filename}", f.write)
        
        ftp.quit()
        print(f" [OK] Pobrano plik z FTP: {local_filename}")
        return True
    except error_perm:
        print(f" [INFO] Plik {remote_filename} nie istnieje na serwerze.")
        return False
    except Exception as e:
        print(f" [BLAD] Błąd FTP: {e}")
        return False

# ────────────────────────────────────────────────
# 2. SPRAWDZANIE DATY
# ────────────────────────────────────────────────
def check_if_date_exists(csv_file, target_date_str):
    if not os.path.exists(csv_file):
        return False
    try:
        print(f" [CHECK] Sprawdzam czy data {target_date_str} istnieje w {csv_file}...")
        df = pd.read_csv(csv_file, usecols=['timestamp_utc'])
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], errors='coerce')
        existing_dates = df['timestamp_utc'].dt.strftime('%Y-%m-%d').unique()
        
        if target_date_str in existing_dates:
            print(f" [INFO] Data {target_date_str} już istnieje! Pomijam.")
            return True
        else:
            print(f" [INFO] Brak danych dla {target_date_str}. Pobieram.")
            return False
    except Exception as e:
        print(f" [WARN] Błąd sprawdzania CSV: {e}. Zakładam brak danych.")
        return False

# ────────────────────────────────────────────────
# 3. POBIERANIE Z API (ROZSZERZONY ZAKRES + FILTROWANIE)
# ────────────────────────────────────────────────
def fetch_api_data(mac, station_name, target_date_obj):
    # Start: 00:00:00 wybranego dnia
    start_str = target_date_obj.strftime("%Y-%m-%d 00:00:00")
    
    # Koniec: 01:00:00 DNIA NASTĘPNEGO (żeby złapać "ucięte" rekordy z 23:00-23:59)
    next_day = target_date_obj + timedelta(days=1)
    end_str = next_day.strftime("%Y-%m-%d 01:00:00")
    
    print(f"--- [API FETCH] {station_name} ---")
    print(f" Zakres zapytania (rozszerzony): {start_str} -> {end_str}")
    
    params = {
        "application_key": APP_KEY,
        "api_key": API_KEY,
        "mac": mac,
        "start_date": start_str,
        "end_date": end_str,
        "call_back": FIELDS
    }
    
    try:
        r = requests.get("https://api.ecowitt.net/api/v3/device/history", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        if data.get("code") != 0:
            print(f" [BLAD] API Code: {data.get('code')} Msg: {data.get('msg')}")
            return None
            
        hist = data.get("data", {})
        if not hist:
            print(" [INFO] Pusty obiekt 'data'.")
            return None
            
        temp_list = hist.get("outdoor", {}).get("temperature", {}).get("list", {})
        timestamps = sorted([int(k) for k in temp_list.keys() if k.isdigit()])
        
        if not timestamps:
            print(" [INFO] Brak rekordów.")
            return None
            
        print(f" [OK] Pobrano surowych rekordów: {len(timestamps)}")
        
        rows = []
        for ts in timestamps:
            str_ts = str(ts)
            # Konwersja na obiekt datetime UTC
            dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
            
            row = {
                "timestamp_utc": dt_utc.isoformat(), 
                "station": station_name,
                "dt_obj": dt_utc # tymczasowa kolumna do filtrowania
            }
            
            def get_val(g, s):
                return hist.get(g, {}).get(s, {}).get("list", {}).get(str_ts)
            
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
            row["ch1_temp_c"] = f_to_c(get_val("temp_ch1", "temperature"))
            row["ch2_temp_c"] = f_to_c(get_val("temp_ch2", "temperature"))
            row["ch3_temp_c"] = f_to_c(get_val("temp_ch3", "temperature"))
            row["lds_air_ft"] = safe_float(get_val("ch_lds1", "air_ch1"))
            row["lds_depth_ft"] = safe_float(get_val("ch_lds1", "depth_ch1"))
            row["lds_heat"] = safe_int(get_val("ch_lds1", "ldsheat_ch1"))
            
            rows.append(row)
        
        # Tworzymy DataFrame
        df = pd.DataFrame(rows)
        
        # ── FILTROWANIE PRECYZYJNE ──
        # Ponieważ pobraliśmy do 01:00 następnego dnia, musimy usunąć to, co "wylało się" poza właściwy dzień
        target_date_only = target_date_obj.date()
        
        # Filtrujemy: zostawiamy tylko rekordy gdzie data == target_date_only
        # (dzieki temu złapiemy 23:55 z właściwego dnia, ale wytniemy 00:05 z następnego)
        mask = df['dt_obj'].dt.date == target_date_only
        df_filtered = df.loc[mask].copy()
        
        # Sprzątanie tymczasowej kolumny
        df_filtered = df_filtered.drop(columns=['dt_obj'])
        
        print(f" [FILTR] Po odcięciu nadmiarowej godziny zostało {len(df_filtered)} rekordów dla {target_date_only}.")
        
        return df_filtered
        
    except Exception as e:
        print(f" [BLAD] API request: {e}")
        return None

# ────────────────────────────────────────────────
# 4. UPLOAD FTP
# ────────────────────────────────────────────────
def upload_ftp_file(local_file, remote_file):
    print(f"--- [FTP UPLOAD] Wysyłanie: {remote_file} ---")
    try:
        host = FTP_HOST_CFG
        if not host or "twoj_host" in host:
            print(" [INFO] Brak konfiguracji FTP. Pomijam upload.")
            return

        ftp = FTP(host)
        ftp.login(FTP_USER_CFG, FTP_PASS_CFG)
        try:
            ftp.cwd(FTP_DIR)
        except error_perm:
            try:
                ftp.mkd(FTP_DIR)
                ftp.cwd(FTP_DIR)
            except: pass

        with open(local_file, 'rb') as f:
            ftp.storbinary(f"STOR {remote_file}", f)
        
        ftp.quit()
        print(f" [SUKCES] Plik wysłany na FTP.")
    except Exception as e:
        print(f" [BLAD] Upload FTP: {e}")

# ────────────────────────────────────────────────
# GŁÓWNA LOGIKA
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n================ START SKRYPTU ================")
    
    # Ustawiamy cel: Wczoraj
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    target_date_str = yesterday.strftime("%Y-%m-%d")
    
    print(f" Cel: Pobranie i uzupełnienie danych dla: {target_date_str}")
    
    for station_name, mac in STATIONS.items():
        filename = FILES_MAP[station_name]
        print(f"\n >>> Stacja: {station_name} <<<")
        
        # 1. Pobierz obecny plik
        download_ftp_file(filename, filename)
        
        # 2. Sprawdź czy już mamy ten dzień
        if check_if_date_exists(filename, target_date_str):
            continue
        
        # 3. Pobierz z API (z marginesem +1h)
        df_new = fetch_api_data(mac, station_name, yesterday)
        
        if df_new is None or df_new.empty:
            print(f" [INFO] Brak nowych danych API dla {station_name}.")
            continue
            
        # 4. Dopisz do pliku
        try:
            if os.path.exists(filename):
                print(" [MERGE] Łączenie z plikiem lokalnym...")
                df_old = pd.read_csv(filename)
                
                df_old['timestamp_utc'] = pd.to_datetime(df_old['timestamp_utc'])
                df_new['timestamp_utc'] = pd.to_datetime(df_new['timestamp_utc'])
                
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                df_combined.drop_duplicates(subset=['timestamp_utc', 'station'], keep='last', inplace=True)
                df_combined.sort_values('timestamp_utc', inplace=True)
            else:
                print(" [NEW] Tworzę nowy plik.")
                df_combined = df_new
            
            df_combined.to_csv(filename, index=False)
            print(f" [SAVE] Zapisano. Wierszy razem: {len(df_combined)}")
            
            # 5. Wyślij
            upload_ftp_file(filename, filename)
            
        except Exception as e:
            print(f" [CRITICAL] Błąd operacji na plikach: {e}")

    print("\n================ KONIEC ================")

