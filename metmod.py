import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import ftplib
import io
import sys
import os

# --- KONFIGURACJA (Maskowanie dla GitHub Actions) ---
FTP_HOST_CFG = os.getenv("FTP_HOST", "corn.cba.pl") 
FTP_USER_CFG = os.getenv("FTP_USER", "stacja2@stacja.meteo-krosno.pl")
FTP_PASS_CFG = os.getenv("FTP_PASS", "Mateusz07")
FTP_DIR = "stacja.meteo-krosno.pl/"  # Ścieżka bazowa na serwerze
FILE_NAME = "zbiorczy.csv"           # Nazwa pliku
WMO_ID = '12670'

def safe_get(lst, idx, default=''):
    return lst[idx] if idx < len(lst) else default

# --- KROK 1: Data ---
yesterday = (datetime.now() - timedelta(days=1))
current_date = yesterday.strftime('%Y-%m-%d')
year, month, day = yesterday.year, yesterday.month, yesterday.day

print(f"[{datetime.now().strftime('%H:%M:%S')}] Rozpoczynam dla: {current_date}")

# --- KROK 2: Pobranie danych ze strony Meteomodel ---
try:
    url = f"https://meteomodel.pl/aktualne-dane-pomiarowe/?data={current_date}&rodzaj=da&wmoid={WMO_ID}&dni=1&ord=asc"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    
    rows = []
    for tr in table.find_all('tr')[1:]:
        row = [img.get('alt', '1') if (img := td.find('img')) else td.text.strip() for td in tr.find_all('td')]
        if row: rows.append(row)
    
    if not rows:
        print("Brak danych w tabeli na stronie.")
        sys.exit(0) # Kończymy bez błędu, po prostu brak danych

    data_row = rows[0] if rows[0][0] != 'Data' else rows[1]
    print(f"Pobrano dane: {data_row}")
except Exception as e:
    print(f"BŁĄD pobierania danych: {e}")
    sys.exit(1)

# --- KROK 3: FTP i wczytywanie pliku ---
ftp = ftplib.FTP(FTP_HOST_CFG)
try:
    ftp.login(FTP_USER_CFG, FTP_PASS_CFG)
    
    # Przejdź do katalogu jeśli podano
    if FTP_DIR:
        try:
            ftp.cwd(FTP_DIR)
        except:
            print(f"Nie można wejść do katalogu {FTP_DIR} (może już w nim jesteś)")

    raw_content = io.BytesIO()
    file_exists = False
    try:
        ftp.retrbinary(f"RETR {FILE_NAME}", raw_content.write)
        raw_content.seek(0)
        file_exists = True
    except ftplib.error_perm:
        print(f"Plik {FILE_NAME} nie istnieje, zostanie utworzony.")

    if file_exists:
        content = raw_content.getvalue()
        try:
            text_content = content.decode('cp1250')
        except UnicodeDecodeError:
            text_content = content.decode('utf-8', errors='replace')
        
        df_collective = pd.read_csv(io.StringIO(text_content), sep=';', header=None)
        print(f"Wczytano {len(df_collective)} wierszy.")
    else:
        df_collective = pd.DataFrame()

    # --- KROK 4: Przygotowanie nowego wiersza ---
    # Nazwa z Twojego pliku (dokładnie 30 znaków licząc ze spacjami)
    nazwa_stacji = "KROSNO                        " 
    REQUIRED_COLS = 64
    new_data = [''] * REQUIRED_COLS

    def fmt(val):
        if val == '-' or val is None: return ''
        return str(val).replace('.', ',')

    new_data[0] = '349210670'
    new_data[1] = nazwa_stacji
    new_data[2] = year
    new_data[3] = month
    new_data[4] = day
    new_data[5] = fmt(safe_get(data_row, 2))   # TMAX
    new_data[6] = fmt(safe_get(data_row, 3))   # TMIN
    new_data[7] = fmt(safe_get(data_row, 4))   # TSR
    new_data[8] = fmt(safe_get(data_row, 6))   # T5CM
    new_data[9] = fmt(safe_get(data_row, 17))  # OPADY
    new_data[12] = fmt(safe_get(data_row, 21)) # SNOW
    new_data[16] = fmt(safe_get(data_row, 20)) # SUN
    new_data[61] = fmt(safe_get(data_row, 12)) # ws
    new_data[62] = fmt(safe_get(data_row, 14)) # g1
    new_data[63] = fmt(safe_get(data_row, 15)) # mslp

    # --- KROK 5: Łączenie i wysyłka ---
    new_row_df = pd.DataFrame([new_data])
    df_collective = pd.concat([df_collective, new_row_df], ignore_index=True)

    output_buffer = io.BytesIO()
    csv_text = df_collective.to_csv(index=False, header=False, sep=';', lineterminator='\r\n')
    output_buffer.write(csv_text.encode('cp1250', errors='replace'))
    output_buffer.seek(0)

    ftp.storbinary(f"STOR {FILE_NAME}", output_buffer)
    print("SUKCES: Plik zaktualizowany na FTP.")

except Exception as e:
    print(f"BŁĄD FTP: {e}")
    sys.exit(1)
finally:
    try:
        ftp.quit()
    except:
        pass
