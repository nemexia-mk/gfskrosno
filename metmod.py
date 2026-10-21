"""Meteomodel -> zbiorczy.csv: walidacja, upsert, blokada i atomowa publikacja.
Braki nigdy nie kasują pomiarów. Nie zamieniamy opadu 24 h sumą 12 h.
"""
import argparse
import csv
import ftplib
import io
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG = logging.getLogger('metmod')
STATION, FILE_NAME, WIDTH = '349210670', 'zbiorczy.csv', 64
FIELDS = {
    5: ('Maksymalna temperatura dobowa', -90, 60),
    6: ('Minimalna temperatura dobowa', -90, 60),
    7: ('Średnia temperatura dobowa', -90, 60),
    8: ('Minimalna temperatura przy gruncie', -100, 70),
    9: ('Opad dobowy za BIEŻĄCĄ dobę', 0, 1000),
    12: ('Wysokość pokrywy śnieżnej', 0, 500),
    16: ('Usłonecznienie za poprzednią dobę', 0, 24),
    61: ('Średnia dobowa prędkość wiatru', 0, 100),
    62: ('Maksymalny poryw', 0, 150),
    63: ('Średnie ciśnienie zredukowane do poziomu morza', 850, 1100),
}


def normal(text):
    return ' '.join(str(text).replace('\\n', ' ').split()).casefold()


def number(text, lo, hi):
    text = str(text).strip().replace('−', '-').replace(',', '.')
    if text in ('', '-', '—', '–', 'NaN', 'None'):
        return ''
    if not re.fullmatch(r'[+-]?\d+(?:\.\d+)?', text):
        raise ValueError(f'Nieprawidłowa liczba: {text!r}')
    value = Decimal(text)
    if not value.is_finite() or not Decimal(str(lo)) <= value <= Decimal(str(hi)):
        raise ValueError(f'Wartość poza zakresem: {text}')
    return format(value, 'f').replace('.', ',')


def parse_day(html, target):
    table = BeautifulSoup(html, 'html.parser').find('table', id='tablepl')
    if table is None:
        raise ValueError('Brak tabeli pomiarowej tablepl')
    header = next((tr for tr in table.find_all('tr') if tr.find(['td', 'th'])
                   and tr.find(['td', 'th']).get_text(strip=True) == 'Data'), None)
    if header is None:
        raise ValueError('Brak nagłówków tabeli')
    heads, mapping = header.find_all(['td', 'th']), {}
    for dest, (title, _, _) in FIELDS.items():
        found = [i for i, cell in enumerate(heads)
                 if normal(cell.get('title', '')).startswith(normal(title))]
        if len(found) != 1:
            raise ValueError(f'Zmieniony nagłówek: {title}')
        mapping[dest] = found[0]
    matches = [tr.find_all('td') for tr in table.find_all('tr') if tr.find('td')
               and tr.find('td').get_text(strip=True) == target.isoformat()]
    if len(matches) != 1 or len(matches[0]) != len(heads):
        raise ValueError(f'Brak jednoznacznego, pełnego wiersza dla {target}')
    values = {dest: number(matches[0][src].get_text(' ', strip=True), *FIELDS[dest][1:])
              for dest, src in mapping.items()}
    if any(values[k] == '' for k in (5, 6, 7)):
        raise ValueError('Niekompletna temperatura')
    tx, tn, tm = (Decimal(values[k].replace(',', '.')) for k in (5, 6, 7))
    if not tn <= tm <= tx:
        raise ValueError('Niespójna temperatura Tmin/Tśr/Tmax')
    if values[12] != '':
        values[50] = '1' if Decimal(values[12].replace(',', '.')) > 0 else '0'
    return values


def session():
    client = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=('GET',), respect_retry_after_header=True)
    client.mount('https://', HTTPAdapter(max_retries=retry))
    client.headers['User-Agent'] = 'Meteo-Krosno-metmod/2.0'
    return client


def fetch_day(client, target):
    r = client.get('https://meteomodel.pl/aktualne-dane-pomiarowe/',
                   params={'data': target.isoformat(), 'rodzaj': 'da', 'wmoid': '12670',
                           'dni': 1, 'ord': 'asc'}, timeout=(10, 30))
    r.raise_for_status()
    return parse_day(r.content, target)


def row_key(row):
    try:
        parts = [Decimal(row[i].strip().replace(',', '.')) for i in (2, 3, 4)]
        if any(not v.is_finite() or v != v.to_integral_value() for v in parts):
            raise ValueError('Ułamkowa data CSV')
        return row[0].strip(), date(*(int(v) for v in parts))
    except (ArithmeticError, ValueError, IndexError) as exc:
        raise ValueError('Nieprawidłowa data istniejącego CSV') from exc


def merge_csv(raw, updates):
    if not raw:
        raise ValueError('Puste archiwum; odmowa automatycznego tworzenia CSV')
    try:
        text = raw.decode('utf-8-sig')
        encoding = 'utf-8-sig' if raw.startswith(b'\xef\xbb\xbf') else 'utf-8'
    except UnicodeDecodeError:
        text, encoding = raw.decode('cp1250'), 'cp1250'
    rows = list(csv.reader(io.StringIO(text, newline=''), delimiter=';'))
    header = []
    if rows and any('TMAX' in v.upper() for v in rows[0]) and not rows[0][0].strip().isdigit():
        header = [rows.pop(0)]
    if not rows or any(len(r) < 51 for r in rows):
        raise ValueError('Nieprawidłowa szerokość istniejącego CSV')
    width, merged = max(WIDTH, max(map(len, rows))), {}
    for source in rows:
        key = row_key(source)
        row = source + [''] * (width - len(source))
        if key in merged:
            row = [new if new.strip() else old for old, new in zip(merged[key], row)]
        merged[key] = row
    count = len(merged)
    for target, values in sorted(updates.items()):
        key = (STATION, target)
        row = merged.get(key, [''] * width)
        row[:5] = [STATION, 'KROSNO'.ljust(30), str(target.year), str(target.month), str(target.day)]
        for column, value in values.items():
            if value != '':
                row[column] = value
        merged[key] = row
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, delimiter=';', lineterminator='\r\n')
    writer.writerows(header)
    writer.writerows(merged[k] for k in sorted(merged, key=lambda k: (k[1], k[0])))
    result = buffer.getvalue().encode(encoding, errors='strict')
    if len(merged) < count or len(result) < len(raw) * .85 * count / len(rows):
        raise ValueError('Podejrzane zmniejszenie archiwum; wymagana kontrola')
    return result


def retrieve(ftp, name):
    buffer = io.BytesIO()
    ftp.retrbinary('RETR ' + name, buffer.write)
    return buffer.getvalue()


def publish(updates):
    user, password = os.getenv('FTP_USER'), os.getenv('FTP_PASS')
    if not user or not password:
        raise ValueError('Brak FTP_USER lub FTP_PASS')
    cls = ftplib.FTP_TLS if os.getenv('FTP_TLS', '0') == '1' else ftplib.FTP
    ftp, locked, token = cls(timeout=30), False, uuid4().hex
    lock, temporary = '.metmod.lock', '.zbiorczy.' + token + '.upload'
    try:
        ftp.connect(os.getenv('FTP_HOST', 'corn.cba.pl'))
        ftp.login(user, password)
        if isinstance(ftp, ftplib.FTP_TLS):
            ftp.prot_p()
        ftp.cwd(os.getenv('FTP_DIR', 'stacja.meteo-krosno.pl/'))
        ftp.mkd(lock)
        locked = True
        before = retrieve(ftp, FILE_NAME)  # Błąd RETR nigdy nie oznacza nowego pliku.
        after = merge_csv(before, updates)
        if after == before:
            LOG.info('Bez zmian; archiwum jest aktualne')
            return
        backup = FILE_NAME + '.' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.' + token[:8] + '.bak'
        ftp.storbinary('STOR ' + backup, io.BytesIO(before))
        if retrieve(ftp, backup) != before:
            raise ValueError('Błąd weryfikacji kopii bezpieczeństwa')
        ftp.storbinary('STOR ' + temporary, io.BytesIO(after))
        if retrieve(ftp, temporary) != after:
            raise ValueError('Błąd weryfikacji nowego CSV')
        if retrieve(ftp, FILE_NAME) != before:
            raise ValueError('Równoległa zmiana archiwum; odmowa nadpisania')
        ftp.rename(temporary, FILE_NAME)  # Brak fallbacku na nieatomowy STOR.
        if retrieve(ftp, FILE_NAME) != after:
            raise ValueError('Błąd kontroli po publikacji; kopia zachowana')
        LOG.info('CSV zweryfikowany i opublikowany; kopia: %s', backup)
    finally:
        if locked:
            try:
                ftp.delete(temporary)
            except ftplib.all_errors:
                pass
            try:
                ftp.rmd(lock)
            except ftplib.all_errors:
                LOG.warning('Nie usunięto blokady FTP .metmod.lock; wymagana kontrola')
        try:
            ftp.quit()
        except ftplib.all_errors:
            ftp.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', type=date.fromisoformat, help='Ostatnia doba YYYY-MM-DD')
    parser.add_argument('--lookback', type=int, default=7, help='Ponowny odczyt dni (1–31)')
    parser.add_argument('--dry-run', action='store_true', help='Test źródła bez FTP')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    today = datetime.now(ZoneInfo('Europe/Warsaw')).date()
    target = args.date or today - timedelta(days=1)
    if not 1 <= args.lookback <= 31 or target >= today:
        parser.error('Wymagana zakończona doba i lookback 1–31')
    updates, failures = {}, []
    with session() as client:
        for offset in range(args.lookback - 1, -1, -1):
            day = target - timedelta(days=offset)
            try:
                updates[day] = fetch_day(client, day)
                LOG.info('%s: poprawny wiersz; %s/%s parametrów', day,
                         sum(v != '' for v in updates[day].values()), len(updates[day]))
            except (requests.RequestException, ValueError) as exc:
                LOG.warning('%s: pominięto (%s)', day, exc)
                failures.append(day)
    if not updates:
        LOG.error('Brak poprawnych danych; CSV pozostaje bez zmian')
        return 1
    if not args.dry_run:
        try:
            publish(updates)
        except (OSError, ValueError, *ftplib.all_errors) as exc:
            LOG.error('Publikacja przerwana: %s', exc)
            return 1
    return 1 if target in failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
