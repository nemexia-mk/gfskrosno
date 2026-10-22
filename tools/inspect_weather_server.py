import ftplib
import io
import os
import csv

with ftplib.FTP(os.environ['FTP_HOST'], os.environ['FTP_USER'], os.environ['FTP_PASS'], timeout=40) as ftp:
    print('FTP_ROOT', ftp.pwd(), ftp.nlst())
    for directory in ('/stacja.meteo-krosno.pl', '/meteo-krosno.pl'):
        try:
            ftp.cwd(directory)
            print('DIRECTORY', directory, ftp.nlst())
        except ftplib.error_perm as exc:
            print('DIRECTORY_MISSING', directory, str(exc))
    for path in ('/stacja.meteo-krosno.pl/gfs-tab.csv', '/stacja.meteo-krosno.pl/gfs-conv.csv'):
        try:
            b = io.BytesIO(); ftp.retrbinary('RETR ' + path, b.write)
            rows = list(csv.reader(io.StringIO(b.getvalue().decode('utf-8-sig'))))
            print('CSV_SCHEMA', path, rows[0], 'ROWS', len(rows)-1)
        except ftplib.all_errors as exc:
            print('CSV_MISSING', path, str(exc))
    for directory in ('/meteo-krosno.pl/wp-content/themes/meteo-krosno-modern', '/meteo-krosno.pl/public_html/wp-content/themes/meteo-krosno-modern'):
        try:
            ftp.cwd(directory); print('THEME', directory, ftp.nlst())
        except ftplib.error_perm:
            pass
