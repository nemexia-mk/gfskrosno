"""Publish rain verification with guarded sources, private backups and rollback."""
import ftplib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from urllib.request import urlopen
from uuid import uuid4

ROOT = '/meteo-krosno.pl/wp-content'
THEME = ROOT + '/themes/meteo-krosno-modern'
BACKUPS = THEME + '/cache/weather-backups'
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
EXPECTED_APP = 'c5fbba99bf25478b497c99516243d5bcfe530745ef94bc3e5ccffd634ac51c6e'


def read(ftp, path):
    buffer = io.BytesIO()
    ftp.retrbinary('RETR ' + path, buffer.write)
    return buffer.getvalue()


def optional_read(ftp, path):
    try:
        return read(ftp, path)
    except ftplib.error_perm as error:
        if str(error).startswith('550') and PurePosixPath(path).name not in {PurePosixPath(item).name for item in ftp.nlst(str(PurePosixPath(path).parent))}:
            return None
        raise


def write(ftp, path, content):
    ftp.storbinary('STOR ' + path, io.BytesIO(content))
    assert read(ftp, path) == content, 'Readback failed: ' + path


def publish(ftp, path, content):
    temporary = path + '.rain-' + STAMP
    write(ftp, temporary, content)
    ftp.rename(temporary, path)
    assert read(ftp, path) == content, 'Publish readback failed: ' + path


def main():
    with ftplib.FTP(os.environ['FTP_HOST'], os.environ['FTP_USER'], os.environ['FTP_PASS'], timeout=45) as ftp:
        protection = read(ftp, BACKUPS + '/.htaccess')
        assert b'denied' in protection or b'Deny from all' in protection, 'Backups must be private'
        staged = []
        for local, remote in [
            ('site/mu-plugins/mk-gfs-rain-verification.php', ROOT + '/mu-plugins/mk-gfs-rain-verification.php'),
            ('site/app.js', THEME + '/assets/app.js'),
        ]:
            old = optional_read(ftp, remote)
            new = Path(local).read_bytes()
            if remote.endswith('/app.js'):
                assert old == new or (old is not None and hashlib.sha256(old).hexdigest() == EXPECTED_APP), 'Live frontend changed since review'
            else:
                assert old is None or old == new, 'Rain plugin path already occupied by different code'
            staged.append((remote, old, new))
        backup = BACKUPS + '/rain-' + STAMP
        ftp.mkd(backup)
        manifest = []
        for index, (path, old, new) in enumerate(staged):
            name = str(index) + '-' + PurePosixPath(path).name
            if old is not None:
                write(ftp, backup + '/' + name, old)
            manifest.append({'path': path, 'backup': name if old is not None else None, 'sha256': hashlib.sha256(old).hexdigest() if old is not None else None})
        write(ftp, backup + '/manifest.json', json.dumps(manifest, indent=2).encode())
        changed = []
        try:
            for path, old, new in staged:
                assert optional_read(ftp, path) == old, 'Live file changed while staging: ' + path
                if old == new:
                    continue
                changed.append((path, old))
                publish(ftp, path, new)
                print('VERIFIED', path, hashlib.sha256(new).hexdigest(), flush=True)
            # Verify routing before concluding the transaction; the old temperature API stays intact.
            for station in ['turaszowka', 'lesniowka']:
                with urlopen('https://meteo-krosno.pl/?mk_api=verification&metric=rain&station=' + station + '&rain_check=' + STAMP, timeout=50) as response:
                    assert response.status == 200
                    data = json.load(response)
                    assert data['metric'] == 'rain' and data['station'] == station and data['intervalHours'] == 6
                    assert len(data['stats']) == 3 and isinstance(data['pairs'], list)
            with urlopen('https://meteo-krosno.pl/?mk_api=verification&rain_check=' + STAMP, timeout=50) as response:
                data = json.load(response)
                assert response.status == 200 and 'stats' in data and 'pairs' in data and data.get('metric') != 'rain'
        except Exception:
            for path, old in reversed(changed):
                if old is None:
                    ftp.delete(path)
                else:
                    publish(ftp, path, old)
            raise
        # Only this route needs a new app.js version in its cached HTML.
        directory = ROOT + '/cache/page_enhanced/meteo-krosno.pl/sprawdzalnosc'
        try:
            names = ftp.nlst(directory)
        except ftplib.error_perm:
            names = []
        for item in names:
            name = PurePosixPath(item).name
            if not name.startswith('_index') or '.html' not in name:
                continue
            path = directory + '/' + name
            try:
                content = read(ftp, path)
                write(ftp, backup + '/cache-' + name, content)
                ftp.delete(path)
            except ftplib.error_perm as error:
                if not (str(error).startswith('550') and 'No such file' in str(error)):
                    raise
        print('BACKUP', backup, flush=True)


if __name__ == '__main__':
    main()
