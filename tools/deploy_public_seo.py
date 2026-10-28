"""Publish SEO through the existing FTP workflow, retaining protected backups."""
import ftplib
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from datetime import datetime, timezone
from uuid import uuid4
from urllib.request import urlopen

ROOT = '/meteo-krosno.pl'
THEME = ROOT + '/wp-content/themes/meteo-krosno-modern'
BACKUPS = THEME + '/cache/weather-backups'
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]


def read(ftp, path):
    stream = io.BytesIO()
    ftp.retrbinary('RETR ' + path, stream.write)
    return stream.getvalue()


def optional_read(ftp, path):
    try:
        return read(ftp, path)
    except ftplib.error_perm as exc:
        if str(exc).startswith('550'):
            names = {PurePosixPath(p).name for p in ftp.nlst(str(PurePosixPath(path).parent))}
            if PurePosixPath(path).name not in names:
                return None
        raise


def write(ftp, path, data):
    ftp.storbinary('STOR ' + path, io.BytesIO(data))
    assert read(ftp, path) == data, 'Write verification failed: ' + path


def publish(ftp, path, data):
    temporary = path + '.seo-' + STAMP
    write(ftp, temporary, data)
    ftp.rename(temporary, path)
    assert read(ftp, path) == data


def clear_page_cache(ftp, directory, depth=0):
    if depth > 12:
        raise RuntimeError('Unexpected page-cache nesting')
    try:
        entries = list(ftp.mlsd(directory))
    except ftplib.error_perm as exc:
        if str(exc).startswith('550'):
            return
        raise
    for name, facts in entries:
        if name in ('.', '..') or '/' in name:
            continue
        path = directory + '/' + name
        if facts.get('type') == 'dir':
            clear_page_cache(ftp, path, depth + 1)
        elif facts.get('type') == 'file' and name.startswith('_index') and '.html' in name:
            try:
                ftp.delete(path)  # Generated W3 cache only; no user content.
            except ftplib.error_perm as exc:
                # WordPress/W3 may invalidate the same cache after drafting a page.
                if str(exc).startswith('550') and 'No such file' in str(exc):
                    continue
                raise


def main():
    with ftplib.FTP(os.environ['FTP_HOST'], os.environ['FTP_USER'], os.environ['FTP_PASS'], timeout=45) as ftp:
        protection = read(ftp, BACKUPS + '/.htaccess')
        assert b'denied' in protection or b'Deny from all' in protection, 'Backups must be private'
        plugin_path = ROOT + '/wp-content/mu-plugins/mk-public-seo.php'
        plugin_old = optional_read(ftp, plugin_path)
        plugin_new = Path('site/mu-plugins/mk-public-seo.php').read_bytes()
        previous = {'a462894b3cade867c54b6b1d4561d3494544b72c246a46298dc0d428f3100115', '1d6cb92fb55f98c12c511dc4b8f4e67ea8aa3cf9708f8fd4d6cec1f33c6b22d4', 'a2b509ec8e3b197a8b5fd6090e2ce97838d5bd0d4b44d2ddb5dd8d98045e1ef0'}
        assert plugin_old in (None, plugin_new) or hashlib.sha256(plugin_old).hexdigest() in previous, 'SEO module changed since review'
        app_path = THEME + '/assets/app.js'
        app_old = read(ftp, app_path)
        app_new = Path('site/app.js').read_bytes()
        # The only frontend change is the heading; accept unrelated live updates.
        old_heading = '<div class="mk-day-date"><h1>\'+esc(fullLabel(today))+\'</h1><p id="mk-day-duration">'.encode()
        new_heading = '<div class="mk-day-date"><h1>Pogoda w Krośnie</h1><p>\'+esc(fullLabel(today))+\'</p><p id="mk-day-duration">'.encode()
        if old_heading in app_old:
            assert app_old.count(old_heading) == 1
            app_new = app_old.replace(old_heading, new_heading)
        else:
            assert new_heading in app_old, 'Unexpected home heading'
            app_new = app_old
        # Check the exact generated JS before any remote writes.
        checkfile = Path('/tmp/meteo-seo-app.js'); checkfile.write_bytes(app_new)
        subprocess.run(['node', '--check', str(checkfile)], check=True)
        robots_path = ROOT + '/robots.txt'
        robots_old = optional_read(ftp, robots_path)
        if robots_old:
            text = robots_old.decode('utf-8-sig')
            text = re.sub(r'^\s*Sitemap:.*$', '', text, flags=re.M | re.I).rstrip()
        else:
            text = 'User-agent: *\nDisallow: /wp-admin/\nAllow: /wp-admin/admin-ajax.php'
        robots_new = (text + '\n\nSitemap: https://meteo-krosno.pl/sitemap.xml\n').encode()
        files = [(app_path, app_old, app_new), (robots_path, robots_old, robots_new), (plugin_path, plugin_old, plugin_new)]
        backup = BACKUPS + '/seo-' + STAMP; ftp.mkd(backup)
        manifest = []
        for i, (path, old, new) in enumerate(files):
            backup_name = str(i) + '-' + PurePosixPath(path).name
            if old is not None:
                write(ftp, backup + '/' + backup_name, old)
            manifest.append({'path': path, 'backup': backup_name if old is not None else None, 'sha256': hashlib.sha256(old).hexdigest() if old is not None else None})
        write(ftp, backup + '/manifest.json', json.dumps(manifest, indent=2).encode())
        changed = []
        try:
            for path, old, new in files:
                assert optional_read(ftp, path) == old, 'Live file changed during staging: ' + path
                if old == new:
                    continue
                changed.append((path, old))
                publish(ftp, path, new)
                print('VERIFIED', path, hashlib.sha256(new).hexdigest(), flush=True)
        except Exception:
            for path, old in reversed(changed):
                if old is None:
                    ftp.delete(path)
                else:
                    publish(ftp, path, old)
            raise
        # A noncached request applies the draft migration and one-time native W3 flush.
        with urlopen('https://meteo-krosno.pl/?mkseo_refresh=' + STAMP, timeout=40) as response:
            assert response.status == 200 and b'mk-seo-content' in response.read(), 'SEO activation failed'
        clear_page_cache(ftp, ROOT + '/wp-content/cache/page_enhanced/meteo-krosno.pl')
        print('BACKUP', backup, flush=True)
        print('Sample status backup is retained privately in the WordPress option mkseo_sample_previous_status_v1.', flush=True)


if __name__ == '__main__':
    main()
