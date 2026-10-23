import ftplib, os, io, hashlib, gzip, json
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
ASSET='/meteo-krosno.pl/wp-content/themes/meteo-krosno-modern/assets/app.js'
BACKUP='/meteo-krosno.pl/wp-content/themes/meteo-krosno-modern/cache/weather-backups/20261022T175849Z-298ac382'
CACHE='/meteo-krosno.pl/wp-content/cache/page_enhanced/meteo-krosno.pl'
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def read(ftp,path):
    out=io.BytesIO(); ftp.retrbinary('RETR '+path,out.write); return out.getvalue()
def save(ftp,path,data):
    ftp.storbinary('STOR '+path,io.BytesIO(data))
    assert read(ftp,path)==data,'Byte verification failed: '+path
data=Path('site/app.js').read_bytes()
with ftplib.FTP(os.environ['FTP_HOST'],os.environ['FTP_USER'],os.environ['FTP_PASS'],timeout=40) as ftp:
    old=read(ftp,ASSET)
    if old==data:
        print('T2M_LAYER_ALREADY_VERIFIED',hashlib.sha256(data).hexdigest())
        raise SystemExit(0)
    assert hashlib.sha256(old).hexdigest()=='375803d674b5e9dadd8056f5bb49704546f801dbb8ec35d1b4b8cb87e1ba1a81','Live asset changed; stop to avoid overwriting'
    backup=BACKUP+'/app-before-t2m-layer-'+stamp+'.js'
    save(ftp,backup,old)
    temporary=ASSET+'.layer-'+stamp
    save(ftp,temporary,data)
    ftp.rename(temporary,ASSET)
    assert read(ftp,ASSET)==data
    print('T2M_LAYER_DEPLOYED',hashlib.sha256(data).hexdigest(),'BACKUP',backup)
    for slug in ('','meteorogram','prognoza'):
        directory=CACHE+('/'+slug if slug else '')
        try: names=ftp.nlst(directory)
        except ftplib.error_perm: continue
        for path in names:
            name=PurePosixPath(path).name
            if not name.startswith('_index') or '.html' not in name: continue
            source=directory+'/'+name; cached=read(ftp,source)
            plain=gzip.decompress(cached) if cached.startswith(b'\x1f\x8b') else cached
            if b'meteo-krosno-modern/assets/app.js' not in plain: continue
            save(ftp,BACKUP+'/layer-pagecache-'+stamp+'-'+(slug or 'home')+'-'+name,cached)
            ftp.delete(source)
            print('PAGE_CACHE_REFRESHED',slug or 'home')
