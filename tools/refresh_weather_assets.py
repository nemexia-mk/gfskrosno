import ftplib,os,io,gzip,re
from pathlib import PurePosixPath
ROOT='/meteo-krosno.pl/wp-content/cache/page_enhanced/meteo-krosno.pl'
BACKUP='/meteo-krosno.pl/wp-content/themes/meteo-krosno-modern/cache/weather-backups/20261022T175849Z-298ac382'
def read(ftp,path):
    out=io.BytesIO();ftp.retrbinary('RETR '+path,out.write);return out.getvalue()
with ftplib.FTP(os.environ['FTP_HOST'],os.environ['FTP_USER'],os.environ['FTP_PASS'],timeout=40) as ftp:
    count=0
    for slug in ('','meteorogram','prognoza'):
        directory=ROOT+('/'+slug if slug else '')
        try: names=ftp.nlst(directory)
        except ftplib.error_perm: continue
        for path in names:
            name=PurePosixPath(path).name
            if not name.startswith('_index') or '.html' not in name: continue
            source=directory+'/'+name
            data=read(ftp,source)
            plain=gzip.decompress(data) if data.startswith(b'\x1f\x8b') else data
            if b'meteo-krosno-modern/assets/app.js' not in plain: continue
            target=BACKUP+'/pagecache-'+(slug or 'home')+'-'+name
            ftp.storbinary('STOR '+target,io.BytesIO(data))
            assert read(ftp,target)==data,'Nieudana kopia cache'
            ftp.delete(source);count+=1;print('CACHE_REFRESHED',source)
    print('CACHE_FILES_BACKED_UP_AND_REFRESHED',count)
