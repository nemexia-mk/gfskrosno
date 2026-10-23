import ftplib,os,io,hashlib,gzip
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
ROOT='/meteo-krosno.pl/wp-content'
BACKUP=ROOT+'/themes/meteo-krosno-modern/cache/weather-backups/20261022T175849Z-298ac382'
CACHE=ROOT+'/cache/page_enhanced/meteo-krosno.pl'
STAMP=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
HASHES={'js':'83cd11ee6fc870e1441faa5a4c34dc0eb38ace72056b45d453c655828472eff5','css':'a7934aee085d60abdca57545dde05c0ae92332a694e20e5255a1f82e4dbc9125'}
PLUGIN=ROOT+'/mu-plugins/mk-station-wind-analytics.php'
def read(f,p):
    b=io.BytesIO();f.retrbinary('RETR '+p,b.write);return b.getvalue()
def write(f,p,b):
    f.storbinary('STOR '+p,io.BytesIO(b));assert read(f,p)==b,'Verification '+p
def publish(f,p,b):
    t=p+'.station-'+STAMP;write(f,t,b);f.rename(t,p);assert read(f,p)==b
with ftplib.FTP(os.environ['FTP_HOST'],os.environ['FTP_USER'],os.environ['FTP_PASS'],timeout=45) as f:
    staged=[]
    for ext,expected in HASHES.items():
        p=ROOT+'/plugins/mk-station-dashboards/assets/dashboard.'+ext
        old=read(f,p);new=Path('station-dashboard/assets/dashboard.'+ext).read_bytes()
        assert old==new or hashlib.sha256(old).hexdigest()==expected,'Public asset changed; stop '+p
        staged.append((p,old,new))
    plugin=Path('station-dashboard/mu-plugins/mk-station-wind-analytics.php').read_bytes()
    try:names=f.nlst(ROOT+'/mu-plugins')
    except ftplib.error_perm:f.mkd(ROOT+'/mu-plugins');names=[]
    exists=any(PurePosixPath(n).name==PurePosixPath(PLUGIN).name for n in names)
    if exists:assert read(f,PLUGIN)==plugin,'New module path occupied; stop'
    for p,old,new in staged:
        if old!=new:write(f,BACKUP+'/station-before-'+STAMP+'-'+PurePosixPath(p).name,old)
    changed=[];created=False
    try:
        for p,old,new in staged:
            if old!=new:publish(f,p,new);changed.append((p,old))
        if not exists:publish(f,PLUGIN,plugin);created=True
    except Exception:
        for p,old in reversed(changed):publish(f,p,old)
        if created:f.delete(PLUGIN)
        raise
    for slug in ('stacja-turaszowka','stacja-lesniowka'):
        directory=CACHE+'/'+slug
        try:names=f.nlst(directory)
        except ftplib.error_perm:continue
        for path in names:
            name=PurePosixPath(path).name
            if not name.startswith('_index') or '.html' not in name:continue
            source=directory+'/'+name;cached=read(f,source)
            plain=gzip.decompress(cached) if cached.startswith(b'\x1f\x8b') else cached
            if b'mk-station-dashboards/assets/dashboard.js' not in plain:continue
            write(f,BACKUP+'/station-cache-'+STAMP+'-'+slug+'-'+name,cached);f.delete(source)
    print('STATION_INSTRUMENTS_DEPLOYED; backups:',BACKUP,'timestamp:',STAMP)
