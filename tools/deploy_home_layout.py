"""Back up and publish the reviewed home/calendar and climate color assets."""
import ftplib,os,io,hashlib,gzip,json,re
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from uuid import uuid4
ROOT='/meteo-krosno.pl/wp-content'; THEME=ROOT+'/themes/meteo-krosno-modern'; BACKUPS=THEME+'/cache/weather-backups'; CACHE=ROOT+'/cache/page_enhanced/meteo-krosno.pl'
STAMP=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8]
EXPECTED_APP='cc6a730801e64efc1679eff9b8c1527be391801f312f4bc71dd563282e46243c'
def read(f,p):
    b=io.BytesIO();f.retrbinary('RETR '+p,b.write);return b.getvalue()
def optional_read(f,p):
    try:return read(f,p)
    except ftplib.error_perm as exc:
        if str(exc).startswith('550') and PurePosixPath(p).name not in {PurePosixPath(n).name for n in f.nlst(str(PurePosixPath(p).parent))}:return None
        raise
def write(f,p,b):
    f.storbinary('STOR '+p,io.BytesIO(b));assert read(f,p)==b,'Write verification failed: '+p
def publish(f,p,b):
    tmp=p+'.layout-'+STAMP;write(f,tmp,b);f.rename(tmp,p);assert read(f,p)==b
with ftplib.FTP(os.environ['FTP_HOST'],os.environ['FTP_USER'],os.environ['FTP_PASS'],timeout=45) as f:
    protection=read(f,BACKUPS+'/.htaccess');assert b'denied' in protection or b'Deny from all' in protection
    try:
        ht=read(f,'/meteo-krosno.pl/.htaccess').decode('utf-8','replace')
        for line in ht.splitlines():
            if 'RewriteRule' in line and 'klimat-' in line:print('CLIMATE_ROUTE',line[:220])
    except ftplib.error_perm:pass
    files=[('site/vendor/astronomy-2.1.19.min.js',THEME+'/assets/astronomy-2.1.19.min.js'),('site/climate-panels-20261028.css',THEME+'/assets/climate-panels-20261028.css'),('site/mu-plugins/00-mk-climate-palette.php',ROOT+'/mu-plugins/00-mk-climate-palette.php'),('site/app.js',THEME+'/assets/app.js')]
    staged=[]
    for local,remote in files:
        old=optional_read(f,remote);new=Path(local).read_bytes()
        if remote.endswith('/app.js'):assert old==new or hashlib.sha256(old).hexdigest()==EXPECTED_APP,'Live app changed since review'
        else:assert old is None or old==new,'New asset path already occupied: '+remote
        staged.append((remote,old,new))
    path=THEME+'/assets/style.css';old=read(f,path);extra=Path('site/home-layout-20261028.css').read_bytes()
    if b'/* MK HOME 20261028 */' in old:
        assert old.endswith(extra),'Unexpected version of home styles';new=old
    else:new=old+b'\n'+extra
    # Styles are published before the new home template.
    staged.insert(0,(path,old,new))
    backup=BACKUPS+'/home-'+STAMP;f.mkd(backup)
    for i,(p,old,new) in enumerate(staged):
        if old is not None:write(f,backup+'/'+str(i)+'-'+PurePosixPath(p).name,old)
    manifest={'files':[{'path':p,'sha256':hashlib.sha256(old).hexdigest() if old is not None else None} for p,old,new in staged]}
    write(f,backup+'/manifest.json',json.dumps(manifest,indent=2).encode())
    changed=[]
    try:
        for p,old,new in staged:
            assert optional_read(f,p)==old,'File changed while staging: '+p
            if old==new:continue
            changed.append((p,old));publish(f,p,new);print('VERIFIED',p,hashlib.sha256(new).hexdigest())
    except Exception:
        for p,old in reversed(changed):
            if old is None:f.delete(p)
            else:publish(f,p,old)
        raise
    for slug in ('','klimat-miesieczny','klimat-roczny'):
        directory=CACHE+('/'+slug if slug else '')
        try:items=f.nlst(directory)
        except ftplib.error_perm:continue
        for item in items:
            name=PurePosixPath(item).name
            if not name.startswith('_index') or '.html' not in name:continue
            path=directory+'/'+name;content=read(f,path)
            write(f,backup+'/cache-'+(slug or 'home')+'-'+name,content);f.delete(path)
    print('BACKUP',backup)
