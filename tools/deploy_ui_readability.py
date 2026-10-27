"""Publish only reviewed UI assets, verify backups and roll back failed writes."""
import ftplib,os,io,hashlib,gzip,json
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from uuid import uuid4
ROOT='/meteo-krosno.pl/wp-content'
THEME=ROOT+'/themes/meteo-krosno-modern'
BACKUPS=THEME+'/cache/weather-backups'
CACHE=ROOT+'/cache/page_enhanced/meteo-krosno.pl'
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8]
def read(f,p):
    b=io.BytesIO();f.retrbinary('RETR '+p,b.write);return b.getvalue()
def write(f,p,b):
    f.storbinary('STOR '+p,io.BytesIO(b));assert read(f,p)==b,'Verification failed: '+p
def publish(f,p,b):
    t=p+'.ui-'+stamp;write(f,t,b);f.rename(t,p);assert read(f,p)==b
expected=json.loads(Path('deployment/ui-20261027/expected.json').read_text())
with ftplib.FTP(os.environ['FTP_HOST'],os.environ['FTP_USER'],os.environ['FTP_PASS'],timeout=45) as f:
    # Existing protected backup area, established by previous deployments.
    protection=read(f,BACKUPS+'/.htaccess')
    assert b'denied' in protection or b'Deny from all' in protection,'Missing backup protection'
    backup=BACKUPS+'/ui-'+stamp;f.mkd(backup)
    staged=[]
    for name,path in [('app.js',THEME+'/assets/app.js'),('station-live.js',ROOT+'/plugins/mk-station-dashboards/assets/station-live.js')]:
        old=read(f,path);new=Path('site/'+name).read_bytes()
        assert old==new or hashlib.sha256(old).hexdigest()==expected[name],'Live asset changed; stop: '+path
        staged.append((path,old,new))
    csspath=THEME+'/assets/style.css';old=read(f,csspath);extra=Path('site/ui-readability.css').read_bytes()
    marker=b'/* Meteo Krosno: czytelno'
    if marker in old:
        assert old.endswith(extra),'Different readability styles already present'
        staged.append((csspath,old,old))
    else:staged.append((csspath,old,old+b'\n'+extra))
    # Cache-bust the static station script URL in its existing PHP renderer.
    candidates=[]
    for directory in (ROOT+'/plugins/mk-station-dashboards',ROOT+'/mu-plugins'):
        for item in f.nlst(directory):
            name=PurePosixPath(item).name
            if not name.endswith('.php'):continue
            path=directory+'/'+name;code=read(f,path)
            needle=b'station-live.js?v=2.5.3'
            target=b'station-live.js?v=2.5.4'
            if needle in code:
                staged.append((path,code,code.replace(needle,target)));candidates.append(path)
            elif target in code:candidates.append(path)
    assert candidates,'Static station script URL not found; inspect renderer before changing it'
    print('STATION_CACHE_VERSION',candidates)
    for i,(p,old,new) in enumerate(staged):write(f,backup+'/'+str(i)+'-'+PurePosixPath(p).name,old)
    write(f,backup+'/manifest.json',json.dumps({'files':[{'path':p,'sha256':hashlib.sha256(old).hexdigest()} for p,old,new in staged]},indent=2).encode())
    changed=[]
    try:
        for p,old,new in staged:
            assert read(f,p)==old,'File changed while staging: '+p
            if old==new:continue
            changed.append((p,old));publish(f,p,new)
            print('VERIFIED',p,hashlib.sha256(new).hexdigest())
    except Exception:
        for p,old in reversed(changed):publish(f,p,old)
        raise
    # Back up the matching page caches before removing them; sources are untouched.
    for slug in ('','prognoza','meteorogram','stacje','sprawdzalnosc','stacja-turaszowka','stacja-lesniowka'):
        directory=CACHE+('/'+slug if slug else '')
        try:names=f.nlst(directory)
        except ftplib.error_perm:continue
        for item in names:
            name=PurePosixPath(item).name
            if not name.startswith('_index') or '.html' not in name:continue
            p=directory+'/'+name;content=read(f,p)
            plain=gzip.decompress(content) if content.startswith(b'\x1f\x8b') else content
            if b'meteo-krosno-modern' not in plain and b'mk-station-dashboards' not in plain:continue
            write(f,backup+'/cache-'+(slug or 'home')+'-'+name,content);f.delete(p)
    print('BACKUP',backup)
