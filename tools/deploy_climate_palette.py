"""Attach the palette to existing standalone report templates without logging source."""
import ftplib,os,io,hashlib,subprocess,tempfile,json,re,posixpath
from pathlib import PurePosixPath
from datetime import datetime,timezone
from uuid import uuid4
ROOT='/meteo-krosno.pl'; STATION='/stacja.meteo-krosno.pl'; THEME=ROOT+'/wp-content/themes/meteo-krosno-modern'; BACKUPS=THEME+'/cache/weather-backups'
STAMP=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8]
LINK=b'<link id="mk-climate-palette" rel="stylesheet" href="https://meteo-krosno.pl/wp-content/themes/meteo-krosno-modern/assets/climate-panels-20261028.css" media="screen">\n'
def read(f,p):
    b=io.BytesIO();f.retrbinary('RETR '+p,b.write);return b.getvalue()
def write(f,p,b):
    f.storbinary('STOR '+p,io.BytesIO(b));assert read(f,p)==b

def publish(f,p,b):
    temp=p+'.palette-'+STAMP;write(f,temp,b);f.rename(temp,p);assert read(f,p)==b
with ftplib.FTP(os.environ['FTP_HOST'],os.environ['FTP_USER'],os.environ['FTP_PASS'],timeout=45) as f:
    protection=read(f,BACKUPS+'/.htaccess');assert b'denied' in protection or b'Deny from all' in protection
    candidates=set();route_files={ROOT+'/index.php'}
    def listing(directory):
        try:return [directory+'/'+PurePosixPath(n).name for n in f.nlst(directory)]
        except ftplib.error_perm:return []
    # Only weather report names are considered as editable templates.
    pattern=re.compile(r'klimat|climate',re.I)
    for directory in (ROOT,STATION,THEME,THEME+'/cache',THEME+'/templates',ROOT+'/wp-content/mu-plugins'):
        for path in listing(directory):
            name=PurePosixPath(path).name
            if directory.endswith('/mu-plugins') and name.endswith('.php') and name!='00-mk-climate-palette.php':route_files.add(path)
            if pattern.search(name):
                if name.endswith(('.php','.html')):candidates.add(path)
                elif '.' not in name:
                    candidates.update(p for p in listing(path) if p.endswith(('.php','.html')))
    # Resolve report file references locally on the runner. No source or fragments
    # of private PHP code are copied to the repository, artifacts or job logs.
    for route in route_files:
        try:code=read(f,route).decode('utf-8','replace')
        except ftplib.error_perm:continue
        if not pattern.search(code):continue
        literals=re.findall(r'''['"]([^'"\n<>]{1,200}\.(?:php|html))['"]''',code)
        for value in literals:
            if not pattern.search(value):continue
            if value.startswith(('http://','https://')):
                from urllib.parse import urlparse
                url=urlparse(value)
                if url.hostname not in ('meteo-krosno.pl','stacja.meteo-krosno.pl'):continue
                value='/'+url.hostname+url.path;candidates.add(value);continue
            for base in (str(PurePosixPath(route).parent),ROOT,ROOT+'/wp-content',THEME):
                path=posixpath.normpath(base+'/'+value.lstrip('/'))
                if path.startswith((ROOT+'/',STATION+'/')) and 'backup' not in path:candidates.add(path)
            if value.startswith((ROOT+'/',STATION+'/')):candidates.add(value)
    staged=[];found={'month':False,'year':False}
    for path in sorted(candidates):
        if 'backup' in path.lower():continue
        try:old=read(f,path)
        except ftplib.error_perm:continue
        month=b'Rekordy i odleg' in old and b'insight' in old
        year=b'Raport Klimatyczny' in old and b'summary-box' in old
        if not (month or year) or b'</head>' not in old:continue
        found['month']|=month;found['year']|=year
        new=old if b'id="mk-climate-palette"' in old else old.replace(b'</head>',LINK+b'</head>',1)
        if path.endswith('.php'):
            with tempfile.NamedTemporaryFile(suffix='.php') as temp:
                temp.write(new);temp.flush();result=subprocess.run(['php','-l',temp.name],capture_output=True)
                assert result.returncode==0,'PHP syntax check failed for report template'
        staged.append((path,old,new))
    assert all(found.values()),'Could not identify both standalone report templates; nothing published'
    assert len(staged)<=10,'Too many report template candidates; nothing published'
    backup=BACKUPS+'/climate-'+STAMP;f.mkd(backup)
    for i,(path,old,new) in enumerate(staged):write(f,backup+'/'+str(i)+'-'+PurePosixPath(path).name,old)
    write(f,backup+'/manifest.json',json.dumps({'files':[{'path':p,'sha256':hashlib.sha256(o).hexdigest()} for p,o,_ in staged]},indent=2).encode())
    changed=[]
    try:
        for path,old,new in staged:
            assert read(f,path)==old,'Report changed during preparation'
            if old==new:continue
            changed.append((path,old));publish(f,path,new)
    except Exception:
        for path,old in reversed(changed):publish(f,path,old)
        raise
    print('CLIMATE_PALETTE_VERIFIED',len(staged),'templates; source not logged')
    print('BACKUP',backup)

