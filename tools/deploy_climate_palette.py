"""Attach the reviewed palette to the standalone climate PHP entry points."""
import ftplib,os,io,hashlib,subprocess,tempfile,json
from pathlib import Path
from datetime import datetime,timezone
from uuid import uuid4
ROOT='/meteo-krosno.pl'; BACKUPS=ROOT+'/wp-content/themes/meteo-krosno-modern/cache/weather-backups'
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
    staged=[]
    for slug in ['klimat-miesieczny','klimat-roczny']:
        path=ROOT+'/'+slug+'/index.php';old=read(f,path)
        if b'id="mk-climate-palette"' in old:new=old
        else:
            assert b'</head>' in old,'Expected standalone HTML head missing: '+path
            new=old.replace(b'</head>',LINK+b'</head>',1)
        with tempfile.NamedTemporaryFile(suffix='.php') as temp:
            temp.write(new);temp.flush();result=subprocess.run(['php','-l',temp.name],capture_output=True,text=True)
            assert result.returncode==0,'PHP syntax check failed: '+path
        staged.append((slug,path,old,new))
    backup=BACKUPS+'/climate-'+STAMP;f.mkd(backup)
    for slug,path,old,new in staged:write(f,backup+'/'+slug+'.php',old)
    write(f,backup+'/manifest.json',json.dumps({'files':[{'path':p,'sha256':hashlib.sha256(o).hexdigest()} for _,p,o,_ in staged]},indent=2).encode())
    changed=[]
    try:
        for slug,path,old,new in staged:
            assert read(f,path)==old,'Source changed during preparation'
            if old==new:continue
            changed.append((path,old));publish(f,path,new);print('CLIMATE_PALETTE_VERIFIED',slug)
    except Exception:
        for path,old in reversed(changed):publish(f,path,old)
        raise
    print('BACKUP',backup)
