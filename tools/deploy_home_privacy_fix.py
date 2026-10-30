"""Narrow frame-promo and privacy-click repair; no APK/API/redirect changes."""
import ftplib,io,os,json
from pathlib import Path,PurePosixPath
import hashlib
from datetime import datetime,timezone
from uuid import uuid4
ROOT='/meteo-krosno.pl/wp-content';BACKUPS=ROOT+'/themes/meteo-krosno-modern/cache/weather-backups'
STAMP=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8]
def read(f,p):
 b=io.BytesIO();f.retrbinary('RETR '+p,b.write);return b.getvalue()
def write(f,p,b):
 f.storbinary('STOR '+p,io.BytesIO(b));assert read(f,p)==b
def publish(f,p,b):
 t=p+'.ui-'+STAMP;write(f,t,b);f.rename(t,p);assert read(f,p)==b
expected=json.loads(Path('ui-fixes/expected.json').read_text())
files=[('site/publication-assets/consent-20261029.js',ROOT+'/mu-plugins/mk-publication-assets/consent-20261029.js'),('site/mu-plugins/mk-buycoffee.php',ROOT+'/mu-plugins/mk-buycoffee.php'),('public-app/mk-android-release.php',ROOT+'/mu-plugins/mk-android-release.php')]
with ftplib.FTP(os.environ['FTP_HOST'],os.environ['FTP_USER'],os.environ['FTP_PASS'],timeout=45) as f:
 guard=read(f,BACKUPS+'/.htaccess');assert b'denied' in guard or b'Deny from all' in guard
 staged=[]
 for local,remote in files:
  old=read(f,remote);new=Path(local).read_bytes();assert old==new or hashlib.sha256(old).hexdigest()==expected[local],'Source changed since review; stop';staged.append((remote,old,new))
 backup=BACKUPS+'/home-privacy-'+STAMP;f.mkd(backup)
 for i,(p,old,new) in enumerate(staged):write(f,backup+'/'+str(i)+'-'+PurePosixPath(p).name,old)
 write(f,backup+'/manifest.json',json.dumps([{'path':p,'backup':str(i)+'-'+PurePosixPath(p).name} for i,(p,old,new) in enumerate(staged)]).encode())
 changed=[]
 try:
  for p,old,new in staged:
   assert read(f,p)==old
   if old==new:continue
   changed.append((p,old));publish(f,p,new)
  cache=ROOT+'/cache/page_enhanced/meteo-krosno.pl'
  for sub in ['', '/aplikacja']:
   try:items=f.nlst(cache+sub)
   except ftplib.error_perm:continue
   for item in items:
    n=PurePosixPath(item).name
    if n.startswith('_index') and '.html' in n:f.delete(cache+sub+'/'+n)
 except Exception:
  for p,old in reversed(changed):publish(f,p,old)
  raise
 print('Frame promotions and privacy clicks repaired; recovery copy:',backup)
