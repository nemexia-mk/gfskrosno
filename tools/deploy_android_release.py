"""Publish signed APK and its dedicated page, without changing station/API/redirect code."""
import ftplib,io,os,json,hashlib,zipfile
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from uuid import uuid4
ROOT='/meteo-krosno.pl';BACKUPS=ROOT+'/wp-content/themes/meteo-krosno-modern/cache/weather-backups'
STAMP=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8]
def read(f,p):
 b=io.BytesIO();f.retrbinary('RETR '+p,b.write);return b.getvalue()
def optional(f,p):
 try:return read(f,p)
 except ftplib.error_perm as e:
  if str(e).startswith('550') and PurePosixPath(p).name not in {PurePosixPath(n).name for n in f.nlst(str(PurePosixPath(p).parent))}:return None
  raise
def write(f,p,b):
 f.storbinary('STOR '+p,io.BytesIO(b));assert read(f,p)==b,'Readback mismatch'
def publish(f,p,b):
 t=p+'.release-'+STAMP;write(f,t,b);f.rename(t,p);assert read(f,p)==b
r=json.loads(Path('public-app/mk-android-release.json').read_text());apk=Path('public-app')/r['filename'];data=apk.read_bytes()
assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
with zipfile.ZipFile(apk) as z:assert z.testzip() is None and 'classes.dex' in z.namelist()
assert r['filename']=='Meteo-Krosno-1.0.0.apk' and r['url']=='/pobierz/'+r['filename']
with ftplib.FTP(os.environ['FTP_HOST'],os.environ['FTP_USER'],os.environ['FTP_PASS'],timeout=45) as f:
 guard=read(f,BACKUPS+'/.htaccess');assert b'denied' in guard or b'Deny from all' in guard
 try:f.mkd(ROOT+'/pobierz')
 except ftplib.error_perm:
  f.cwd(ROOT+'/pobierz');f.cwd('/')
 staged=[]
 files=[(apk,ROOT+r['url'])]
 for name in ['mk-android-release.css','mk-android-release.json','mk-android-release.php']:files.append((Path('public-app')/name,ROOT+'/wp-content/mu-plugins/'+name))
 for local,p in files:
  old=optional(f,p);new=local.read_bytes();assert old is None or old==new,'Release path already occupied; no files changed';staged.append((p,old,new))
 backup=BACKUPS+'/android-release-'+STAMP;f.mkd(backup)
 write(f,backup+'/manifest.json',json.dumps([{'path':p,'previouslyExisted':old is not None} for p,old,new in staged]).encode())
 changed=[]
 try:
  for p,old,new in staged:
   assert optional(f,p)==old,'File changed while preparing release'
   if old==new:continue
   changed.append((p,old));publish(f,p,new)
  # Ensure the first visit reaches WP, which invalidates W3 menu/page caches once.
  cache=ROOT+'/wp-content/cache/page_enhanced/meteo-krosno.pl'
  for sub in ['', '/aplikacja', '/page-sitemap.xml']:
   try:names=f.nlst(cache+sub)
   except ftplib.error_perm:continue
   for item in names:
    name=PurePosixPath(item).name
    if name.startswith('_index') and '.html' in name:
     p=cache+sub+'/'+name;write(f,backup+'/cache-'+(sub.strip('/') or 'home')+'-'+name,read(f,p));f.delete(p)
 except Exception:
  for p,old in reversed(changed):
   if old is None:f.delete(p)
   else:publish(f,p,old)
  raise
 print('Published APK and Android page module; private recovery copy:',backup)
