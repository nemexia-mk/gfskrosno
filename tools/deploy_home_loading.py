"""Publish only the home loading module; preserve SEO, station code and redirects."""
import ftplib,io,os,json,subprocess
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from uuid import uuid4
ROOT='/meteo-krosno.pl/wp-content'
BACKUPS=ROOT+'/themes/meteo-krosno-modern/cache/weather-backups'
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
 t=p+'.loading-'+STAMP;write(f,t,b);f.rename(t,p);assert read(f,p)==b
with ftplib.FTP(os.environ['FTP_HOST'],os.environ['FTP_USER'],os.environ['FTP_PASS'],timeout=45) as f:
 assert b'denied' in read(f,BACKUPS+'/.htaccess') or b'Deny from all' in read(f,BACKUPS+'/.htaccess')
 staged=[]
 # Assets first, PHP entry point last: no half-installed module can activate.
 for name in ['mk-home-loading.css','mk-home-loading.js','mk-home-loading.php']:
  p=ROOT+'/mu-plugins/'+name;old=optional(f,p);new=Path('site/mu-plugins/'+name).read_bytes()
  assert old is None or old==new,'New loading module path already occupied; stop'
  staged.append((p,old,new))
 backup=BACKUPS+'/home-loading-'+STAMP;f.mkd(backup)
 write(f,backup+'/manifest.json',json.dumps([{'path':p,'previouslyExisted':old is not None} for p,old,new in staged]).encode())
 changed=[]
 try:
  for p,old,new in staged:
   assert optional(f,p)==old,'File changed while preparing'
   if old==new:continue
   changed.append((p,old));publish(f,p,new)
  cache=ROOT+'/cache/page_enhanced/meteo-krosno.pl'
  for folder in [cache,cache+'/strona-glowna']:
   try:names=f.nlst(folder)
   except ftplib.error_perm:continue
   for item in names:
    name=PurePosixPath(item).name
    if name.startswith('_index') and '.html' in name:
     p=folder+'/'+name;write(f,backup+'/'+('alias-' if folder!=cache else 'root-')+name,read(f,p));f.delete(p)
 except Exception:
  for p,old in reversed(changed):
   if old is None:f.delete(p)
   else:publish(f,p,old)
  raise
 print('Published home loading module; private recovery manifest:',backup)
