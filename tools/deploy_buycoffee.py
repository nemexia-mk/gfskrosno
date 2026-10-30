"""Publish only the BuyCoffee section; preserve SEO, station code and redirects."""
import ftplib,io,os,json,subprocess,hashlib
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
 qr=None
 for candidate in ['/meteo-krosno/QR_buycoffee.png','/meteo-krosno.pl/QR_buycoffee.png','/meteo-krosno.pl/meteo-krosno/QR_buycoffee.png','/stacja.meteo-krosno.pl/QR_buycoffee.png']:
  try:qr=read(f,candidate);break
  except ftplib.error_perm as e:
   if not str(e).startswith('550'):raise
 if qr is None:
  def locate(folder,depth=0):
   if depth>3:return None
   try:items=list(f.mlsd(folder))
   except ftplib.error_perm:return None
   for name,facts in items:
    if name in ['.','..'] or '/' in name:continue
    path=folder.rstrip('/')+'/'+name
    if name.lower()=='qr_buycoffee.png' and facts.get('type')=='file':return read(f,path)
    if facts.get('type')=='dir' and name not in ['cache','weather-backups','node_modules','.git']:
     found=locate(path,depth+1)
     if found:return found
   return None
  for folder in ['/meteo-krosno','/meteo-krosno.pl']:
   qr=locate(folder)
   if qr:break
 assert qr and len(qr)<2000000 and qr.startswith(b'\x89PNG\r\n\x1a\n'),'Specified QR_buycoffee.png not found or not a valid PNG; no files changed'
 artifact=Path('coffee-artifact');artifact.mkdir(exist_ok=True);(artifact/'QR_buycoffee.png').write_bytes(qr)
 p=ROOT+'/themes/meteo-krosno-modern/assets/qr-buycoffee.png';old=optional(f,p)
 assert old is None or old==qr,'QR asset path already occupied'
 staged.append((p,old,qr))
 # Assets first, PHP entry point last: no half-installed module can activate.
 for name in ['mk-buycoffee.php']:
  p=ROOT+'/mu-plugins/'+name;old=optional(f,p);new=Path('site/mu-plugins/'+name).read_bytes()
  assert old is None or old==new or hashlib.sha256(old).hexdigest()=='e6235ead9d92260bb8d919733eca1c1d3fd3380f377c6fc60b64cc40500a64ac','BuyCoffee module changed since review; stop'
  staged.append((p,old,new))
 backup=BACKUPS+'/buycoffee-'+STAMP;f.mkd(backup)
 for i,(p,old,new) in enumerate(staged):
  if old is not None:write(f,backup+'/'+str(i)+'-'+PurePosixPath(p).name,old)
 write(f,backup+'/manifest.json',json.dumps([{'path':p,'previouslyExisted':old is not None} for p,old,new in staged]).encode())
 changed=[]
 try:
  for p,old,new in staged:
   assert optional(f,p)==old,'File changed while preparing'
   if old==new:continue
   changed.append((p,old));publish(f,p,new)
  cache=ROOT+'/cache/page_enhanced/meteo-krosno.pl'
  for folder in [cache,cache+'/strona-glowna',cache+'/o-nas']:
   try:names=f.nlst(folder)
   except ftplib.error_perm:continue
   for item in names:
    name=PurePosixPath(item).name
    if name.startswith('_index') and '.html' in name:
     p=folder+'/'+name;write(f,backup+'/'+(PurePosixPath(folder).name+'-')+name,read(f,p));f.delete(p)
 except Exception:
  for p,old in reversed(changed):
   if old is None:f.delete(p)
   else:publish(f,p,old)
  raise
 print('Published BuyCoffee section; private recovery manifest:',backup)
