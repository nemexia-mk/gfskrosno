"""Publish reviewed assets and HTTPS rules; private source stays on the server/backups.
No private production source or configuration is emitted to logs or artifacts.
"""
import ftplib,io,os,json,re
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from urllib.request import urlopen
from uuid import uuid4
ROOT='/meteo-krosno.pl';WP=ROOT+'/wp-content';MU=WP+'/mu-plugins';BACKUPS=WP+'/themes/meteo-krosno-modern/cache/weather-backups'
STAMP=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8]
HTTPS_RULES=b'''# BEGIN MK HTTPS 20261029
<IfModule mod_rewrite.c>
RewriteEngine On
RewriteCond %{HTTP_HOST} ^(www\\.)?meteo-krosno\\.pl(:[0-9]+)?$ [NC]
RewriteCond %{HTTPS} !=on
RewriteCond %{HTTP:X-Forwarded-Proto} !^https(?:,|$) [NC]
RewriteRule ^ https://meteo-krosno.pl%{REQUEST_URI} [R=301,L,NE]
</IfModule>
# END MK HTTPS 20261029
'''
def read(f,p):
 b=io.BytesIO();f.retrbinary('RETR '+p,b.write);return b.getvalue()
def optional(f,p):
 try:return read(f,p)
 except ftplib.error_perm as e:
  if str(e).startswith('550') and PurePosixPath(p).name not in {PurePosixPath(x).name for x in f.nlst(str(PurePosixPath(p).parent))}:return None
  raise
def write(f,p,b):
 f.storbinary('STOR '+p,io.BytesIO(b));assert read(f,p)==b,'Readback failed'
def publish(f,p,b):
 tmp=p+'.publish-'+STAMP;write(f,tmp,b);f.rename(tmp,p);assert read(f,p)==b

def main():
 import hashlib
 with ftplib.FTP(os.environ['FTP_HOST'],os.environ['FTP_USER'],os.environ['FTP_PASS'],timeout=45) as f:
  protection=read(f,BACKUPS+'/.htaccess');assert b'denied' in protection or b'Deny from all' in protection
  assetdir=MU+'/mk-publication-assets'
  try:f.mkd(assetdir)
  except ftplib.error_perm as e:
   if not str(e).startswith('550'):raise
  staged=[]
  files=[('site/publication-assets/consent-20261029.js',assetdir+'/consent-20261029.js'),('site/publication-assets/consent-20261029.css',assetdir+'/consent-20261029.css'),('site/mu-plugins/mk-publication-policy.php',MU+'/mk-publication-policy.php'),('site/mu-plugins/mk-public-seo.php',MU+'/mk-public-seo.php'),('site/mu-plugins/mk-publication.php',MU+'/mk-publication.php')]
  for local,remote in files:
   old=optional(f,remote);new=Path(local).read_bytes()
   if remote.endswith('/mk-public-seo.php'):assert old==new or (old is not None and hashlib.sha256(old).hexdigest()=='bcefc81e1ecb15ba454e53fddb555b91886a36f761949d209adad7e2dccd1e02'),'Live SEO changed since review'
   else:assert old is None or old==new,'Publication path contains another version'
   staged.append((remote,old,new))
  ht=ROOT+'/.htaccess';old=optional(f,ht);current=old or b''
  if b'# BEGIN MK HTTPS 20261029' in current:assert HTTPS_RULES in current;new=current
  else:new=HTTPS_RULES+b'\n'+current
  staged.append((ht,old,new))
  backup=BACKUPS+'/publication-'+STAMP;f.mkd(backup)
  manifest=[]
  for i,(p,old,new) in enumerate(staged):
   name=str(i)+'-'+PurePosixPath(p).name
   if old is not None:write(f,backup+'/'+name,old)
   manifest.append({'path':p,'backup':name if old is not None else None})
  write(f,backup+'/manifest.json',json.dumps(manifest,indent=2).encode())
  changed=[]
  try:
   for p,old,new in staged:
    assert optional(f,p)==old,'File changed during staging'
    if old==new:continue
    changed.append((p,old));publish(f,p,new);print('PUBLISHED',PurePosixPath(p).name,flush=True)
  except Exception:
   for p,old in reversed(changed):
    if old is None:f.delete(p)
    else:publish(f,p,old)
   raise
  # A noncached API call applies the backed-up WordPress migrations and flushes W3.
  with urlopen('https://meteo-krosno.pl/?mk_api=publication-status&activate='+STAMP,timeout=70) as r:status=json.load(r)
  assert status['migrationComplete'] and status['testRemoved'] and not status['oldImageReferencesRemaining'],status
  assert status['privacyPublished'] or status['privacyDraftReady'],status
  assert status['cacheFlushed'],'WordPress cache flush did not run; stop for cache review'
  print('CONTENT_CHECKS',json.dumps(status),flush=True)
  print('PRIVATE_BACKUP_CREATED',flush=True)
if __name__=='__main__':main()
