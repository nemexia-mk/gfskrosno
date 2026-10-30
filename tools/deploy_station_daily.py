"""Narrow station CSV repair; private receiver code never leaves server/private backups/logs."""
import ftplib,io,os,re,hashlib,json,subprocess,tempfile
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from uuid import uuid4
ROOT='/meteo-krosno.pl'; PLUGIN=ROOT+'/wp-content/plugins/mk-station-dashboards'
BACKUPS=ROOT+'/wp-content/themes/meteo-krosno-modern/cache/weather-backups'
STAMP=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:6]
def read(f,p):
 b=io.BytesIO();f.retrbinary('RETR '+p,b.write);return b.getvalue()
def write(f,p,b):
 f.storbinary('STOR '+p,io.BytesIO(b));assert read(f,p)==b,'Readback failed'
def put(f,p,b):
 t=p+'.daily-'+STAMP;write(f,t,b);f.rename(t,p);assert read(f,p)==b
def lint(b):
 with tempfile.NamedTemporaryFile(suffix='.php') as t:
  t.write(b);t.flush();r=subprocess.run(['php','-l',t.name],capture_output=True)
  assert r.returncode==0,'PHP syntax check failed; private source not logged'
def patch_reader(old):
 s=old.decode('utf-8');tag='// MK daily reader v1'
 if tag in s:return old
 needle="function stamp($s){if(!is_string($s))return null;"
 assert needle in s,'Timestamp parser differs; stop'
 s=s.replace(needle,"function stamp($s){if(!is_string($s)||!preg_match('/^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$/D',$s))return null;",1)
 needle="$files=$months;"
 assert needle in s,'Monthly reader differs; stop'
 block="""$files=$months;
// MK daily reader v1: archive parsing is cached; today's small CSV supplies new points.
require_once __DIR__.'/daily-cache.php';
$mksDailyBase='dzis_'.$c['suffix'].'.csv';
$mksDailyPath=remote_file($c['dir'].'/'.$mksDailyBase,30);
$mksReaderCache=\\mksd_reader_cache($id,$range,$dateRequest??'', $mksDailyPath,$now,$start);
if($mksReaderCache){$points=$mksReaderCache['points'];$monthTemps=$mksReaderCache['monthTemps'];$sources=$mksReaderCache['sources'];$files=[];}
if($mksDailyPath)$files[]=$mksDailyBase;
"""
 s=s.replace(needle,block,1)
 needle='ksort($points,SORT_NUMERIC);'
 assert needle in s,'Record sorting differs; stop'
 s=s.replace(needle,"if(!$mksReaderCache)\\mksd_store_reader($id,$range,$dateRequest??'', $now,$start,$points,$monthTemps,$sources);\n$sources=array_values(array_unique($sources));\n"+needle,1)
 return s.encode()
def patch_receiver(old,suffix):
 s=old.decode('utf-8');tag='// MK daily CSV hook v1'
 if tag in s:return old
 # Register a sidecar mirror only; the existing receiver and its auth are untouched.
 assert 'PASSKEY' in s and s.lstrip('\ufeff').startswith('<?php'),'Not an expected station receiver'
 pos=s.index('<?php')+len('<?php')
 strict=re.match(r'\s*declare\s*\(\s*strict_types\s*=\s*1\s*\)\s*;',s[pos:])
 if strict:pos+=strict.end()
 root="__DIR__" if suffix=='root' else "dirname(__DIR__)"
 hook="\n// MK daily CSV hook v1\nrequire_once __DIR__.'/mk-daily-csv.php';\nmksd_register_monthly_mirror("+root+", 'turaszowka');\nmksd_register_monthly_mirror("+root+".'/lesniowka', 'lesniowka');\n"
 s=s[:pos]+hook+s[pos:];return s.encode()

def main():
 module=Path('station-daily/daily-cache.php').read_bytes();lint(module)
 with ftplib.FTP(os.environ['FTP_HOST'],os.environ['FTP_USER'],os.environ['FTP_PASS'],timeout=45) as f:
  assert b'Deny from all' in read(f,BACKUPS+'/.htaccess') or b'denied' in read(f,BACKUPS+'/.htaccess')
  staged=[]
  def stage(p,transform=None,new=None):
   try:old=read(f,p)
   except ftplib.error_perm as e:
    if not str(e).startswith('550') or PurePosixPath(p).name in [PurePosixPath(x).name for x in f.nlst(str(PurePosixPath(p).parent))]:raise
    old=None
   b=transform(old) if transform else new
   if p.endswith('.php'):lint(b)
   if old!=b:staged.append((p,old,b))
  # Roll back only our named HTTP redirect block, preserving every other rule/PHP handler.
  def undo_redirect(old):
   if b'# BEGIN MK HTTPS 20261029' not in old:return old
   m=re.search(rb'# BEGIN MK HTTPS 20261029\r?\n.*?# END MK HTTPS 20261029\r?\n(?:\r?\n)?',old,re.S)
   assert m and b'RewriteRule ^ https://meteo-krosno.pl%{REQUEST_URI} [R=301,L,NE]' in m.group(),'Unknown redirect block'
   return old[:m.start()]+old[m.end():]
  stage(ROOT+'/.htaccess',transform=undo_redirect)
  stage(PLUGIN+'/daily-cache.php',new=module)
  stage(PLUGIN+'/data.php',transform=patch_reader)
  receiver_count=0
  for sub in ['', '/lesniowka']:
   directory='/stacja.meteo-krosno.pl/pwsWD/ecowitt'+sub
   stage(directory+'/mk-daily-csv.php',new=module)
   # Discover existing receiver entry points without printing private source.
   for name in f.nlst(directory):
    base=PurePosixPath(name).name
    if not base.endswith('.php') or re.search(r'(test|backup|old|copy|bak|mk-daily)',base,re.I):continue
    path=directory+'/'+base;source=read(f,path)
    if b'PASSKEY' not in source or b'$_POST' not in source:continue
    stage(path,transform=lambda b,kind='child' if sub else 'root':patch_receiver(b,kind))
    receiver_count+=1
  assert receiver_count, 'No station receiver identified; no files changed'
  backup=BACKUPS+'/station-daily-'+STAMP;f.mkd(backup)
  for i,(p,old,new) in enumerate(staged):
   if old is not None:write(f,backup+'/'+str(i)+'-'+PurePosixPath(p).name,old)
  write(f,backup+'/manifest.json',json.dumps([{'path':p,'backup':str(i)+'-'+PurePosixPath(p).name if old is not None else None} for i,(p,old,new) in enumerate(staged)]).encode())
  changed=[]
  try:
   for p,old,new in staged:
    if old is not None:assert read(f,p)==old,'Production changed during staging'
    put(f,p,new);changed.append((p,old));print('APPLIED',PurePosixPath(p).name,flush=True)
  except Exception:
   for p,old in reversed(changed):
    if old is None:f.delete(p)
    else:put(f,p,old)
   raise
  print('STATION_DAILY_DEPLOYED; original receivers and parser backed up privately',flush=True)
if __name__=='__main__':main()
