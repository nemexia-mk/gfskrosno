"""Wdrożenie plików pogody: zweryfikowana kopia poza katalogami WWW i kontrola SHA256."""
import argparse
import hashlib
import io
import json
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4
import ftplib
import os

STATION='/stacja.meteo-krosno.pl'
ASSETS='/meteo-krosno.pl/wp-content/themes/meteo-krosno-modern/assets'
STATE=Path('.weather-deployment.json')


def read(ftp,path,optional=False):
    out=io.BytesIO()
    try: ftp.retrbinary('RETR '+path,out.write)
    except ftplib.error_perm as exc:
        if optional and str(exc).startswith('550'):
            names={Path(n).name for n in ftp.nlst(str(Path(path).parent))}
            if Path(path).name not in names: return None
        raise
    return out.getvalue()


def put(ftp,path,content):
    ftp.storbinary('STOR '+path,io.BytesIO(content))
    if read(ftp,path)!=content: raise ValueError('Nieudana weryfikacja zapisu '+path)


def replace(ftp,path,content):
    temporary=path+'.'+uuid4().hex+'.upload'
    put(ftp,temporary,content)
    ftp.rename(temporary,path)
    if read(ftp,path)!=content: raise ValueError('Nieudana weryfikacja publikacji '+path)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','site','verify']);args=p.parse_args()
    with ftplib.FTP(os.environ['FTP_HOST'],os.environ['FTP_USER'],os.environ['FTP_PASS'],timeout=45) as ftp:
        if args.stage=='prepare':
            root='/weather-backups'
            try: ftp.mkd(root)
            except ftplib.error_perm: ftp.cwd(root)
            backup=root+'/'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8]
            ftp.mkd(backup)
            paths=[STATION+'/'+n for n in ('gfs_krosno.py','gfs_conv.py','gfs-tab.csv','gfs-conv.csv','gfs-extra.json','gfs-extra.csv')]+[ASSETS+'/app.js',ASSETS+'/gfs-extra.json']
            manifest={'backup':backup,'files':[]}
            for i,path in enumerate(paths):
                content=read(ftp,path,optional=path!=ASSETS+'/app.js')
                record={'path':path,'backup':None,'sha256':None}
                if content is not None:
                    target=backup+'/'+str(i)+'-'+Path(path).name;put(ftp,target,content)
                    record.update(backup=target,sha256=hashlib.sha256(content).hexdigest())
                manifest['files'].append(record)
            # Oryginały z GitHuba zachowujemy również, gdy serwer nie miał kopii .py.
            for name in ('gfs_krosno.py','gfs_conv.py'):
                put(ftp,backup+'/github-original-'+name,Path('deployment/baseline/'+name).read_bytes())
            put(ftp,backup+'/manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2).encode())
            STATE.write_text(json.dumps(manifest))
            print('BACKUP_VERIFIED',backup)
            for name in ('gfs_krosno.py','gfs_conv.py'):
                replace(ftp,STATION+'/'+name,Path(name).read_bytes());print('SCRIPT_VERIFIED',name)
        elif args.stage=='site':
            if not STATE.is_file(): raise ValueError('Brak potwierdzonej kopii')
            # Ochrona zmian dokonanych przez użytkownika w trakcie testu.
            manifest=json.loads(STATE.read_text())
            record=next(r for r in manifest['files'] if r['path']==ASSETS+'/app.js')
            if hashlib.sha256(read(ftp,ASSETS+'/app.js')).hexdigest()!=record['sha256']:
                raise ValueError('Kod strony zmienił się od wykonania kopii')
            replace(ftp,ASSETS+'/app.js',Path('site/app.js').read_bytes());print('SITE_VERIFIED',ASSETS+'/app.js')
        else:
            for name in ('gfs_krosno.py','gfs_conv.py'):
                assert read(ftp,STATION+'/'+name)==Path(name).read_bytes(),name
            assert read(ftp,ASSETS+'/app.js')==Path('site/app.js').read_bytes()
            from gfs_krosno import LEGACY_COLUMNS as SCHEMA
            import csv
            header=next(csv.reader(io.StringIO(read(ftp,STATION+'/gfs-tab.csv').decode('utf-8-sig'))))
            assert header==SCHEMA,'Zmieniony schemat CSV'
            extra=json.loads(read(ftp,ASSETS+'/gfs-extra.json'))
            assert len(extra['rows'])==209,'Niepełna prognoza'
            run=datetime.fromisoformat(extra['run'].replace('Z','+00:00'))
            expected=Path('gfs_krosno_full')/('krosno_gfs_'+run.strftime('%Y%m%d_%H')+'z.csv')
            assert read(ftp,STATION+'/gfs-tab.csv')==expected.read_bytes(),'Niezgodny CSV GFS'
            assert read(ftp,STATION+'/gfs-conv.csv')==Path('gfs_krosno_conv/gfs-conv.csv').read_bytes(),'Niezgodny CSV konwekcji'
            print('DEPLOYMENT_VERIFIED',extra['run'],len(extra['rows']),'terminów; schemat CSV zachowany')


if __name__=='__main__':main()
