#!/usr/bin/env python3
"""Konwekcja z tego samego runu co gfs-tab.csv: ecCodes i profile MetPy.
Stare kolumny pozostają zgodne. Nieuzasadnione procenty i liczba błysków są puste.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from time import monotonic, sleep
import numpy as np
import pandas as pd
import gfs_krosno as gfs

LOG = logging.getLogger('gfs_conv')
OUTPUT_DIR = 'gfs_krosno_conv'
LEGACY_COLUMNS = ['Czas','T+','T2M [°C]','CAPE [J/kg]','MUCAPE [J/kg]',
 'DCAPE [J/kg]','CIN [J/kg]','DLS 0-6km [m/s]','DLS 0-1km [m/s]',
 'SRH 0-3km','SRH 0-1km approx','BRN','STP (stary)','STP (pełny MetPy)',
 'SHIP','EHI','PWAT [mm]','LCL [m]','Opad [mm/h]','Kumulacyjny opad [mm]',
 'Prob Burzy [%]','Prob SC [%]','Prob Tornado [%]','Prob Grad [%]',
 'Prob DB (Wiatr) [%]','Prob Derecho [%]','Prob Ulewa [%]','Storm Mode',
 'Rotacja','Prędkość Burzy [km/h]','Grad [cm]','DCP (Derecho)','Błyski [1/min]',
 'Halny','Orografia','Poziom (ESTOFEX)','Widzialność Tatr']


def build_dataframe(legacy, extras):
    by_time = {r['date']:r for r in extras}
    rows = []; cumulative = 0.; cumulative_complete = True
    for _,r in legacy.iterrows():
        date = r['Czas'].strftime('%Y-%m-%d %H:%M:%S'); e = by_time.get(date,{})
        rain, duration = r['RRR [mm]'], e.get('stepHours',0)
        if r['T+ (h)'] > 0:
            if gfs.finite(rain): cumulative += rain
            else: cumulative_complete = False
        row = dict.fromkeys(LEGACY_COLUMNS,np.nan)
        row.update({'Czas':date,'T+':int(r['T+ (h)']),'T2M [°C]':r['T2M [°C]'],
                    'CAPE [J/kg]':r['CAPE [J/kg]'],'CIN [J/kg]':e.get('cin'),
                    'LCL [m]':r['LCL_m'],'Opad [mm/h]':rain/duration if duration else np.nan,
                    'Kumulacyjny opad [mm]':cumulative if cumulative_complete else np.nan})
        for key,column in {'muCape':'MUCAPE [J/kg]','dCape':'DCAPE [J/kg]',
                           'shear06':'DLS 0-6km [m/s]','shear01':'DLS 0-1km [m/s]',
                           'srh03':'SRH 0-3km','srh01':'SRH 0-1km approx',
                           'pwat':'PWAT [mm]','stormSpeed':'Prędkość Burzy [km/h]'}.items(): row[column] = e.get(key)
        row.update({'Potencjał burzowy [0-4]':e.get('stormLevel'),
                    'Opis potencjału burzowego':e.get('stormLabel','Brak danych'),
                    'Rodzaj opadu':e.get('type',r['Rodzaj opadu']),
                    'Źródło rodzaju opadu':e.get('precipSource'),
                    'Opad konwekcyjny [mm/krok]':e.get('convRain'),
                    'Odbiciowość modelowa [dBZ]':e.get('reflectivity'),
                    'Wilgotność [%]':e.get('humidity'),'Temperatura mokra [°C]':e.get('wetBulb'),
                    'K Index [°C]':e.get('kIndex'),'Total Totals [°C]':e.get('totalTotals'),
                    'Izoterma 0°C [m AGL]':e.get('freezing'),'Wiatr 100m [m/s]':e.get('wind100'),
                    'Promieniowanie [W/m²]':e.get('solar')})
        rows.append(row)
    return pd.DataFrame(rows)


def save_outputs(df,run_date,run_hour):
    if df.empty: raise ValueError('Brak danych konwekcyjnych')
    root = Path(OUTPUT_DIR); root.mkdir(parents=True,exist_ok=True)
    csv_path = root/'gfs-conv.csv'; df.to_csv(csv_path,index=False,encoding='utf-8',na_rep='')
    xlsx_path = root/f'krosno_conv_{run_date}_{run_hour}z.xlsx'
    with pd.ExcelWriter(xlsx_path,engine='xlsxwriter') as writer:
        df.to_excel(writer,index=False,sheet_name='Konwekcja')
        ws = writer.sheets['Konwekcja']; ws.freeze_panes(1,2)
        ws.autofilter(0,0,len(df),len(df.columns)-1); ws.set_column(0,len(df.columns)-1,20)
        pd.DataFrame({'Metodyka':[
            'Czas UTC. Potencjał burzowy 0–4: autorska kategoria środowiska GFS 0,25°.',
            'Nie jest procentem prawdopodobieństwa, liczbą błysków ani ostrzeżeniem.',
            'MUCAPE i DCAPE liczone z profilu temperatury i wilgotności przez MetPy.',
            'Uskoki i SRH dotyczą warstw AGL, nie zamienników 925/500 hPa.',
            'Stare kolumny procentów, rozmiaru gradu i ESTOFEX pozostają puste.',
            'Brak danych pozostaje pusty; rzeczywiste zera są zachowane.'
        ]}).to_excel(writer,index=False,sheet_name='Metodyka')
        writer.sheets['Metodyka'].set_column(0,0,110)
    return [str(csv_path),str(xlsx_path)]


def upload_to_ftp(files,run_date,run_hour):
    ftp = gfs.ftp_connection()
    if ftp is None: raise ValueError('Brak konfiguracji FTP')
    try:
        for path in files:
            if path.endswith('.csv'):
                gfs.ensure_ftp_directory(ftp,gfs.FTP_BASE_DIR+'/archiv_conv')
                name = f'gfs_conv_tab_{run_date[:4]}_{run_date[4:6]}_{run_date[6:8]}_{run_hour}.csv'
                gfs.ftp_store(ftp,path,name)
            ftp.cwd(gfs.FTP_BASE_DIR); gfs.ftp_store(ftp,path,Path(path).name)
        LOG.info('CSV i XLSX konwekcji opublikowane dla %s%sZ',run_date,run_hour)
    finally:
        try: ftp.quit()
        except Exception: ftp.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',help='Run YYYYMMDDHH; wspólny z gfs_krosno.py')
    parser.add_argument('--hours',help='Terminy testowe oddzielone przecinkiem')
    parser.add_argument('--no-ftp',action='store_true')
    parser.add_argument('--reuse-gribs',action='store_true',help='Użyj wyników generatora bez ponownego pobierania')
    parser.add_argument('--max-wait',type=float,default=30)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.run:
        from datetime import datetime
        try:
            run = datetime.strptime(args.run,'%Y%m%d%H')
            if len(args.run)!=10 or run.hour not in (0,6,12,18): raise ValueError()
        except ValueError: parser.error('Nieprawidłowy run YYYYMMDDHH')
        rd,rh = run.strftime('%Y%m%d'),run.strftime('%H')
    else: rd,rh = gfs.select_run()
    try:
        hours = sorted(set(map(int,args.hours.split(',')))) if args.hours else gfs.FORECAST_HOURS
        if not hours or any(h not in gfs.FORECAST_HOURS for h in hours) or args.max_wait<=0: raise ValueError()
    except ValueError: parser.error('Nieprawidłowe terminy lub limit czasu')
    try:
        sidecar = Path(gfs.OUTPUT_DIR)/f'gfs_extra_{rd}_{rh}.json'
        legacy_path = Path(gfs.OUTPUT_DIR)/f'krosno_gfs_{rd}_{rh}z.csv'
        if args.reuse_gribs and sidecar.is_file() and legacy_path.is_file():
            extra = json.loads(sidecar.read_text(encoding='utf-8'))
            if extra['run'] != gfs.run_datetime(rd,rh).isoformat()+'Z': raise ValueError('Niezgodny run rozszerzeń')
            df = pd.read_csv(legacy_path,parse_dates=['Czas'])
            extras = [r for r in extra['rows'] if r['lead'] in hours]
            df = df[df['T+ (h)'].isin(hours)].copy()
        else:
            deadline = monotonic()+args.max_wait*60
            while True:
                _,missing = gfs.download_missing_gribs_parallel(hours,rd,rh,deadline)
                if not missing: break
                if args.reuse_gribs or monotonic()>=deadline: raise ValueError(f'Brak plików: {missing}')
                sleep(min(30,max(0,deadline-monotonic())))
            df,_ = gfs.process_local_gribs(hours,rd,rh,cache={})
            extras = df.attrs.get('extra_rows',[])
        if set(df['T+ (h)']) != set(hours) or {r['lead'] for r in extras} != set(hours):
            raise ValueError('Niepełny zestaw terminów konwekcji')
        if not any(gfs.finite(r.get('cin')) and gfs.finite(r.get('pwat')) and gfs.finite(r.get('shear06')) for r in extras):
            raise ValueError('Brak podstawowych pól diagnostyki konwekcji')
        files = save_outputs(build_dataframe(df,extras),rd,rh)
        if not args.no_ftp: upload_to_ftp(files,rd,rh)
        LOG.info('Konwekcja: %s/%s terminów; run %s%sZ',len(df),len(hours),rd,rh)
        return 0
    except Exception:
        LOG.exception('Analiza konwekcji nie powiodła się')
        return 1


if __name__=='__main__': sys.exit(main())
