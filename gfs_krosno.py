#!/usr/bin/env python3
"""GFS Krosno: GRIB -> CSV/XLSX/meteorogram, wszystkie daty w UTC.

RRR [mm] to opad w pojedynczym kroku (1 h do T+120, dalej 3 h).
APCP jest rozpakowywane według startStep/endStep; PRATE chwilowe NIE
jest zamieniane na sumę. Awaryjnie używane jest tylko PRATE uśrednione
po przedziale, dla którego można odtworzyć tę samą sumę.
SNOW [cm] oznacza grubość pokrywy SNOD, nie WEASD i nie świeży śnieg.
Zależności: requests numpy pandas matplotlib xlsxwriter python-dotenv eccodes.
Pakiet eccodes jest również zależnością używanego dotąd cfgrib.
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta, time, timezone
from time import sleep, monotonic
from concurrent.futures import ThreadPoolExecutor, as_completed
from ftplib import FTP, error_perm

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from dotenv import load_dotenv
from eccodes import codes_grib_new_from_file, codes_get, codes_release, codes_grib_find_nearest

LOG = logging.getLogger('gfs_krosno')
OUTPUT_DIR = 'gfs_krosno_full'
TOP_LAT, BOTTOM_LAT, LEFT_LON, RIGHT_LON = 50.0, 49.4, 21.3, 22.01
KROSNO_LAT, KROSNO_LON = 49.69, 21.77
BASE_URL = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl'
HEADERS = {'User-Agent': 'Meteo-Krosno-GFS/2.0'}
FORECAST_HOURS = list(range(121)) + list(range(123, 385, 3))
RETRY_INTERVAL_SECONDS = 2 * 60
MAX_TOTAL_WAIT_MINUTES = 90
MAX_WORKERS = 4
REQUEST_TIMEOUT_SECONDS = 90
PRECIP_ALLOW_AVERAGED_PRATE = True
# False oznacza wyłącznie rodzaj opadu wskazany przez GFS; bez flag -> brak danych.
PRECIP_TYPE_TEMPERATURE_FALLBACK = True
# Nadpisywanie gfs-tab.csv przez aktualny run pozostaje celowe.
FTP_BASE_DIR = '/stacja.meteo-krosno.pl'
FTP_ARCH_DIR = FTP_BASE_DIR + '/archiv'

PREC_TYPE_TO_COLOR = {'Deszcz':'#0FB00F','Śnieg':'#ADD8E6',
                     'Deszcz ze śniegiem':'#00FFBB','Deszcz marznący':'#FFA500',
                     'Krupy lodowe':'#D7BDE2'}

def select_run(now=None):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    current = now.time().replace(tzinfo=None)
    if time(3, 30) <= current < time(9, 35):
        hour = '00'
    elif time(9, 35) <= current < time(15, 30):
        hour = '06'
    elif time(15, 30) <= current < time(21, 30):
        hour = '12'
    else:
        hour = '18'
        if current < time(3, 30):
            now -= timedelta(days=1)
    return now.strftime('%Y%m%d'), hour


def run_datetime(run_date, run_hour):
    # Datetime bez tzinfo jest w tym generatorze zawsze UTC (zgodność z Excel/PHP).
    return datetime.strptime(run_date + run_hour, '%Y%m%d%H')


def local_grib(run_date, run_hour, fh):
    return Path(OUTPUT_DIR) / f'krosno_{run_date}_{run_hour}z_f{fh:03d}.grib2'


def build_url(file_name, run_date, run_hour):
    return BASE_URL, build_params(file_name, run_date, run_hour)


def build_params(file_name, run_date, run_hour):
    params = {'file':file_name, 'dir':f'/gfs.{run_date}/{run_hour}/atmos',
              'subregion':'on','toplat':TOP_LAT,'bottomlat':BOTTOM_LAT,
              'leftlon':LEFT_LON,'rightlon':RIGHT_LON}
    for level in ('2_m_above_ground','10_m_above_ground','850_mb','surface',
                  'mean_sea_level','low_cloud_layer','middle_cloud_layer',
                  'high_cloud_layer','entire_atmosphere',
                  'entire_atmosphere_(considered_as_a_single_layer)'):
        params['lev_' + level] = 'on'
    for variable in ('TMP','DPT','TCDC','LCDC','MCDC','HCDC','PRATE','APCP',
                     'GUST','UGRD','VGRD','PRMSL','CAPE','LFTX','SNOD',
                     'VIS','CRAIN','CSNOW','CFRZR','CICEP'):
        params['var_' + variable] = 'on'
    return params


def finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def grib_get(handle, key, default=None):
    try:
        return codes_get(handle, key)
    except Exception:
        return default


def step_hours(value, unit):
    factors = {0:1/60,1:1,2:24,10:3,11:6,12:12,13:1/3600,14:0.25,15:0.5,
               'm':1/60,'h':1,'d':24,'s':1/3600}
    if unit not in factors or not finite(value):
        return None
    return float(value) * factors[unit]


def identify_field(short, level_type, level, category, number):
    # Nazwa wraz z poziomem i statystyką: bez niejednoznacznych fallbacków.
    if level_type == 'heightAboveGround' and level == 2:
        if short in ('2t','t2m','tmp2m','t','tmp'): return 't2m'
        if short in ('2d','d2m','dpt','dew2m'): return 'd2m'
    if level_type == 'heightAboveGround' and level == 10:
        if short in ('10u','u10','u','ugrd'): return 'u10'
        if short in ('10v','v10','v','vgrd'): return 'v10'
        if short == 'gust': return 'gust'
    if level_type == 'isobaricInhPa' and level == 850 and short in ('t','tmp'): return 't850'
    if level_type == 'meanSea' and short in ('prmsl','msl','mslet'): return 'msl'
    if level_type == 'lowCloudLayer' and short in ('lcc','lcdc','tcc','tcdc'): return 'lcc'
    if level_type == 'middleCloudLayer' and short in ('mcc','mcdc','tcc','tcdc'): return 'mcc'
    if level_type == 'highCloudLayer' and short in ('hcc','hcdc','tcc','tcdc'): return 'hcc'
    if level_type in ('atmosphere','entireAtmosphere') and short in ('tcc','tcdc'): return 'tcc'
    if level_type != 'surface': return None
    if category == 1 and number == 8: return 'apcp'
    if category == 1 and number == 7: return 'prate'
    if category == 1 and number == 11: return 'snow_depth'
    aliases = {'tp':'apcp','apcp':'apcp','prate':'prate','sde':'snow_depth',
               'snod':'snow_depth','gust':'gust','cape':'cape','lftx':'lftx',
               'vis':'vis','crain':'crain','csnow':'csnow','cfrzr':'cfrzr','cicep':'cicep'}
    return aliases.get(short)


def convert_value(value, field, units):
    if not finite(value) or abs(float(value)) >= 1e19:
        return np.nan
    value = float(value)
    if field in ('t2m','d2m','t850'):
        return value - 273.15 if units == 'K' else value
    if field == 'msl': return value / 100 if units == 'Pa' else value
    if field == 'vis': return value / 1000 if units == 'm' else value
    if field == 'snow_depth':
        if units == 'm': return value * 100
        if units == 'cm': return value
        return np.nan
    if field == 'apcp':
        if units == 'm': return value * 1000
        if units in ('kg m**-2','kg m-2','kg/m^2','mm'): return value
        return np.nan
    if field == 'prate':
        if units in ('kg m**-2 s**-1','kg m-2 s-1','kg/m^2/s','mm s**-1'): return value
        if units == 'mm h**-1': return value / 3600
        return np.nan
    if field in ('tcc','lcc','mcc','hcc'):
        if units in ('1','(0 - 1)','fraction'): value *= 100
        return float(np.clip(value, 0, 100))
    return value


def decode_grib(path, expected_fh, cache=None):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0: return None
    cache_key = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
    if cache is not None and cache_key in cache: return cache[cache_key]
    result = {'fields':{}, 'accumulations':[], 'averaged_rates':[]}
    try:
        with path.open('rb') as file:
            while True:
                handle = codes_grib_new_from_file(file)
                if handle is None: break
                try:
                    short = str(grib_get(handle,'shortName','')).lower()
                    field = identify_field(short,grib_get(handle,'typeOfLevel'),
                                           grib_get(handle,'level'),grib_get(handle,'parameterCategory'),
                                           grib_get(handle,'parameterNumber'))
                    if field is None: continue
                    step_type = grib_get(handle,'stepType')
                    unit = grib_get(handle,'stepUnits',1)
                    start = step_hours(grib_get(handle,'startStep'), unit)
                    end = step_hours(grib_get(handle,'endStep'), unit)
                    if end is not None and not np.isclose(end, expected_fh): continue
                    if field == 'apcp' and step_type != 'accum': continue
                    if field == 'prate' and step_type != 'avg': continue
                    if field not in ('apcp','prate') and step_type != 'instant': continue
                    point = codes_grib_find_nearest(handle,KROSNO_LAT,KROSNO_LON,npoints=1)[0]
                    value = convert_value(point['value'],field,str(grib_get(handle,'units','')))
                    if not finite(value): continue
                    if field in ('apcp','prate'):
                        if start is None or end is None or end < start: continue
                        record = {'start':start,'end':end,'value':value}
                        target = 'accumulations' if field == 'apcp' else 'averaged_rates'
                        if record not in result[target]: result[target].append(record)
                    else:
                        result['fields'][field] = value
                except Exception as exc:
                    LOG.warning('Nie odczytano pola %s w %s: %s',short,path.name,exc)
                finally:
                    codes_release(handle)
        if not result['fields']:
            LOG.warning('Brak odczytanych pól chwilowych: %s', path.name)
            return None
    except Exception as exc:
        LOG.warning('Nie udało się odczytać GRIB %s: %s',path.name,exc)
        return None
    if cache is not None: cache[cache_key] = result
    return result


def precipitation_for_step(fh, decoded, previous_fh):
    if fh == 0: return 0.0, 'analysis'
    current = decoded.get(fh)
    if not current: return np.nan, 'missing'
    previous = decoded.get(previous_fh)
    for source, key in (('APCP','accumulations'),('PRATE_AVG','averaged_rates')):
        if source == 'PRATE_AVG' and not PRECIP_ALLOW_AVERAGED_PRATE: continue
        candidates = sorted(current[key], key=lambda r:r['end']-r['start'])
        for record in candidates:
            start, end = record['start'], record['end']
            if not np.isclose(end,fh) or start > previous_fh + 1e-7: continue
            amount = record['value'] if source == 'APCP' else record['value']*(end-start)*3600
            if np.isclose(start,previous_fh):
                return (max(0.,amount),source) if amount >= -1e-6 else (np.nan,'missing')
            if previous is None: continue
            baselines = [r for r in previous[key] if np.isclose(r['start'],start) and np.isclose(r['end'],previous_fh)]
            for baseline in baselines:
                before = baseline['value'] if source == 'APCP' else baseline['value']*(previous_fh-start)*3600
                difference = amount - before
                if difference >= -1e-6: return max(0.,difference),source
    return np.nan, 'missing'


def lcl_height_m(t_c, td_c):
    return round(125*max(0.,t_c-td_c),1) if finite(t_c) and finite(td_c) else np.nan


def detect_precip_type(rain, t2m_c, t850_c, flags=None):
    if not finite(rain): return 'Brak danych'
    if rain <= 0: return 'Brak'
    flags = flags or {}
    if any(finite(v) for v in flags.values()):
        active = {key for key,value in flags.items() if finite(value) and value >= 0.5}
        if 'cfrzr' in active: return 'Deszcz marznący'
        if 'cicep' in active: return 'Krupy lodowe'
        if {'crain','csnow'} <= active: return 'Deszcz ze śniegiem'
        if 'csnow' in active: return 'Śnieg'
        if 'crain' in active: return 'Deszcz'
    if not PRECIP_TYPE_TEMPERATURE_FALLBACK or not (finite(t2m_c) and finite(t850_c)):
        return 'Nieokreślony'
    if t2m_c <= 0 and t850_c < 0: return 'Śnieg'
    if 0 < t2m_c < 2 and t850_c < 0: return 'Deszcz ze śniegiem'
    if t2m_c < 0 and t850_c > 0: return 'Deszcz marznący'
    return 'Deszcz'


def storm_risk_category(cape, li):
    if not finite(cape): return 'Brak danych'
    # Wskaźnik niestabilności, nie prawdopodobieństwo burzy ani oficjalne ostrzeżenie.
    category = 0 if cape <= 400 else 1 if cape <= 1000 else 2 if cape <= 2000 else 3
    if finite(li) and li <= -2: category = min(3, category+1)
    return ['Niskie','Średnie','Wysokie','Ekstremalne'][category]


def daily_precip_type(series):
    wet = series[~series.isin(['Brak','Brak danych','Nieokreślony'])]
    if wet.empty:
        return 'Brak' if (series == 'Brak').any() else 'Brak danych'
    types = set(wet)
    if 'Deszcz marznący' in types: return 'Deszcz marznący'
    if 'Deszcz ze śniegiem' in types or {'Deszcz','Śnieg'} <= types: return 'Deszcz ze śniegiem'
    return wet.mode().iat[0]


def process_local_gribs(forecast_hours, run_date, run_hour, cache=None):
    hours = sorted(set(forecast_hours))
    decoded = {fh:decode_grib(local_grib(run_date,run_hour,fh),fh,cache) for fh in hours}
    decoded = {fh:data for fh,data in decoded.items() if data is not None}
    rows = []
    sources = {'APCP':0,'PRATE_AVG':0,'missing':0,'analysis':0}
    for index, fh in enumerate(hours):
        if fh not in decoded: continue
        fields = decoded[fh]['fields']
        value = lambda key: fields.get(key,np.nan)
        previous = hours[index-1] if index else 0
        rain, source = precipitation_for_step(fh,decoded,previous)
        sources[source] += 1
        flags = {key:value(key) for key in ('crain','csnow','cfrzr','cicep')}
        kind = detect_precip_type(rain,value('t2m'),value('t850'),flags)
        u, v = value('u10'), value('v10')
        wind = float(np.hypot(u,v)) if finite(u) and finite(v) else np.nan
        direction = float((np.degrees(np.arctan2(-u,-v))+360)%360) if finite(wind) and wind > 0 else np.nan
        rows.append({'Czas':run_datetime(run_date,run_hour)+timedelta(hours=fh), 'T+ (h)':fh,
                     'T2M [°C]':value('t2m'),'D2M [°C]':value('d2m'),'T850 [°C]':value('t850'),
                     'MSLP [hPa]':value('msl'),'CL [%]':value('lcc'),'CM [%]':value('mcc'),
                     'CH [%]':value('hcc'),'CC [%]':value('tcc'),'RRR [mm]':rain,
                     'Rodzaj opadu':kind,'SNOW [cm]':value('snow_depth'),'WSPD [m/s]':wind,
                     'GUST [m/s]':value('gust'),'WDIR [°]':direction,'CAPE [J/kg]':value('cape'),
                     'LIFTED [°C]':value('lftx'),'VIS [km]':value('vis')})
    if not rows: return pd.DataFrame(),pd.DataFrame()
    df = pd.DataFrame(rows).sort_values('Czas').reset_index(drop=True)
    # Dotychczasowa dokładność parametrów. Opad zaokrąglamy dopiero
    # po obliczeniu sum dobowych, nie podczas odejmowania akumulacji APCP.
    decimal_places = {
        'T2M [°C]':1, 'D2M [°C]':1, 'T850 [°C]':1, 'MSLP [hPa]':1,
        'CL [%]':1, 'CM [%]':1, 'CH [%]':1, 'CC [%]':1,
        'SNOW [cm]':1, 'WSPD [m/s]':1, 'GUST [m/s]':1, 'VIS [km]':1,
        'WDIR [°]':2, 'CAPE [J/kg]':2, 'LIFTED [°C]':2,
    }
    for column, digits in decimal_places.items():
        df[column] = df[column].round(digits)
    df['Date'] = df['Czas'].dt.date
    grouped = df.groupby('Date')
    daily = grouped.agg(Tmax=('T2M [°C]','max'),Tmin=('T2M [°C]','min'),
                        Wsp_sred=('WSPD [m/s]','mean'),Pres_sred=('MSLP [hPa]','mean'),
                        CAPE_max=('CAPE [J/kg]','max'),LIFTED_min=('LIFTED [°C]','min'),
                        VIS_min=('VIS [km]','min'),T_mean=('T2M [°C]','mean'),Td_mean=('D2M [°C]','mean'))
    # Opad opisuje okres kończący się w Czas. Krok kończący się o 00 UTC
    # należy do poprzedniej doby. Daty samych wierszy CSV nadal pozostają UTC.
    rain_rows = df[df['T+ (h)'] > 0].copy()
    rain_rows['Opad_Date'] = (rain_rows['Czas']-pd.Timedelta(microseconds=1)).dt.date
    rain_grouped = rain_rows.groupby('Opad_Date')
    daily['Suma_opadu'] = rain_grouped['RRR [mm]'].sum(min_count=1).reindex(daily.index)
    daily['Opad_dostepne_kroki'] = rain_grouped['RRR [mm]'].count().reindex(daily.index,fill_value=0)
    rain_expected = pd.Series([run_datetime(run_date,run_hour)+timedelta(hours=fh)-timedelta(microseconds=1) for fh in hours if fh > 0],dtype='datetime64[ns]')
    daily['Opad_oczekiwane_kroki'] = rain_expected.dt.date.value_counts().reindex(daily.index,fill_value=0)
    daily['Opad_brakujace_wartosci'] = daily['Opad_oczekiwane_kroki']-daily['Opad_dostepne_kroki']
    daily['Dostepne_terminy'] = grouped.size()
    # Liczba brakujących plików i zakres doby są jawne w arkuszu dziennym.
    expected = pd.Series([run_datetime(run_date,run_hour)+timedelta(hours=fh) for fh in hours])
    expected_counts = expected.dt.date.value_counts()
    daily['Oczekiwane_terminy'] = expected_counts.reindex(daily.index).fillna(0).astype(int)
    daily['Brakujace_terminy'] = daily['Oczekiwane_terminy']-daily['Dostepne_terminy']
    daily['PrecType'] = rain_grouped['Rodzaj opadu'].agg(daily_precip_type).reindex(daily.index).fillna('Brak danych')
    daily = daily.reset_index()
    daily['LCL_m'] = daily.apply(lambda r:lcl_height_m(r['T_mean'],r['Td_mean']),axis=1)
    daily['StormRisk'] = daily.apply(lambda r:storm_risk_category(r['CAPE_max'],r['LIFTED_min']),axis=1)
    daily['Date_str'] = daily['Date'].astype(str)
    df['LCL_m'] = df.apply(lambda r:lcl_height_m(r['T2M [°C]'],r['D2M [°C]']),axis=1)
    # Aliasy zachowane dla dotychczasowego PHP; nowa widoczna kolumna jest po RRR.
    df['PrecType_step'] = df['Rodzaj opadu']
    df['PrecType'] = df['Rodzaj opadu']
    df['StormRisk'] = df.apply(lambda r:storm_risk_category(r['CAPE [J/kg]'],r['LIFTED [°C]']),axis=1)
    LOG.info('Opad: %s; dane: %s/%s terminów. Czas CSV: UTC.',sources,len(df),len(hours))
    for field in ('T2M [°C]','RRR [mm]','CC [%]','SNOW [cm]'):
        if df[field].isna().any(): LOG.warning('%s: brakuje %s wartości',field,int(df[field].isna().sum()))
    # Kolumna RRR oraz prezentowana suma dobowa: 1 miejsce, jak wcześniej.
    df['RRR [mm]'] = df['RRR [mm]'].round(1)
    daily['Suma_opadu'] = daily['Suma_opadu'].round(1)
    return df,daily


def save_outputs(df, daily, run_date, run_hour):
    Path(OUTPUT_DIR).mkdir(parents=True,exist_ok=True)
    if df.empty:
        print("⚠️ Brak danych do zapisania (brak pobranych plików).")
        return []

    xlsx_path = os.path.join(OUTPUT_DIR, f"krosno_gfs_{run_date}_{run_hour}z.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="prognoza", index=False)
        daily.to_excel(writer, sheet_name="dzienna_prognoza", index=False)
        workbook = writer.book
        worksheet = writer.sheets["prognoza"]
        
        border_fmt = workbook.add_format({'border': 1})
        worksheet.conditional_format(f'A1:{chr(65 + len(df.columns) - 1)}{len(df) + 1}', {'type': 'no_blanks', 'format': border_fmt})
        
        if "Rodzaj opadu" in df.columns:
            col_idx = df.columns.get_loc("Rodzaj opadu")
            rng = f"{chr(65+col_idx)}2:{chr(65+col_idx)}{len(df)+1}"
            fmt_rain = workbook.add_format({'bg_color': PREC_TYPE_TO_COLOR.get("Deszcz", "#90EE90"), 'border': 1})
            fmt_snow = workbook.add_format({'bg_color': PREC_TYPE_TO_COLOR.get("Śnieg", "#ADD8E6"), 'border': 1})
            fmt_mix = workbook.add_format({'bg_color': PREC_TYPE_TO_COLOR.get("Deszcz ze śniegiem", "#00FFBB"), 'border': 1})
            fmt_freeze = workbook.add_format({'bg_color': PREC_TYPE_TO_COLOR.get("Deszcz marznący", "#FFA500"), 'border': 1})
            
            worksheet.conditional_format(rng, {'type': 'cell', 'criteria': 'equal to', 'value': '"Deszcz"', 'format': fmt_rain})
            worksheet.conditional_format(rng, {'type': 'cell', 'criteria': 'equal to', 'value': '"Śnieg"', 'format': fmt_snow})
            worksheet.conditional_format(rng, {'type': 'cell', 'criteria': 'equal to', 'value': '"Deszcz ze śniegiem"', 'format': fmt_mix})
            worksheet.conditional_format(rng, {'type': 'cell', 'criteria': 'equal to', 'value': '"Deszcz marznący"', 'format': fmt_freeze})
            
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max() if len(df)>0 else 0, len(col)) + 2
            worksheet.set_column(i, i, max_len)
            
    print("\n✅ Excel zapisany:", xlsx_path)
    
    csv_path = os.path.join(OUTPUT_DIR, f"krosno_gfs_{run_date}_{run_hour}z.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print("✅ CSV zapisany:", csv_path)
    
    df_plot = df[df["T+ (h)"] <= 120].copy() if not df.empty else pd.DataFrame()
    if not df_plot.empty:
        expected_times = pd.date_range(df_plot["Czas"].min(), df_plot["Czas"].max(), freq="h")
        df_plot = df_plot.set_index("Czas").reindex(expected_times).rename_axis("Czas").reset_index()
    out_png = os.path.join(OUTPUT_DIR, f"meteorogram_krosno_120h.png")
    
    if not df_plot.empty:
        fig, axes = plt.subplots(7, 1, figsize=(13, 15), sharex=True)
        fig.subplots_adjust(hspace=0.3)
        time_axis = df_plot["Czas"]
        
        axes[0].plot(time_axis, df_plot["T2M [°C]"], color="#D62728", label="Temperatura")
        axes[0].plot(time_axis, df_plot["D2M [°C]"], color="#1F77B4", linestyle="--", label="Punkt rosy")
        axes[0].set_ylabel("°C")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[0].grid(True, ls=":")
        
        # Szerokość słupka (width) zmniejszona do 0.03, by uniknąć nachodzenia przy kroku 1h
        axes[1].bar(time_axis, df_plot["RRR [mm]"], width=0.03, color="#1F77B4", label="Opad [mm]")
        axes[1].plot(time_axis, df_plot["RRR [mm]"].cumsum(), color="#000080", linewidth=1, label="Suma dostępnych opadów")
        axes[1].set_ylabel("mm")
        axes[1].set_ylim(bottom=0)
        axes[1].legend(loc="upper left", fontsize=8)
        axes[1].grid(True, ls=":")
        
        axes[2].plot(time_axis, df_plot["MSLP [hPa]"], color="#000000")
        axes[2].set_ylabel("hPa")
        axes[2].grid(True, ls=":")
        
        axes[3].plot(time_axis, df_plot["WSPD [m/s]"], color="#FF7F0E", label="Wiatr")
        axes[3].plot(time_axis, df_plot["GUST [m/s]"], color="#D62728", linestyle="--", label="Porywy")
        axes[3].set_ylabel("m/s")
        axes[3].legend(loc="upper left", fontsize=8)
        axes[3].grid(True, ls=":")
        
        low = df_plot["CL [%]"]
        mid = df_plot["CM [%]"]
        high = df_plot["CH [%]"]
        axes[4].fill_between(time_axis, 0, low, color="#b0c4de", label="Niskie")
        axes[4].plot(time_axis, mid, color="#778899", label="Średnie")
        axes[4].plot(time_axis, high, color="#2f4f4f", label="Wysokie")
        axes[4].plot(time_axis, df_plot["CC [%]"], color="#263238", linewidth=1.5, label="Ogółem")
        axes[4].set_ylabel("Chmury [%]")
        axes[4].set_ylim(0, 100)
        axes[4].legend(loc="upper left", fontsize=8)
        axes[4].grid(True, ls=":")
        
        axes[5].plot(time_axis, df_plot["CAPE [J/kg]"], color="#8A2BE2", label="CAPE")
        li_axis = axes[5].twinx()
        li_axis.plot(time_axis, df_plot["LIFTED [°C]"], color="#2ca02c", linestyle="--", label="Lifted")
        li_axis.set_ylabel("LI [°C]")
        li_axis.legend(loc="upper right", fontsize=8)
        axes[5].set_ylabel("CAPE [J/kg]")
        axes[5].set_ylim(bottom=0)
        axes[5].legend(loc="upper left", fontsize=8)
        axes[5].grid(True, ls=":")
        
        axes[6].bar(time_axis, df_plot["SNOW [cm]"], width=0.03, color="#87CEFA", label="Pokrywa śnieżna [cm]")
        ax7b = axes[6].twinx()
        ax7b.plot(time_axis, df_plot["VIS [km]"].fillna(np.nan), color="#8B4513", linewidth=1, label="Widzialność [km]")
        axes[6].set_ylabel("cm")
        axes[6].set_ylim(bottom=0)
        ax7b.set_ylabel("km")
        axes[6].legend(loc="upper left", fontsize=8)
        ax7b.legend(loc="upper right", fontsize=8)
        axes[6].grid(True, ls=":")
        
        date_fmt = DateFormatter("%d.%m\n%H UTC")
        axes[-1].xaxis.set_major_formatter(date_fmt)
        
        plt.suptitle(f"GFS Krosno – Meteorogram 120h ({run_date}{run_hour}Z)", fontsize=14, weight="bold")
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print("✅ Meteorogram zapisany:", out_png)
    else:
        print("⚠️ Brak danych do meteorogramu.")
    return [xlsx_path, csv_path] + ([out_png] if not df_plot.empty else [])



def download_missing_gribs_parallel(forecast_hours, run_date, run_hour, deadline=None):
    downloaded, pending = [], []
    Path(OUTPUT_DIR).mkdir(parents=True,exist_ok=True)
    for fh in forecast_hours:
        path = local_grib(run_date,run_hour,fh)
        if path.is_file() and path.stat().st_size > 2048:
            downloaded.append(str(path))
        else:
            pending.append(fh)
    def fetch_single(fh):
        if deadline is not None and monotonic() >= deadline:
            return fh,None,'Przekroczono limit czasu'
        path = local_grib(run_date,run_hour,fh)
        temporary = path.with_suffix(f'.{os.getpid()}.part')
        file_name = f'gfs.t{run_hour}z.pgrb2.0p25.f{fh:03d}'
        url,params = build_url(file_name,run_date,run_hour)
        timeout = REQUEST_TIMEOUT_SECONDS if deadline is None else max(1,min(REQUEST_TIMEOUT_SECONDS,deadline-monotonic()))
        try:
            response = requests.get(url,params=params,headers=HEADERS,timeout=timeout)
            if response.status_code != 200:
                return fh,None,f'HTTP {response.status_code}'
            if not response.content.startswith(b'GRIB'):
                return fh,None,'Odpowiedź nie zawiera GRIB'
            temporary.write_bytes(response.content)
            os.replace(temporary,path)
            LOG.info('Pobrano f%03d: %.1f kB',fh,path.stat().st_size/1024)
            return fh,str(path),None
        except Exception as exc:
            return fh,None,str(exc)
        finally:
            temporary.unlink(missing_ok=True)
    if pending:
        LOG.info('Pobieranie %s plików, %s równoległe połączenia',len(pending),MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(fetch_single,fh) for fh in pending]
            for future in as_completed(futures):
                fh,path,error = future.result()
                if path: downloaded.append(path)
                else: LOG.warning('f%03d: %s',fh,error)
    missing = [fh for fh in forecast_hours if not local_grib(run_date,run_hour,fh).is_file()
               or local_grib(run_date,run_hour,fh).stat().st_size <= 2048]
    return downloaded,missing


def ftp_connection():
    load_dotenv()
    host,user,password = os.getenv('FTP_HOST'),os.getenv('FTP_USER'),os.getenv('FTP_PASS')
    if not all((host,user,password)): return None
    return FTP(host,user,password,timeout=30)


def ensure_ftp_directory(ftp,directory):
    try:
        ftp.cwd(directory)
    except error_perm:
        ftp.mkd(directory)
        ftp.cwd(directory)


def ftp_store(ftp,path,name):
    # Nazwy publiczne pozostają takie same; podmiana dopiero po pełnej wysyłce.
    temporary = f'.{name}.{os.getpid()}.upload'
    with open(path,'rb') as file:
        ftp.storbinary(f'STOR {temporary}',file)
    try:
        ftp.rename(temporary,name)
    except error_perm as exc:
        # Część serwerów nie pozwala zmienić nazwy na już istniejącą.
        # Zachowujemy obsługę zwykłego STOR, jak w poprzedniej wersji.
        LOG.warning('FTP rename %s niedostępne (%s); używam STOR',name,exc)
        with open(path,'rb') as file: ftp.storbinary(f'STOR {name}',file)
        try: ftp.delete(temporary)
        except error_perm: pass


def upload_to_ftp(files_to_send,run_date,run_hour,publish_latest=True):
    if not files_to_send: return False
    ftp = None
    try:
        ftp = ftp_connection()
        if ftp is None:
            LOG.warning('Brak konfiguracji FTP — wyniki pozostają lokalnie')
            return False
        for path in files_to_send:
            if not os.path.isfile(path): continue
            if str(path).endswith('.csv'):
                is_3h = str(path).endswith('_3h.csv')
                current_name = 'gfs-tab-3h.csv' if is_3h else 'gfs-tab.csv'
                prefix = 'gfs_tab_3h' if is_3h else 'gfs_tab'
                archive_name = f'{prefix}_{run_date[:4]}_{run_date[4:6]}_{run_date[6:8]}_{run_hour}.csv'
                ensure_ftp_directory(ftp,FTP_ARCH_DIR)
                ftp_store(ftp,path,archive_name)
                LOG.info('FTP: archiv/%s',archive_name)
                if publish_latest:
                    ftp.cwd(FTP_BASE_DIR)
                    ftp_store(ftp,path,current_name)
                    LOG.info('FTP: nadpisano %s aktualnym runem %s%sZ',current_name,run_date,run_hour)
            elif publish_latest:
                ftp.cwd(FTP_BASE_DIR)
                ftp_store(ftp,path,os.path.basename(path))
        return True
    except Exception as exc:
        LOG.error('Wysyłka FTP nie powiodła się: %s',exc)
        return False
    finally:
        if ftp is not None:
            try: ftp.quit()
            except Exception: ftp.close()


def check_and_fetch_previous_cycle(current_run_date,current_run_hour,deadline=None):
    previous = run_datetime(current_run_date,current_run_hour)-timedelta(hours=6)
    date,hour = previous.strftime('%Y%m%d'),previous.strftime('%H')
    name = f'gfs_tab_{date[:4]}_{date[4:6]}_{date[6:8]}_{hour}.csv'
    ftp = None
    try:
        ftp = ftp_connection()
        if ftp is None: return
        ftp.cwd(FTP_ARCH_DIR)
        if name in {os.path.basename(item) for item in ftp.nlst()}:
            LOG.info('Poprzedni run jest już w archiwum: %s',name)
            return
    except error_perm:
        pass
    except Exception as exc:
        LOG.warning('Nie można sprawdzić archiwum FTP: %s',exc)
        return
    finally:
        if ftp is not None:
            try: ftp.quit()
            except Exception: ftp.close()
    LOG.info('Uzupełniam poprzedni run %s%sZ w archiwum',date,hour)
    # Jedna próba; nie opóźniamy bieżącego runu wielokrotnym oczekiwaniem na stary.
    _,missing = download_missing_gribs_parallel(FORECAST_HOURS,date,hour,deadline)
    if 0 in missing:
        LOG.warning('Poprzedni run niedostępny; przechodzę do bieżącego')
        return
    df,daily = process_local_gribs(FORECAST_HOURS,date,hour,cache={})
    if not df.empty:
        files = save_outputs(df,daily,date,hour)
        # Nadpisanie głównego CSV wykonuje bieżący run, nie uzupełnianie historii.
        upload_to_ftp(files,date,hour,publish_latest=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='GFS Krosno; daty i doby w UTC')
    parser.add_argument('--run',help='Konkretny run YYYYMMDDHH, np. 2026100600')
    parser.add_argument('--no-ftp',action='store_true',help='Test lokalny bez wysyłki FTP')
    parser.add_argument('--skip-previous',action='store_true',help='Bez uzupełniania poprzedniego runu')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.run:
        try:
            dt = datetime.strptime(args.run,'%Y%m%d%H')
            if len(args.run)!=10 or dt.hour not in (0,6,12,18): raise ValueError()
            date,hour = dt.strftime('%Y%m%d'),dt.strftime('%H')
        except ValueError:
            parser.error('--run wymaga YYYYMMDDHH z godziną 00,06,12,18')
    else:
        date,hour = select_run()
    Path(OUTPUT_DIR).mkdir(parents=True,exist_ok=True)
    LOG.info('Start GFS %s%sZ; zapis czasu UTC',date,hour)
    deadline = monotonic()+MAX_TOTAL_WAIT_MINUTES*60
    if not args.no_ftp and not args.skip_previous:
        check_and_fetch_previous_cycle(date,hour,deadline)
    cache = {}
    last_signature = None
    had_data = False
    upload_ok = args.no_ftp
    while monotonic() < deadline:
        _,missing = download_missing_gribs_parallel(FORECAST_HOURS,date,hour,deadline)
        signature = tuple((fh,local_grib(date,hour,fh).stat().st_size,
                           local_grib(date,hour,fh).stat().st_mtime_ns)
                          for fh in FORECAST_HOURS if local_grib(date,hour,fh).is_file())
        if signature != last_signature or not upload_ok:
            df,daily = process_local_gribs(FORECAST_HOURS,date,hour,cache)
            if not df.empty:
                had_data = True
                files = save_outputs(df,daily,date,hour)
                upload_ok = args.no_ftp or upload_to_ftp(files,date,hour,publish_latest=True)
                last_signature = signature
                LOG.info('Prognoza częściowa: %s/%s terminów; brak plików: %s',len(df),len(FORECAST_HOURS),len(missing))
                if not missing:
                    LOG.info('Pobrano wszystkie pliki. Braki parametrów, jeśli są, wymieniono w logu.')
                    break
        if not missing and not had_data:
            LOG.error('Pliki pobrane, ale brak danych do zapisania')
            break
        remaining = deadline-monotonic()
        if remaining > 0: sleep(min(RETRY_INTERVAL_SECONDS,remaining))
    if not had_data:
        LOG.error('Brak danych; dotychczasowy gfs-tab.csv nie został zastąpiony')
        return 1
    if not upload_ok:
        LOG.error('Wyniki lokalne gotowe, ale FTP nie zakończyło się powodzeniem')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
