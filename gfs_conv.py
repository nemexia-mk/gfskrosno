#!/usr/bin/env python3

import os
import requests
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from time import sleep
from dotenv import load_dotenv
from ftplib import FTP

# -----------------------
# CONFIG
# -----------------------
OUTPUT_DIR = "gfs_conv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KROSNO_LAT = 49.69
KROSNO_LON = 21.77

BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

FORECAST_HOURS = list(range(0, 121, 3))  # tylko 120h – burze

RETRY_INTERVAL = 600

# -----------------------
# RUN LOGIC
# -----------------------
now = datetime.utcnow()
t = now.time()

if t >= time(20,0) or t < time(3,0):
    RUN_HOUR = "18"
    RUN_DATE = (now if t>=time(22,0) else now - timedelta(days=1)).strftime("%Y%m%d")
elif t < time(8,30):
    RUN_HOUR = "00"
    RUN_DATE = now.strftime("%Y%m%d")
elif t < time(14,30):
    RUN_HOUR = "06"
    RUN_DATE = now.strftime("%Y%m%d")
else:
    RUN_HOUR = "12"
    RUN_DATE = now.strftime("%Y%m%d")

CYCLE_DIR = f"gfs.{RUN_DATE}/{RUN_HOUR}/atmos"

# -----------------------
# PARAMS (KONWEKCJA)
# -----------------------
PARAMS = (
    "&lev_2_m_above_ground=on"
    "&lev_10_m_above_ground=on"
    "&lev_850_mb=on"
    "&lev_surface=on"
    "&lev_mean_sea_level=on"
    "&lev_entire_atmosphere_%28considered_as_a_single_layer%29=on"
    "&var_TMP=on"
    "&var_DPT=on"
    "&var_RH=on"
    "&var_UGRD=on"
    "&var_VGRD=on"
    "&var_CAPE=on"
    "&var_CIN=on"
    "&var_LFTX=on"
    "&var_PWAT=on"
    "&var_PRATE=on"
)

def build_url(fh):
    file = f"gfs.t{RUN_HOUR}z.pgrb2.0p25.f{fh:03d}"
    return f"{BASE_URL}?file={file}&dir=/{CYCLE_DIR}{PARAMS}"

# -----------------------
# HELPERS
# -----------------------
def open_ds(path, filt):
    try:
        return xr.open_dataset(path, engine="cfgrib",
                               backend_kwargs={"filter_by_keys": filt, "indexpath": ""})
    except:
        return None

def get_val(ds, name):
    try:
        val = ds[name].sel(latitude=KROSNO_LAT, longitude=KROSNO_LON, method="nearest")
        return float(val.values)
    except:
        return np.nan

def lcl(t, td):
    if np.isnan(t) or np.isnan(td):
        return np.nan
    return 125*(t-td)

def stp(cape, srh, shear, lcl_h):
    if np.isnan(cape) or np.isnan(shear):
        return np.nan
    return (cape/1500.0) * (srh/150.0 if not np.isnan(srh) else 1) * (shear/20.0) * ((2000-lcl_h)/1000.0)

# -----------------------
# DOWNLOAD
# -----------------------
def download(fh):
    path = f"{OUTPUT_DIR}/f{fh:03d}.grib2"
    if os.path.exists(path):
        return path

    r = requests.get(build_url(fh), timeout=60)
    if r.status_code != 200:
        return None

    with open(path, "wb") as f:
        f.write(r.content)
    return path

# -----------------------
# MAIN PROCESS
# -----------------------
rows = []

for fh in FORECAST_HOURS:
    path = download(fh)
    if not path:
        continue

    ds2 = open_ds(path, {"typeOfLevel":"heightAboveGround","level":2})
    ds10 = open_ds(path, {"typeOfLevel":"heightAboveGround","level":10})
    dss = open_ds(path, {"typeOfLevel":"surface"})
    ds850 = open_ds(path, {"typeOfLevel":"isobaricInhPa","level":850})

    t2 = get_val(ds2,"t2m") - 273.15
    td = get_val(ds2,"d2m") - 273.15
    cape = get_val(dss,"cape")
    cin = get_val(dss,"cin")
    li = get_val(dss,"lftx")
    pwat = get_val(dss,"pwat")
    rh = get_val(ds2,"r2")

    u = get_val(ds10,"u10")
    v = get_val(ds10,"v10")
    wind = np.sqrt(u*u+v*v)

    t850 = get_val(ds850,"t") - 273.15

    lcl_h = lcl(t2, td)

    shear = wind * 1.5  # uproszczenie
    srh = wind * 10     # uproszczenie

    stp_val = stp(cape, srh, shear, lcl_h)

    valid = datetime.strptime(RUN_DATE+RUN_HOUR,"%Y%m%d%H") + timedelta(hours=fh)

    rows.append({
        "time": valid,
        "T2M": round(t2,1),
        "Td2M": round(td,1),
        "CAPE": round(cape,0),
        "CIN": round(cin,0),
        "LI": round(li,1),
        "PWAT": round(pwat,1),
        "RH": round(rh,1),
        "WIND": round(wind,1),
        "T850": round(t850,1),
        "LCL": round(lcl_h,0),
        "SHEAR": round(shear,1),
        "SRH": round(srh,0),
        "STP": round(stp_val,2)
    })

# -----------------------
# SAVE CSV
# -----------------------
df = pd.DataFrame(rows)
csv_path = f"{OUTPUT_DIR}/gfs-conv.csv"
df.to_csv(csv_path, index=False)

print("✅ zapisano:", csv_path)

# -----------------------
# FTP
# -----------------------
load_dotenv()

try:
    ftp = FTP(os.getenv("FTP_HOST"),
              os.getenv("FTP_USER"),
              os.getenv("FTP_PASS"))

    ftp.cwd("/stacja.meteo-krosno.pl/")
    with open(csv_path,"rb") as f:
        ftp.storbinary("STOR gfs-conv.csv", f)

    ftp.quit()
    print("📤 wysłano FTP")
except Exception as e:
    print("FTP error:", e)
