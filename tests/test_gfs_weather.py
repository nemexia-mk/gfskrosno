import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import gfs_krosno as g
import gfs_conv as c
from eccodes import codes_grib_new_from_samples,codes_set,codes_write,codes_release

SCHEMA = ['Czas','T+ (h)','T2M [°C]','D2M [°C]','T850 [°C]','MSLP [hPa]',
 'CL [%]','CM [%]','CH [%]','CC [%]','RRR [mm]','Rodzaj opadu','SNOW [cm]',
 'WSPD [m/s]','GUST [m/s]','WDIR [°]','CAPE [J/kg]','LIFTED [°C]','VIS [km]',
 'Date','LCL_m','PrecType_step','PrecType','StormRisk']


def fixture(fh):
    return {'fields':{'t2m':12.,'d2m':8.,'t850':2.,'msl':1012.,'lcc':30.,
      'mcc':40.,'hcc':60.,'tcc':80.,'u10':3.,'v10':4.,'gust':8.,'cape':0.,
      'lftx':5.,'snow_depth':0.,'vis':10.,'crain':1.,'csnow':0.,'cfrzr':0.,
      'cicep':0.,'cin':0.,'pwat':20.,'rh2m':76.},
      'accumulations':[{'start':0,'end':fh,'value':fh*.5}], 'averaged_rates':[],
      'conv_accumulations':[{'start':0,'end':fh,'value':fh*.2}],
      'sun_accumulations':[{'start':0,'end':fh,'value':fh*1800.}], 'periods':{}}


class WeatherTests(unittest.TestCase):
    def test_original_csv_schema_and_zero_values(self):
        with patch.object(g,'decode_grib',side_effect=lambda p,fh,*a:fixture(fh)):
            df,_ = g.process_local_gribs([0,1,2,3],'20261022','06')
        self.assertEqual(list(df.columns),SCHEMA)
        self.assertEqual(df['RRR [mm]'].tolist(),[0.,.5,.5,.5])
        self.assertEqual(df.attrs['extra_rows'][2]['convRain'],.2)
        self.assertEqual(df.attrs['extra_rows'][2]['sunshine'],.5)
        self.assertTrue(all(r['stormLevel']==0 for r in df.attrs['extra_rows']))
    def test_intervals_preserve_resets_and_missing_baseline(self):
        d={1:fixture(1),2:fixture(2),3:fixture(3)}
        self.assertAlmostEqual(g.accumulation_for_step(3,d,2,'conv_accumulations'),.2)
        self.assertTrue(np.isnan(g.accumulation_for_step(3,{3:d[3]},2,'conv_accumulations')))
        d[3]['conv_accumulations']=[{'start':2,'end':3,'value':.7}]
        self.assertEqual(g.accumulation_for_step(3,d,2,'conv_accumulations'),.7)
    def test_storm_categories_need_cin_and_activity(self):
        self.assertIsNone(g.storm_potential(2000,None,1,40,5))
        self.assertEqual(g.storm_potential(2000,-300,1,40,5),0)
        self.assertEqual(g.storm_potential(2000,-20,0,0,0),1)
        self.assertEqual(g.storm_potential(2000,-20,1,40,5),4)
        self.assertEqual(g.storm_potential(300,-20,1,25,1),2)
    def test_flags_win_over_temperature_for_freezing_rain(self):
        self.assertEqual(g.detect_precip_type(1,3,-5,{'cfrzr':1,'crain':1}),'Deszcz marznący')
        self.assertEqual(g.detect_precip_type(1,1,-3,{'crain':1,'csnow':1}),'Deszcz ze śniegiem')
    def test_real_eccodes_profile_and_temperature_conversion(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'sample.grib2'
            with path.open('wb') as out:
                for short,kind,level,value in [('t','isobaricInhPa',850,273.15),('2t','heightAboveGround',2,283.15)]:
                    h=codes_grib_new_from_samples('regular_ll_sfc_grib2')
                    try:
                        codes_set(h,'typeOfLevel',kind); codes_set(h,'level',level)
                        codes_set(h,'shortName',short); codes_set(h,'forecastTime',3)
                        from eccodes import codes_set_values,codes_get_size
                        codes_set_values(h,np.full(codes_get_size(h,'values'),value))
                        codes_write(h,out)
                    finally: codes_release(h)
            d=g.decode_grib(path,3)
            self.assertAlmostEqual(d['fields']['t850'],0.,places=3)
            self.assertAlmostEqual(d['fields']['profile_850_t'],0.,places=3)
            self.assertAlmostEqual(d['fields']['t2m'],10.,places=3)
    def test_profile_uses_real_height_for_shear(self):
        from metpy.calc import pressure_to_height_std
        from metpy.units import units
        f=fixture(0)['fields']; f.update(terrain=100.,surface_pressure=100000.)
        for p in g.PROFILE_LEVELS:
            z=float(pressure_to_height_std(p*units.hPa).to('m').magnitude)
            for k,v in {'z':z,'t':15-z*.0065,'rh':70.,'u':z*.002,'v':0.}.items(): f[f'profile_{p}_{k}']=v
        f.update(u10=.2,v10=0.)
        out=g.convective_diagnostics(f)
        self.assertAlmostEqual(out['shear06'],12.,places=1)
        self.assertNotIn('dCape',out)
    def test_save_outputs_separates_extensions(self):
        with patch.object(g,'decode_grib',side_effect=lambda p,fh,*a:fixture(fh)):
            df,daily=g.process_local_gribs([0,1,2],'20261022','06')
        with tempfile.TemporaryDirectory() as root,patch.object(g,'OUTPUT_DIR',root):
            files=g.save_outputs(df,daily,'20261022','06')
            legacy=pd.read_csv(next(p for p in files if p.endswith('06z.csv')))
            self.assertEqual(list(legacy.columns),SCHEMA)
            extra=json.loads(Path(next(p for p in files if p.endswith('.json'))).read_text())
            self.assertEqual(extra['run'],'2026-10-22T06:00:00Z')
            self.assertEqual(len(extra['rows']),3)
    def test_convection_never_fabricates_probabilities_or_dcape(self):
        with patch.object(g,'decode_grib',side_effect=lambda p,fh,*a:fixture(fh)):
            df,_=g.process_local_gribs([0,1,2],'20261022','06')
        result=c.build_dataframe(df,df.attrs['extra_rows'])
        self.assertEqual(list(result.columns[:37]),c.LEGACY_COLUMNS)
        self.assertTrue(result['Prob Tornado [%]'].isna().all())
        self.assertTrue(result['Błyski [1/min]'].isna().all())
        self.assertTrue(result['DCAPE [J/kg]'].isna().all())
        self.assertEqual(result['CAPE [J/kg]'].tolist(),[0.,0.,0.])
    def test_download_query_includes_precip_flags_and_profiles(self):
        p=g.build_params('test','20261022','06')
        for name in ('CIN','PWAT','ACPCP','CSNOW','CFRZR','CICEP','RH','DSWRF'):
            self.assertEqual(p['var_'+name],'on')
        self.assertEqual(p['lev_100_m_above_ground'],'on')
    def test_missing_day_never_becomes_dry(self):
        d={1:fixture(1),3:fixture(3)}
        rain,source=g.precipitation_for_step(3,d,2)
        self.assertTrue(np.isnan(rain)); self.assertEqual(source,'missing')

if __name__=='__main__': unittest.main()
