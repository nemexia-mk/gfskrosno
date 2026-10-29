"""Read-only verification of public results, without outputting private source."""
from urllib.request import urlopen,Request,build_opener,HTTPRedirectHandler
from urllib.error import HTTPError
import json,re,xml.etree.ElementTree as ET
class NoRedirect(HTTPRedirectHandler):
 def redirect_request(self,*args):return None
for path in ['/','/prognoza/','/wp-content/uploads/2019/08/IMG_3942.jpg']:
 try:r=build_opener(NoRedirect).open('http://meteo-krosno.pl'+path,timeout=30)
 except HTTPError as e:r=e
 assert r.status==301 and r.headers['Location'].startswith('https://meteo-krosno.pl/'),(path,r.status)
 print('HTTPS_REDIRECT',path,flush=True)
paths=['/','/prognoza/','/meteorogram/','/sprawdzalnosc/','/stacja-lesniowka/','/2022/04/08/podsumowanie-styczen-2022/','/2019/08/14/meteo-info-1-front-cieply/']
for path in paths:
 with urlopen('https://meteo-krosno.pl'+path,timeout=40) as r:
  html=r.read().decode();assert r.status==200
  assert 'mk-privacy-root' in html,path
  assert not re.search(r'<script[^>]+src=["\'][^"\']*(?:googletagmanager|google-analytics)',html,re.I),path
  assert not re.search(r'gtag\s*\(',html),path
  assert 'http://meteo-krosno.pl/wp-content/uploads/' not in html,path
  assert len(re.findall('<h1\\b',html,re.I))==1,path
 print('HTML_PRIVACY_AND_IMAGES',path,flush=True)
try:r=urlopen('https://meteo-krosno.pl/2019/08/14/testowy-wpis-prognozy/',timeout=35)
except HTTPError as e:r=e
assert r.status==404,'Test post still public'
with urlopen('https://meteo-krosno.pl/post-sitemap.xml',timeout=35) as r:
 b=r.read();ET.fromstring(b);assert b'testowy-wpis-prognozy' not in b
with urlopen('https://meteo-krosno.pl/?mk_api=live&publication_check=1',timeout=45) as r:data=json.load(r)
station=next(s for s in data['stations'] if s['id']=='lesniowka')
assert isinstance(station.get('humidity'),(float,int)) and 0<=station['humidity']<=100,'Missing actual Lesniowka humidity'
assert station['timestamps']['humidity'] and station.get('humiditySensor') in ['tf_ch1','temp_and_humidity_ch1'],'Humidity is not from the selected sensor'
print('LIVE_API_HUMIDITY',station['humidity'],'timestamp',station['timestamps']['humidity'],flush=True)
print('Public publication checks passed.',flush=True)
