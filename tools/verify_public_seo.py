"""Read-only checks against server HTML, including content before JavaScript."""
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import re
import xml.etree.ElementTree as ET

BASE = 'https://meteo-krosno.pl'


def get(path):
    try:
        response = urlopen(Request(BASE + path, headers={'User-Agent': 'MeteoKrosno-SEO-check/1.0', 'Cache-Control': 'no-cache'}), timeout=40)
    except HTTPError as exc:
        response = exc
    return response.status, response.read().decode('utf-8')


for path in ['/', '/prognoza/', '/meteorogram/', '/analizator/', '/miesiac-na-zywo/', '/stacje/', '/stacja-turaszowka/', '/klimat-miesieczny/']:
    status, html = get(path)
    assert status == 200, (path, status)
    assert len(re.findall(r'<title\b', html, re.I)) == 1, path
    assert '<title>Meteo Krosno</title>' not in html, path
    assert len(re.findall(r'<h1\b', html, re.I)) == 1, path
    assert len(re.findall(r'<meta\b[^>]*name=[\"\']description[\"\']', html, re.I)) == 1, path
    assert len(re.findall(r'<meta\b[^>]*name=[\"\']robots[\"\']', html, re.I)) == 1, path
    if path not in ['/stacja-turaszowka/', '/klimat-miesieczny/']:
        assert 'mk-seo-content' in html, path
    else:
        assert '<main' in html and len(html) > 10000, 'Native server dashboard missing: ' + path
    print('HTML VERIFIED', path, flush=True)
status, html = get('/przykladowa-strona/')
assert status == 404 and 'piña colad' not in html and 'XYZ Doohickey' not in html
assert 'noindex,follow' in html
status, html = get('/?mk_api=content&path=przykladowa-strona')
assert 'piña colad' not in html and 'XYZ Doohickey' not in html, 'Draft leaked through content API'
namespace = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
all_urls = []
for path in ['/sitemap.xml', '/page-sitemap.xml', '/post-sitemap.xml']:
    status, xml = get(path); assert status == 200
    root = ET.fromstring(xml)
    urls = [e.text for e in root.findall('.//s:loc', namespace)]
    assert len(urls) == len(set(urls))
    assert all(u.startswith(BASE + '/') for u in urls)
    assert not any(any(word in u for word in ['przykladowa-strona', 'podsumowanie-testowe', 'szukac-dostepne', 'rezultaty-wyszukiwania']) for u in urls)
    if path == '/page-sitemap.xml':
        assert BASE + '/prognoza/' in urls and BASE + '/analizator/' in urls and BASE + '/klimat-roczny/' in urls
    all_urls += urls
    print('XML VERIFIED', path, len(urls), flush=True)
_, robots = get('/robots.txt')
assert 'Sitemap: ' + BASE + '/sitemap.xml' in robots and 'sitemap.rss' not in robots
print('Public SEO verification passed.', flush=True)
