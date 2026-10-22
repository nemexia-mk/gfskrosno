# GFS Krosno: rozszerzona prognoza (22.10.2026)

`gfs-tab.csv` zachowuje dotychczasowe 24 kolumny, kolejność, jednostki, UTC i zaokrąglenia. Zmiana schematu przerywa publikację. Nowe pola są oddzielone w `gfs-extra.json`, `gfs-extra.csv` oraz archiwach `gfs_extra_YYYY_MM_DD_HH.*`.

Rozszerzenia: wilgotność, temperatura odczuwalna wyliczona przez MetPy (wind chill/heat index), temperatura mokra, CIN, PWAT, opad konwekcyjny, modelowa odbiciowość, flagi deszczu/śniegu/opadu marznącego/ziaren lodowych, udział fazy zamarzniętej, wiatr 80/100 m, izoterma 0°C i pułap chmur nad gruntem, temperatura gleby i powierzchni oraz promieniowanie. Wskaźniki z profilu obejmują uskoki 0–1/3/6 km, K Index, Total Totals, a przy sygnale chwiejności również obliczenia parcelowe MUCAPE/DCAPE i SRH. Braki pozostają puste/null; nie tworzymy zastępczego DCAPE ani liczby błysków z CAPE. Usłonecznienie za krok jest zapisywane tylko przy jednoznacznym przedziale akumulacji GRIB.

## Potencjał burzowy

To autorska heurystyka środowiska GFS, nie skalibrowane prawdopodobieństwo i nie prognoza liczby wyładowań. Prezentację kółkami inspirowano ICM; znaczenie liczb ICM nie zostało skopiowane.

- Brak CAPE lub CIN: brak danych.
- CAPE <100 J/kg lub |CIN| >200 J/kg: poziom 0.
- Bez sygnału aktywności: białe kółko, poziom 1.
- Aktywność: opad konwekcyjny >=0,05 mm/h, odbiciowość >=20 dBZ lub opad >=0,2 mm/h przy |CIN| <=50 J/kg.
- Przy aktywności: żółty poziom 2; CAPE >=700 i |CIN| <=100 daje pomarańczowy 3; CAPE >=1500 i |CIN| <=50 daje czerwony 4.

Strona łączy rozszerzenia z bazową prognozą wyłącznie po zgodnym runie, terminie UTC i wyprzedzeniu. Brak rozszerzeń nie blokuje podstawowej prognozy. Stare kolumny gfs-conv.csv zachowano, a nieuzasadnione procenty tornad/derecho, liczba błysków i kategorie przypisywane ESTOFEX pozostają puste.

## Uruchamianie

Wymagane: requests, numpy, pandas, matplotlib, xlsxwriter, python-dotenv, eccodes, metpy. MetPy jest importowane przed ecCodes z powodu zgodności bibliotek natywnych.

```sh
python -m unittest discover -s tests -p test_gfs_weather.py -v
node tests/test_weather_frontend.cjs
python gfs_krosno.py --run 2026102212 --no-ftp --skip-previous
python gfs_conv.py --run 2026102212 --reuse-gribs --no-ftp
```

`--hours 0,1,2,3,6,12,24,72,120,123,384` ogranicza próbę. Produkcja używa wszystkich 209 terminów. Oba harmonogramy GFS wywołują wspólny workflow z blokadą równoległej publikacji. ICON i ECMWF pozostają oddzielne.

## Wdrożenie i odzyskiwanie

Workflow `weather-deploy.yml` najpierw testuje i generuje pełną prognozę bez FTP. Następnie wykonuje i sprawdza bajtowo kopię plików pod `/meteo-krosno.pl/wp-content/themes/meteo-krosno-modern/cache/weather-backups/<czas-UTC>-<identyfikator>/`, z blokadą dostępu HTTP sprawdzaną przed zapisaniem kopii. Manifest zawiera ścieżki i SHA256. Dodatkowo zapisuje oryginały obu skryptów z GitHuba. Po kopii zapisuje skrypty, dane i kod strony; końcowa kontrola porównuje bajty i schemat CSV. Oryginały repozytorium są też w `deployment/baseline/`.

Do wycofania użyj ścieżek z manifestu i odpowiedniej kopii przez FTP; pliki oznaczone `backup: null` nie istniały przed wdrożeniem. Danych dostępowych nie zapisujemy w kodzie. Workflow korzysta z istniejącego sekretu `FTP_PASS`.
