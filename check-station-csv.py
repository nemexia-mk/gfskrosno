import csv, io, json, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path
out=Path("fixtures");out.mkdir()
for station,sub,temp,rh in [("turaszowka","","tempf","humidity"),("lesniowka","lesniowka/","tf_ch1","humidity1")]:
    url="https://meteo-krosno.pl/wp-content/plugins/mk-station-dashboards/station-api.php?station="+station+"&range=6h&resolution=3600"
    req=urllib.request.Request(url,headers={"User-Agent":"MeteoKrosnoAndroid/0.8.0"})
    with urllib.request.urlopen(req,timeout=45) as r:
        assert r.status==200
        data=json.load(r)
    latest=data["latest"];at=latest["ts"];age=int(time.time())-at
    assert data["station"]==station and -60<=age<=900,(station,age)
    assert latest["humidity"] is not None and 0<=latest["humidity"]<=100
    rawurl="http://stacja.meteo-krosno.pl/pwsWD/ecowitt/"+sub+"dzis_"+station+".csv"
    with urllib.request.urlopen(rawurl,timeout=30) as r: raw=r.read().decode("utf-8-sig")
    rows=list(csv.DictReader(io.StringIO(raw),delimiter=";"))
    row=next((v for v in rows if int(datetime.strptime(v["czas_UTC"],"%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())==at),None)
    assert row is not None,"API record missing in daily CSV"
    expected={"temp":(float(row[temp])-32)*5/9,"humidity":float(row[rh]),"pressure":float(row["baromrelin"])*33.8638866667,"wind":float(row["windspeedmph"])*1.609344,"gust":float(row["windgustmph"])*1.609344,"rain":float(row["dailyrainin"])*25.4,"rainrate":float(row["rainratein"])*25.4}
    for field,value in expected.items():
        assert abs(latest[field]-value)<0.011,(station,field,latest[field],value)
    (out/(station+".json")).write_text(json.dumps(data))
    print(json.dumps({"station":station,"ageSeconds":age,"humidity":latest["humidity"],"csvMatches":list(expected),"status":data.get("status")},ensure_ascii=False))
