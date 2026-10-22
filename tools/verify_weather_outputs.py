import json
from pathlib import Path
import pandas as pd

paths=list(Path('gfs_krosno_full').glob('gfs_extra_*.json'))
assert paths, 'Brak rozszerzeń'
extra=json.loads(paths[-1].read_text())
rows=extra['rows']; assert rows
for key in ('humidity','cin','pwat','convRain','reflectivity','solar','wind100','shear06','wetBulb'):
    known=sum(r.get(key) is not None for r in rows)
    print('FIELD',key,known,'/',len(rows))
    assert known, 'Brak pola '+key
assert all(r['type'] for r in rows)
conv=pd.read_csv('gfs_krosno_conv/gfs-conv.csv')
assert len(conv)==len(rows)
assert conv['Prob Tornado [%]'].isna().all()
print('VERIFIED',extra['run'],len(rows),'terminów GFS i konwekcji')
