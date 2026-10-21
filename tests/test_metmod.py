import csv
import io
import unittest
from datetime import date
from unittest.mock import patch
import ftplib
import metmod as m

DAY = date(2026, 10, 20)

def archive():
    row = [''] * 64
    row[:10] = [m.STATION, 'KROSNO'.ljust(30), '2026', '10', '20', '14,9', '7,6', '10,2', '8', '0,0']
    row[20] = 'historyczne pole'
    s = io.StringIO(); csv.writer(s, delimiter=';', lineterminator='\r\n').writerow(row)
    return s.getvalue().encode('cp1250')


def html():
    heads = ['<th>Data</th>'] + [f'<th title="{v[0]}"></th>' for v in m.FIELDS.values()]
    vals = ['2026-10-20', '14.9', '7.6', '10.2', '8', '0.0', '-', '6.2', '2.6', '-', '1023.3']
    return '<table id="tablepl"><tr>' + ''.join(heads) + '</tr><tr>' + ''.join('<td>'+v+'</td>' for v in vals) + '</tr></table>'


class FakeFTP:
    def __init__(self, *a, **kw):
        self.files = {m.FILE_NAME: archive()}; self.renamed = False
    def connect(self, *a): pass
    def login(self, *a): pass
    def cwd(self, *a): pass
    def mkd(self, *a): pass
    def rmd(self, *a): pass
    def retrbinary(self, cmd, callback): callback(self.files[cmd[5:]])
    def storbinary(self, cmd, data): self.files[cmd[5:]] = data.read()
    def rename(self, old, new): self.files[new] = self.files.pop(old); self.renamed = True
    def delete(self, name):
        if name not in self.files: raise ftplib.error_perm('550 missing')
        del self.files[name]
    def quit(self): pass
    def close(self): pass


class MetmodTests(unittest.TestCase):
    def test_real_header_mapping_and_zero(self):
        values = m.parse_day(html(), DAY)
        self.assertEqual(values[9], '0,0')
        self.assertEqual(values[12], '')
        self.assertNotIn(50, values)
        self.assertEqual(values[16], '6,2')
    def test_wrong_day_rejected(self):
        with self.assertRaises(ValueError): m.parse_day(html(), date(2026,10,19))
    def test_changed_header_rejected(self):
        with self.assertRaises(ValueError): m.parse_day(html().replace('Opad dobowy', 'Inny opad'), DAY)
    def test_invalid_temperature_rejected(self):
        with self.assertRaises(ValueError): m.parse_day(html().replace('<td>7.6</td>', '<td>27.6</td>'), DAY)
    def test_missing_never_erases_and_idempotent(self):
        result = m.merge_csv(archive(), {DAY: {9:'', 16:'0,0'}})
        row = next(csv.reader(io.StringIO(result.decode('cp1250')), delimiter=';'))
        self.assertEqual(row[9], '0,0'); self.assertEqual(row[16], '0,0')
        self.assertEqual(row[20], 'historyczne pole')
        self.assertEqual(result, m.merge_csv(result, {DAY:{9:'',16:'0,0'}}))
    def test_duplicates_collapsed(self):
        result = m.merge_csv(archive()+archive(), {DAY:{16:'6,2'}})
        self.assertEqual(len(list(csv.reader(io.StringIO(result.decode()),delimiter=';'))),1)
    def test_empty_or_truncated_csv_rejected(self):
        for raw in (b'', b'349210670;KROSNO;2026;10;20\n'):
            with self.assertRaises(ValueError): m.merge_csv(raw,{DAY:{9:'0'}})
    def test_publish_backup_and_rename(self):
        ftp = FakeFTP()
        with patch.dict('os.environ',{'FTP_USER':'test','FTP_PASS':'test'}), patch.object(m.ftplib,'FTP',return_value=ftp):
            m.publish({DAY:{16:'6,2'}})
        self.assertTrue(ftp.renamed)
        self.assertTrue(any(k.endswith('.bak') for k in ftp.files))
    def test_failed_download_preserves_archive(self):
        ftp = FakeFTP(); before = ftp.files[m.FILE_NAME]
        ftp.retrbinary = lambda *a: (_ for _ in ()).throw(ftplib.error_perm('550 forbidden'))
        with patch.dict('os.environ',{'FTP_USER':'test','FTP_PASS':'test'}), patch.object(m.ftplib,'FTP',return_value=ftp):
            with self.assertRaises(ftplib.error_perm): m.publish({DAY:{16:'6,2'}})
        self.assertEqual(ftp.files, {m.FILE_NAME:before})
    def test_partial_upload_never_renames(self):
        ftp = FakeFTP(); before = ftp.files[m.FILE_NAME]
        original = ftp.storbinary
        def upload(cmd, data):
            original(cmd, data)
            if cmd.endswith('.upload'): ftp.files[cmd[5:]] = b'partial'
        ftp.storbinary = upload
        with patch.dict('os.environ',{'FTP_USER':'test','FTP_PASS':'test'}), patch.object(m.ftplib,'FTP',return_value=ftp):
            with self.assertRaises(ValueError): m.publish({DAY:{16:'6,2'}})
        self.assertFalse(ftp.renamed); self.assertEqual(ftp.files[m.FILE_NAME],before)

if __name__ == '__main__': unittest.main()
