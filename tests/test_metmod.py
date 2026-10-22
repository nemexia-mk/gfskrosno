import csv
import io
import unittest
from datetime import date
from unittest.mock import Mock, patch
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
    def response(self, content):
        response = Mock(status_code=200, content=content.encode())
        response.raise_for_status.return_value = None
        return response
    def test_batch_fetches_seven_days_in_one_request(self):
        days = [date(2026, 10, d) for d in range(14, 21)]
        page = html().replace('</table>', ''.join(
            html().split('</tr>')[1].replace('2026-10-20', day.isoformat()) + '</tr>'
            for day in days[:-1]) + '</table>')
        client = Mock(); client.get.return_value = self.response(page)
        updates, missing = m.fetch_days(client, days)
        self.assertEqual(len(updates), 7); self.assertEqual(missing, [])
        self.assertEqual(client.get.call_count, 1)
        self.assertEqual(client.get.call_args.kwargs['params']['dni'], 7)
    @patch.object(m.time, 'sleep')
    def test_http_200_without_table_is_retried(self, sleep):
        client = Mock(); client.get.side_effect = [self.response('<html>Chwilowy brak tabeli</html>'), self.response(html())]
        updates, missing = m.fetch_days(client, [DAY])
        self.assertEqual(missing, []); self.assertIn(DAY, updates)
        self.assertEqual(client.get.call_count, 2); sleep.assert_called_once_with(5)
    @patch.object(m.time, 'sleep')
    def test_transient_network_error_is_retried(self, sleep):
        client = Mock(); client.get.side_effect = [m.requests.Timeout('timeout'), self.response(html())]
        updates, missing = m.fetch_days(client, [DAY])
        self.assertEqual(missing, []); self.assertIn(DAY, updates)
    @patch.object(m.time, 'sleep')
    def test_partial_response_preserves_successful_days(self, sleep):
        other = date(2026, 10, 19)
        client = Mock(); client.get.side_effect = [self.response(html()), self.response(html().replace('2026-10-20', other.isoformat()))]
        updates, missing = m.fetch_days(client, [other, DAY])
        self.assertEqual(set(updates), {other, DAY}); self.assertEqual(missing, [])
    @patch.object(m.time, 'sleep')
    def test_missing_day_reported_after_bounded_retries(self, sleep):
        other = date(2026, 10, 19)
        client = Mock(); client.get.return_value = self.response(html())
        updates, missing = m.fetch_days(client, [other, DAY])
        self.assertEqual(set(updates), {DAY}); self.assertEqual(missing, [other])
        self.assertEqual(client.get.call_count, 3); self.assertEqual(sleep.call_count, 2)
    @patch.object(m, 'publish')
    @patch.object(m, 'fetch_days', return_value=({DAY: {9: '0'}}, [date(2026, 10, 19)]))
    def test_main_reports_incomplete_older_day(self, fetch, publish):
        with patch('sys.argv', ['metmod.py', '--date', DAY.isoformat(), '--lookback', '2', '--dry-run']):
            self.assertEqual(m.main(), 1)
        publish.assert_not_called()
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
