<?php
define('ABSPATH', __DIR__); define('DAY_IN_SECONDS', 86400);
$options = []; $actions = [];
function get_option($key, $default = false) { return $GLOBALS['options'][$key] ?? $default; }
function update_option($key, $value, ...$unused) { $GLOBALS['options'][$key] = $value; }
function add_option($key, $value, ...$unused) { if (array_key_exists($key, $GLOBALS['options'])) return false; $GLOBALS['options'][$key] = $value; return true; }
function delete_option($key) { unset($GLOBALS['options'][$key]); }
function delete_transient($key) {}
function add_action($hook, $callback, ...$unused) { $GLOBALS['actions'][$hook][] = $callback; }
require __DIR__ . '/../site/mu-plugins/mk-gfs-imgw-verification.php';
function check($condition, $message) { if (!$condition) throw new RuntimeException($message); }
$run = strtotime('2026-10-19 00:00 UTC'); $saved = $run + 7200;
$data = ['run' => $run * 1000, 'source' => ['fallback' => false], 'rows' => []];
for ($h = 0; $h <= 96; $h++) $data['rows'][] = ['time' => ($run + 3600 * $h) * 1000, 'temp' => $h % 3 === 0 ? 10 : 20, 'rain' => $h === 0 ? 100 : 0.5];
mkiv_archive($data, $saved);
$book = get_option('mkiv_forecasts_2026');
check($book['2026-10-20']['temp']['6–24']['forecast'] === 10.0, 'Eight SYNOP terms, not a different hourly mean');
check($book['2026-10-20']['rain']['24–48']['forecast'] === 12.0, 'All 24 preceding hourly rain increments from 06 to 06 UTC');
check(!isset($book['2026-10-19']), 'No partially observed day or insufficient advance save');
check(isset($book['2026-10-21']['rain']['48–72']), 'Use a long enough forecast for a full day in the last lead bin');
$copy = $data; $copy['rows'][30]['temp'] = -30; $copy['rows'][31]['rain'] = 99;
mkiv_archive($copy, $saved + 3600);
check(get_option('mkiv_forecasts_2026') === $book, 'Same run cannot revise already archived predictions');
$fields = [['rr', 'Rain', 'mm'], ['tm', 'Mean temperature', 'C']];
$climate = ['fields' => $fields, 'rows' => [['2026-10-20', 14, 9]]];
$temp = mkiv_results($climate, 'temp', $run + 10 * DAY_IN_SECONDS);
check($temp['stats'][0]['mae'] == 1 && $temp['stats'][0]['bias'] == 1 && $temp['available'], 'Pair the published daily mean to the same date');
$rain = mkiv_results($climate, 'rain', $run + 10 * DAY_IN_SECONDS);
check($rain['stats'][1]['mae'] == 2 && $rain['stats'][1]['bias'] == -2 && $rain['stats'][1]['pod'] == 100, 'Daily amount errors and rain scores');
check(!mkiv_results($climate, 'rain', strtotime('2026-10-21 05:59 UTC'))['available'], 'Rain day is unfinished until the following 06 UTC');
$missing = $climate; $missing['rows'][0][1] = null;
check(!mkiv_results($missing, 'rain', $run + 10 * DAY_IN_SECONDS)['available'], 'Missing rain is not zero');
$zero = $climate; $zero['rows'][0][1] = 0;
check(mkiv_results($zero, 'rain', $run + 10 * DAY_IN_SECONDS)['available'], 'Measured zero rain remains valid');
// No arbitrary time cutoff: a legitimate matching old forecast is used too.
$old = $book['2026-10-20'];
foreach ($old as $metric => &$bins) foreach ($bins as &$record) {
    $record['start'] = strtotime('1985-01-01 00:00 UTC') + ($metric === 'rain' ? 21600 : 0);
    $record['end'] = $record['start'] + DAY_IN_SECONDS;
    $record['run'] = $record['start'] - $record['lead'] * 3600;
    $record['saved'] = $record['run'] + 7200;
    if ($metric === 'temp') $record['forecast'] = 20;
}
unset($bins, $record);
update_option('mkiv_forecasts_1985', ['1985-01-01' => $old]);
$long = ['fields' => $fields, 'rows' => [['1985-01-01', 14, 15], ['2026-10-20', 14, 9], ['1986-01-01', 5, 20]]];
$all = mkiv_results($long, 'temp', $run + 10 * DAY_IN_SECONDS);
check($all['stats'][0]['mae'] == 3, 'Scan all observations without a rolling date filter; unmatched days excluded');
// The API publishes no row dates, archive range or comparison counts.
$json = json_encode($all);
foreach (['1985', '2026', 'pairs', 'started', 'observations', '"n"', 'days', '.csv'] as $forbidden) check(strpos($json, $forbidden) === false, 'Aggregate response leaked ' . $forbidden);
$options = [];
$bad = $data; $bad['source']['fallback'] = true;
mkiv_archive($bad, $saved); check(!$options, 'Do not archive substitute forecasts');
$bad = $data; $bad['rows'][31]['rain'] = null;
mkiv_archive($bad, $saved);
check(!isset(get_option('mkiv_forecasts_2026')['2026-10-20']['rain']), 'A missing hour invalidates a rain day');
$options = [];
mkiv_archive($data, strtotime('2026-10-19 18:01 UTC'));
check(!isset(get_option('mkiv_forecasts_2026')['2026-10-20']['temp']), 'Forecast saved too late for start minus six hours');
// Import from the pre-existing immutable temperature archive, preserving save times.
$options = ['mkv_runs' => [$run => $run], 'mkv_run_' . $run => ['saved' => $saved, 'rows' => []]];
foreach ($data['rows'] as $row) $options['mkv_run_' . $run]['rows'][$row['time'] / 1000] = $row['temp'];
foreach ($actions['init'] as $callback) $callback();
check(isset(get_option('mkiv_forecasts_2026')['2026-10-20']['temp']), 'Reuse genuine original forecast records');
$imported = $options;
foreach ($actions['init'] as $callback) $callback();
check($options === $imported, 'Import is idempotent');
echo "IMGW daily verification: aligned means/rain days, complete forecasts, original save times, full observation input, aggregate privacy and migration OK\n";
