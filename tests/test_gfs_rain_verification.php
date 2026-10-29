<?php
define('ABSPATH', __DIR__); define('DAY_IN_SECONDS', 86400);
$options = [];
function get_option($key, $default = false) { return $GLOBALS['options'][$key] ?? $default; }
function add_option($key, $value, ...$unused) { if (array_key_exists($key, $GLOBALS['options'])) return false; $GLOBALS['options'][$key] = $value; return true; }
function update_option($key, $value, ...$unused) { $GLOBALS['options'][$key] = $value; }
function delete_option($key) { unset($GLOBALS['options'][$key]); }
function add_action(...$unused) {} function add_filter(...$unused) {} function wp_next_scheduled(...$unused) { return 123; }
require __DIR__ . '/../site/mu-plugins/mk-gfs-rain-verification.php';
function check($condition, $message) { if (!$condition) throw new RuntimeException($message); }
$run = strtotime('2026-10-20 00:00:00 UTC'); $start = $run + 12 * 3600; $end = $start + 21600;
$data = ['run' => $run * 1000, 'source' => ['fallback' => false], 'rows' => []];
for ($h = 0; $h <= 72; $h++) $data['rows'][] = ['time' => ($run + $h * 3600) * 1000, 'rain' => $h === 0 ? 100 : 0.5];
mkrv_capture_forecast($data, $run + 3600);
$original = get_option('mkrv_run_' . $run);
$changed = $data; $changed['rows'][14]['rain'] = 999;
mkrv_capture_forecast($changed, $run + 7200);
check(get_option('mkrv_run_' . $run) === $original, 'An archived run must be immutable');
check(!isset($original['rows'][$run]), 'Lead zero is not an accumulation interval');
$fallback = $data; $fallback['run'] += 21600000; $fallback['source']['fallback'] = true;
mkrv_capture_forecast($fallback, $run + 21600);
check(count(get_option('mkrv_runs')) === 1, 'Fallback runs must be excluded');
function station($id, $at, $total) { return ['stations' => [['id' => $id, 'available' => true, 'fallback' => false, 'monthly' => $total, 'timestamps' => ['monthly' => $at]]]]; }
for ($h = 0; $h <= 6; $h++) mkrv_capture_observations(station('turaszowka', $start + $h * 3600, 10 + $h / 3), $start + $h * 3600);
// Temperature availability must not control rain availability.
check(count(get_option('mkrv_obs_turaszowka')) === 7, 'Independent rain sensor with seven valid readings');
$v = mkrv_results('turaszowka', 7, $end);
check(count($v['pairs']) === 1, 'Exactly one completed six-hour pair');
$pair = $v['pairs'][0];
check($pair['forecast'] === 3.0 && $pair['observed'] === 2.0 && $pair['error'] === 1.0, 'Six hourly increments vs counter difference');
check($pair['lead'] == 12 && $pair['start'] === $start * 1000, 'Lead is measured to period start');
$obs = get_option('mkrv_obs_turaszowka'); $saved = $obs;
unset($obs[$start + 3600]); update_option('mkrv_obs_turaszowka', $obs);
check(!mkrv_results('turaszowka', 7, $end)['pairs'], 'Missing observations are not zero rainfall');
$obs = $saved; $obs[$start + 3600]['total'] = 0; update_option('mkrv_obs_turaszowka', $obs);
check(!mkrv_results('turaszowka', 7, $end)['pairs'], 'Counter resets invalidate the period');
$obs = $saved; $obs[$start]['month'] = '2026-09'; update_option('mkrv_obs_turaszowka', $obs);
check(!mkrv_results('turaszowka', 7, $end)['pairs'], 'Month boundaries are excluded even without a visible counter drop');
$obs = $saved; $obs[$start]['at'] -= 301; update_option('mkrv_obs_turaszowka', $obs);
check(!mkrv_results('turaszowka', 7, $end)['pairs'], 'Boundary tolerance is enforced');
update_option('mkrv_obs_turaszowka', $saved);
$g = $original; $g['saved'] = $start - 21600 + 1; update_option('mkrv_run_' . $run, $g);
check(!mkrv_results('turaszowka', 7, $end)['pairs'], 'No hindsight: saved at least six hours before start');
$g = $original; unset($g['rows'][$start + 3600]); update_option('mkrv_run_' . $run, $g);
check(!mkrv_results('turaszowka', 7, $end)['pairs'], 'Missing forecast hours are excluded');
update_option('mkrv_run_' . $run, $original);
// A newer qualifying run replaces a pair in the same bin, but cannot duplicate it.
$newRun = $run + 6 * 3600; $new = $original; $new['run'] = $newRun; $new['saved'] = $start - 6 * 3600;
update_option('mkrv_run_' . $newRun, $new); update_option('mkrv_runs', [$run => $run, $newRun => $newRun]);
$v = mkrv_results('turaszowka', 7, $end);
check(count($v['pairs']) === 1 && $v['pairs'][0]['run'] === $newRun, 'Newest qualifying run per bin and period');
check(!mkrv_results('lesniowka', 7, $end)['pairs'], 'Stations are independent');
$before = get_option('mkrv_obs_turaszowka');
$bad = station('turaszowka', $end + 3600, 22); $bad['stations'][0]['fallback'] = true;
mkrv_capture_observations($bad, $end + 3600);
mkrv_capture_observations(station('turaszowka', $end + 3600, null), $end + 3600);
mkrv_capture_observations(station('turaszowka', $end + 3600, 22), $end + 4000);
check(get_option('mkrv_obs_turaszowka') === $before, 'Fallback, missing and stale counters are excluded');
$ps = [];
foreach ([[2,2],[2,0],[0,2],[0,0]] as $p) $ps[] = ['bin' => '6–24', 'forecast' => $p[0], 'observed' => $p[1], 'error' => $p[0] - $p[1]];
$stats = mkrv_stats($ps)[0];
check($stats['n'] === 4 && $stats['mae'] == 1 && $stats['bias'] == 0, 'Amount error metrics');
check($stats['hits'] === 1 && $stats['misses'] === 1 && $stats['falseAlarms'] === 1 && $stats['correctDry'] === 1, 'Event contingency counts');
check($stats['pod'] == 50 && $stats['far'] == 50 && abs($stats['csi'] - 100 / 3) < 0.001, 'POD, FAR and CSI');
$dry = mkrv_stats([['bin' => '6–24', 'forecast' => 0, 'observed' => 0, 'error' => 0]])[0];
check($dry['mae'] == 0 && $dry['csi'] === null && $dry['pod'] === null && $dry['far'] === null, 'All dry is not 100 percent rain skill');
$body = json_encode(station('lesniowka', $end, 0));
check(mkrv_capture_response($body, 'live', $end) === $body, 'Response capture must preserve exact JSON bytes');
check(get_option('mkrv_obs_lesniowka')[$end]['total'] === 0.0, 'A public live response archives a real zero independently of cron');
check(mkrv_capture_response('invalid JSON', 'gfs', $end) === 'invalid JSON', 'Capture must not change upstream errors');
echo "Rain verification: immutable forecasts, complete aligned intervals, counter resets, no hindsight, station isolation and scores OK\n";
