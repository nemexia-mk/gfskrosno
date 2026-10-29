<?php
/**
 * Plugin Name: Meteo Krosno – dobowa sprawdzalność GFS względem IMGW
 * Version: 1.0.0
 */
if (!defined('ABSPATH')) exit;

function mkiv_valid($value, $rain = false) {
    return is_numeric($value) && is_finite((float) $value) && $value >= ($rain ? 0 : -90) && $value <= ($rain ? 1000 : 60);
}

/** Keep daily forecasts without a rolling history limit; never backfill a forecast from observations. */
function mkiv_archive_rows($run, $saved, $temperature, $rain) {
    if ($run <= 0 || $saved < $run || $saved - $run > 129600) return;
    if (get_option('mkiv_archive_lock', time()) < time() - 60) delete_option('mkiv_archive_lock');
    if (!add_option('mkiv_archive_lock', time(), '', 'no')) return;
    try {
    $books = []; $changed = [];
    $first = (int) (floor($run / DAY_IN_SECONDS) * DAY_IN_SECONDS);
    for ($midnight = $first; $midnight <= $run + 72 * 3600; $midnight += DAY_IN_SECONDS) {
        $date = gmdate('Y-m-d', $midnight); $year = gmdate('Y', $midnight);
        foreach (['temp', 'rain'] as $metric) {
            $start = $midnight + ($metric === 'rain' ? 6 * 3600 : 0);
            $end = $start + DAY_IN_SECONDS; $lead = ($start - $run) / 3600;
            if ($lead < 6 || $lead > 72 || $saved > $start - 6 * 3600 || $end > $run + 96 * 3600) continue;
            $values = [];
            $step = $metric === 'temp' ? 3 * 3600 : 3600;
            $rows = $metric === 'temp' ? $temperature : $rain;
            $from = $metric === 'temp' ? $start : $start + 3600;
            $through = $metric === 'temp' ? $end - $step : $end;
            for ($time = $from; $time <= $through; $time += $step) {
                if (!array_key_exists($time, $rows) || !mkiv_valid($rows[$time], $metric === 'rain')) continue 2;
                $values[] = (float) $rows[$time];
            }
            $forecast = $metric === 'temp' ? array_sum($values) / count($values) : array_sum($values);
            $bin = $lead <= 24 ? '6–24' : ($lead <= 48 ? '24–48' : '48–72');
            if (!isset($books[$year])) $books[$year] = get_option('mkiv_forecasts_' . $year, []);
            if (isset($books[$year][$date][$metric][$bin]) && $books[$year][$date][$metric][$bin]['run'] >= $run) continue;
            $books[$year][$date][$metric][$bin] = [
                'forecast' => round($forecast, 4), 'run' => $run, 'saved' => $saved,
                'start' => $start, 'end' => $end, 'lead' => $lead, 'bin' => $bin,
            ];
            $changed[$year] = true;
        }
    }
    foreach ($changed as $year => $unused) update_option('mkiv_forecasts_' . $year, $books[$year], false);
    if ($changed) {
        delete_transient('mkiv_result_temp_v1'); delete_transient('mkiv_result_rain_v1');
    }
    } finally { delete_option('mkiv_archive_lock'); }
}

function mkiv_archive($data, $saved = null) {
    $saved = $saved ?? time();
    if (!is_array($data) || !empty($data['source']['fallback']) || empty($data['run'])) return;
    $run = (int) round($data['run'] / 1000); $temperature = []; $rain = [];
    foreach (($data['rows'] ?? []) as $row) {
        $time = (int) round(($row['time'] ?? 0) / 1000);
        if ($time % 3600 || $time < $run || $time > $run + 96 * 3600) continue;
        if (mkiv_valid($row['temp'] ?? null)) $temperature[$time] = (float) $row['temp'];
        if ($time > $run && mkiv_valid($row['rain'] ?? null, true)) $rain[$time] = (float) $row['rain'];
    }
    mkiv_archive_rows($run, $saved, $temperature, $rain);
}

// Reuse the originally saved forecasts, including their original save time.
add_action('init', function () {
    if (get_option('mkiv_imported_v1')) return;
    foreach (get_option('mkv_runs', []) as $run) {
        $record = get_option('mkv_run_' . $run, false);
        if ($record) mkiv_archive_rows((int) $run, (int) $record['saved'], $record['rows'], []);
    }
    foreach (get_option('mkrv_runs', []) as $run) {
        $record = get_option('mkrv_run_' . $run, false);
        if ($record) mkiv_archive_rows((int) $run, (int) $record['saved'], [], $record['rows']);
    }
    update_option('mkiv_imported_v1', 1, false);
}, 30);
add_action('mkrv_collect', function () {
    try { if (function_exists('meteo_gfs')) mkiv_archive(meteo_gfs()); }
    catch (Throwable $error) { error_log('Meteo Krosno: daily forecast collection failed.'); }
}, 20);
add_action('template_redirect', function () {
    if (($_GET['mk_api'] ?? '') !== 'gfs') return;
    ob_start(function ($body) {
        try { $data = json_decode($body, true); if (is_array($data)) mkiv_archive($data); }
        catch (Throwable $error) { error_log('Meteo Krosno: daily forecast capture failed.'); }
        return $body;
    });
}, -50);

function mkiv_statistics($pairs, $metric) {
    $stats = [];
    foreach (['6–24', '24–48', '48–72'] as $bin) {
        $count = 0; $abs = 0; $square = 0; $bias = 0; $within2 = 0;
        $hits = 0; $misses = 0; $falseAlarms = 0;
        foreach ($pairs as $pair) {
            if ($pair['bin'] !== $bin) continue;
            $count++; $error = $pair['forecast'] - $pair['observed'];
            $abs += abs($error); $square += $error * $error; $bias += $error;
            if (abs($error) <= 2) $within2++;
            if ($pair['forecast'] >= 1 && $pair['observed'] >= 1) $hits++;
            elseif ($pair['observed'] >= 1) $misses++;
            elseif ($pair['forecast'] >= 1) $falseAlarms++;
        }
        $stat = [
            'bin' => $bin, 'hasData' => $count > 0, 'preliminary' => $count > 0 && $count < 30,
            'mae' => $count ? $abs / $count : null, 'rmse' => $count ? sqrt($square / $count) : null,
            'bias' => $count ? $bias / $count : null,
        ];
        if ($metric === 'rain') {
            $stat['pod'] = $hits + $misses ? 100 * $hits / ($hits + $misses) : null;
            $stat['far'] = $hits + $falseAlarms ? 100 * $falseAlarms / ($hits + $falseAlarms) : null;
            $stat['csi'] = $hits + $misses + $falseAlarms ? 100 * $hits / ($hits + $misses + $falseAlarms) : null;
        } else $stat['within2'] = $count ? 100 * $within2 / $count : null;
        $stats[] = $stat;
    }
    return $stats;
}

/** Read every available observation; publish aggregates without archive dates or counts. */
function mkiv_results($climate, $metric, $now = null) {
    $now = $now ?? time(); $metric = $metric === 'rain' ? 'rain' : 'temp';
    $fields = array_column($climate['fields'] ?? [], 0);
    $column = array_search($metric === 'rain' ? 'rr' : 'tm', $fields, true);
    if ($column === false) throw new RuntimeException('Missing observation field');
    $column++; // Each normalized row begins with its date, followed by fields.
    $observations = [];
    foreach (($climate['rows'] ?? []) as $row) {
        $date = $row[0] ?? '';
        $day = DateTimeImmutable::createFromFormat('!Y-m-d', $date, new DateTimeZone('UTC'));
        if (!$day || $day->format('Y-m-d') !== $date || !mkiv_valid($row[$column] ?? null, $metric === 'rain')) continue;
        $start = $day->getTimestamp() + ($metric === 'rain' ? 6 * 3600 : 0);
        if ($start + DAY_IN_SECONDS > $now) continue;
        $observations[$date] = (float) $row[$column];
    }
    $books = []; $pairs = [];
    foreach ($observations as $date => $observed) {
        $year = substr($date, 0, 4);
        if (!isset($books[$year])) $books[$year] = get_option('mkiv_forecasts_' . $year, []);
        foreach (($books[$year][$date][$metric] ?? []) as $forecast) {
            $expectedStart = strtotime($date . ' 00:00:00 UTC') + ($metric === 'rain' ? 6 * 3600 : 0);
            if ($forecast['start'] !== $expectedStart || $forecast['end'] !== $expectedStart + DAY_IN_SECONDS || $forecast['saved'] > $forecast['start'] - 6 * 3600 || $forecast['end'] > $now) continue;
            $pairs[] = ['forecast' => $forecast['forecast'], 'observed' => $observed, 'bin' => $forecast['bin']];
        }
    }
    return [
        'station' => 'imgw', 'stationName' => 'Krosno IMGW', 'metric' => $metric,
        'aggregation' => 'daily', 'available' => (bool) $pairs, 'stats' => mkiv_statistics($pairs, $metric),
    ];
}

add_action('template_redirect', function () {
    if (($_GET['mk_api'] ?? '') !== 'verification' || ($_GET['station'] ?? 'imgw') !== 'imgw') return;
    nocache_headers(); header('Content-Type: application/json; charset=UTF-8'); header('X-Robots-Tag: noindex, nofollow');
    try {
        $metric = ($_GET['metric'] ?? '') === 'rain' ? 'rain' : 'temp';
        $key = 'mkiv_result_' . $metric . '_v1'; $result = get_transient($key);
        if (!is_array($result)) {
            $result = mkiv_results(meteo_climate(), $metric);
            set_transient($key, $result, 300);
        }
        echo wp_json_encode($result);
    } catch (Throwable $error) {
        status_header(503); echo wp_json_encode(['error' => 'Dane są chwilowo niedostępne.']);
    }
    exit;
}, -50);
