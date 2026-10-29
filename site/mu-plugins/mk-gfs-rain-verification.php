<?php
/**
 * Plugin Name: Meteo Krosno – sprawdzalność opadów GFS
 * Description: Immutable forecasts and complete six-hour precipitation comparisons.
 * Version: 1.0.0
 */
if (!defined('ABSPATH')) exit;

function mkrv_number($value) {
    return is_numeric($value) && is_finite((float) $value) && (float) $value >= 0;
}

function mkrv_capture_forecast($data, $now = null) {
    $now = $now ?? time();
    if (!is_array($data) || !empty($data['source']['fallback']) || empty($data['run'])) return;
    $run = (int) round($data['run'] / 1000);
    if ($run > $now || $now - $run > 129600) return;
    $key = 'mkrv_run_' . $run;
    if (get_option($key, false) !== false) return; // Never revise a forecast after observing the weather.
    $rows = [];
    foreach (($data['rows'] ?? []) as $row) {
        $end = (int) round(($row['time'] ?? 0) / 1000);
        if ($end % 3600 || $end <= $now || $end <= $run || $end > $run + 72 * 3600 || !mkrv_number($row['rain'] ?? null)) continue;
        $rows[$end] = (float) $row['rain']; // GFS RRR is the preceding hourly interval, not a running total.
    }
    if (!$rows) return;
    ksort($rows);
    if (!add_option($key, ['run' => $run, 'saved' => $now, 'rows' => $rows], '', 'no')) return;
    $index = get_option('mkrv_runs', []);
    $index[$run] = $run;
    foreach ($index as $old) {
        if ($old < $now - 95 * DAY_IN_SECONDS) {
            delete_option('mkrv_run_' . $old);
            unset($index[$old]);
        }
    }
    ksort($index);
    update_option('mkrv_runs', $index, false);
    add_option('mkrv_started', $now, '', 'no');
}

function mkrv_capture_observations($data, $now = null) {
    $now = $now ?? time();
    foreach (($data['stations'] ?? []) as $station) {
        $id = $station['id'] ?? '';
        if (!in_array($id, ['turaszowka', 'lesniowka'], true) || empty($station['available']) || !empty($station['fallback'])) continue;
        // A monthly counter avoids the daily rain-counter reset. Its own month boundary is excluded.
        $at = (int) ($station['timestamps']['monthly'] ?? 0);
        $hour = (int) (round($at / 3600) * 3600);
        if (!$at || $at > $now + 60 || $now - $at > 300 || abs($at - $hour) > 300 || !mkrv_number($station['monthly'] ?? null)) continue;
        $key = 'mkrv_obs_' . $id;
        $observations = get_option($key, []);
        if (isset($observations[$hour]) && abs($observations[$hour]['at'] - $hour) <= abs($at - $hour)) continue;
        $observations[$hour] = [
            'at' => $at, 'total' => (float) $station['monthly'],
            'month' => (new DateTimeImmutable('@' . $at))->setTimezone(new DateTimeZone('Europe/Warsaw'))->format('Y-m'),
        ];
        foreach ($observations as $time => $value) if ($time < $now - 90 * DAY_IN_SECONDS) unset($observations[$time]);
        ksort($observations);
        update_option($key, $observations, false);
    }
}

function mkrv_collect() {
    try { if (function_exists('meteo_gfs')) mkrv_capture_forecast(meteo_gfs()); } catch (Throwable $error) { error_log('Meteo Krosno: rain forecast collection failed.'); }
    try { if (function_exists('meteo_live')) mkrv_capture_observations(meteo_live()); } catch (Throwable $error) { error_log('Meteo Krosno: rain observation collection failed.'); }
}
add_filter('cron_schedules', function ($schedules) {
    $schedules['mkrv_five_minutes'] = ['interval' => 300, 'display' => 'Meteo Krosno: co 5 minut'];
    return $schedules;
});
add_action('init', function () {
    if (!wp_next_scheduled('mkrv_collect')) wp_schedule_event(time(), 'mkrv_five_minutes', 'mkrv_collect');
});
add_action('mkrv_collect', 'mkrv_collect');

function mkrv_observed_sum($observations, $start, $end) {
    $previous = null;
    for ($hour = $start; $hour <= $end; $hour += 3600) {
        if (!isset($observations[$hour])) return null;
        $point = $observations[$hour];
        if (!mkrv_number($point['total'] ?? null) || abs(($point['at'] ?? 0) - $hour) > 300) return null;
        if ($previous !== null && ($point['month'] !== $previous['month'] || $point['total'] < $previous['total'])) return null;
        $previous = $point;
    }
    return round($observations[$end]['total'] - $observations[$start]['total'], 4);
}

function mkrv_stats($pairs) {
    $stats = [];
    foreach (['6–24', '24–48', '48–72'] as $bin) {
        $count = 0; $abs = 0; $square = 0; $bias = 0;
        $hits = 0; $misses = 0; $falseAlarms = 0; $dry = 0;
        foreach ($pairs as $pair) {
            if ($pair['bin'] !== $bin) continue;
            $count++; $error = $pair['error'];
            $abs += abs($error); $square += $error * $error; $bias += $error;
            $forecastWet = $pair['forecast'] >= 1; $observedWet = $pair['observed'] >= 1;
            if ($forecastWet && $observedWet) $hits++;
            elseif ($observedWet) $misses++;
            elseif ($forecastWet) $falseAlarms++;
            else $dry++;
        }
        $stats[] = [
            'bin' => $bin, 'n' => $count,
            'mae' => $count ? $abs / $count : null, 'rmse' => $count ? sqrt($square / $count) : null,
            'bias' => $count ? $bias / $count : null,
            'hits' => $hits, 'misses' => $misses, 'falseAlarms' => $falseAlarms, 'correctDry' => $dry,
            'pod' => $hits + $misses ? 100 * $hits / ($hits + $misses) : null,
            'far' => $hits + $falseAlarms ? 100 * $falseAlarms / ($hits + $falseAlarms) : null,
            'csi' => $hits + $misses + $falseAlarms ? 100 * $hits / ($hits + $misses + $falseAlarms) : null,
        ];
    }
    return $stats;
}

function mkrv_results($id, $days, $now = null) {
    $now = $now ?? time();
    if (!in_array($id, ['turaszowka', 'lesniowka'], true)) $id = 'turaszowka';
    if (!in_array($days, [7, 30, 90], true)) $days = 30;
    $index = get_option('mkrv_runs', []);
    $observations = get_option('mkrv_obs_' . $id, []);
    $pairs = [];
    foreach ($index as $run) {
        if ($run < $now - ($days + 4) * DAY_IN_SECONDS) continue;
        $forecast = get_option('mkrv_run_' . $run, false);
        if (!$forecast) continue;
        foreach ($forecast['rows'] as $time => $unused) {
            $end = (int) $time; $start = $end - 6 * 3600;
            if ($end % (6 * 3600) || $start < $now - $days * DAY_IN_SECONDS || $end > $now || $forecast['saved'] > $start - 6 * 3600) continue;
            $lead = ($start - $run) / 3600;
            if ($lead < 6 || $lead > 72) continue;
            $sum = 0;
            for ($hour = $start + 3600; $hour <= $end; $hour += 3600) {
                if (!isset($forecast['rows'][$hour]) || !mkrv_number($forecast['rows'][$hour])) continue 2;
                $sum += $forecast['rows'][$hour];
            }
            $observed = mkrv_observed_sum($observations, $start, $end);
            if ($observed === null || $observations[$end]['at'] > $now) continue;
            $bin = $lead <= 24 ? '6–24' : ($lead <= 48 ? '24–48' : '48–72');
            $key = $end . '_' . $bin;
            if (isset($pairs[$key]) && $pairs[$key]['run'] >= $run) continue;
            $pairs[$key] = [
                'start' => $start * 1000, 'time' => $end * 1000,
                'observedStartAt' => $observations[$start]['at'] * 1000, 'observedAt' => $observations[$end]['at'] * 1000,
                'forecast' => round($sum, 4), 'observed' => $observed, 'error' => round($sum - $observed, 4),
                'run' => (int) $run, 'saved' => $forecast['saved'], 'lead' => $lead, 'bin' => $bin,
            ];
        }
    }
    $pairs = array_values($pairs);
    usort($pairs, function ($a, $b) { return ($b['time'] <=> $a['time']) ?: ($a['lead'] <=> $b['lead']); });
    return [
        'metric' => 'rain', 'intervalHours' => 6, 'wetThreshold' => 1,
        'station' => $id, 'days' => $days, 'started' => get_option('mkrv_started', null),
        'runs' => count($index), 'observations' => count($observations),
        'pairs' => $pairs, 'stats' => mkrv_stats($pairs), 'nextCollection' => wp_next_scheduled('mkrv_collect') ?: null,
    ];
}

add_action('template_redirect', function () {
    if (($_GET['mk_api'] ?? '') !== 'verification' || ($_GET['metric'] ?? '') !== 'rain') return;
    nocache_headers();
    header('Content-Type: application/json; charset=UTF-8');
    header('X-Robots-Tag: noindex, nofollow');
    echo wp_json_encode(mkrv_results(sanitize_key($_GET['station'] ?? 'turaszowka'), (int) ($_GET['days'] ?? 30)));
    exit;
}, -30);
