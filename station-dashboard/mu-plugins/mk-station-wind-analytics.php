<?php
/**
 * Plugin Name: Meteo Krosno — statystyki wiatru stacji
 * Description: Publiczne statystyki kierunków z archiwum stacji; bez zmian loggera i danych.
 */
if (!defined('ABSPATH')) { return; }

final class MK_Station_Wind_Analytics {
    const CODES = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
    const NAMES = ['północny','północno-północno-wschodni','północno-wschodni','wschodnio-północno-wschodni','wschodni','wschodnio-południowo-wschodni','południowo-wschodni','południowo-południowo-wschodni','południowy','południowo-południowo-zachodni','południowo-zachodni','zachodnio-południowo-zachodni','zachodni','zachodnio-północno-zachodni','północno-zachodni','północno-północno-zachodni'];
    const CALM = 1.8; // 0,5 m/s; prędkości w danych i legendzie podajemy w km/h.
    const BINS = [[1.8,5],[5,10],[10,20],[20,30],[30,null]];

    static function number($value) {
        if ($value === null || trim((string)$value) === '') { return null; }
        $value = str_replace(',', '.', trim((string)$value));
        return is_numeric($value) && is_finite((float)$value) ? (float)$value : null;
    }

    static function record($row) {
        $stamp = null;
        foreach (['czas_UTC','dateutc','timestamp_utc'] as $key) {
            if (!empty($row[$key])) {
                $dt = DateTimeImmutable::createFromFormat('!Y-m-d H:i:s', $row[$key], new DateTimeZone('UTC'));
                if ($dt && $dt->format('Y-m-d H:i:s') === $row[$key]) { $stamp = $dt->getTimestamp(); break; }
            }
        }
        if ($stamp === null) { return null; }
        $speed = null;
        foreach (['wiatr_km_h','wiatr_kmh','wind_kmh'] as $key) {
            $speed = self::number($row[$key] ?? null); if ($speed !== null) { break; }
        }
        if ($speed === null) {
            $mph = self::number($row['windspeedmph'] ?? null);
            if ($mph !== null) { $speed = $mph * 1.609344; }
        }
        $dir = self::number($row['winddir'] ?? $row['kierunek_wiatru_deg'] ?? $row['direction'] ?? null);
        if ($speed !== null && ($speed < 0 || $speed > 400)) { $speed = null; }
        if ($dir !== null && ($dir < 0 || $dir > 360)) { $dir = null; }
        return ['ts'=>$stamp,'wind'=>$speed,'direction'=>$dir];
    }

    static function summarize($rows, $start, $end) {
        $sectors = [];
        foreach (self::CODES as $i=>$code) { $sectors[] = ['code'=>$code,'name'=>self::NAMES[$i],'seconds'=>0.0,'bins'=>array_fill(0,5,0.0)]; }
        $valid = 0.0; $calm = 0.0; $missingDir = 0.0; $samples = 0;
        $count = count($rows);
        foreach ($rows as $i=>$r) {
            $next = $i+1<$count ? $rows[$i+1]['ts'] : $r['ts']+60;
            // Maksymalnie dwie minuty reprezentowane przez próbkę. Luk nie wypełniamy.
            $duration = max(0, min($end,$next,$r['ts']+120)-max($start,$r['ts']));
            if ($duration <= 0 || $r['wind'] === null) { continue; }
            if ($r['wind'] < self::CALM) { $calm += $duration; $valid += $duration; $samples++; continue; }
            if ($r['direction'] === null) { $missingDir += $duration; continue; }
            $sector = ((int)floor((fmod($r['direction'],360)+11.25)/22.5)) % 16;
            $bin = $r['wind']<5?0:($r['wind']<10?1:($r['wind']<20?2:($r['wind']<30?3:4)));
            $sectors[$sector]['seconds'] += $duration;
            $sectors[$sector]['bins'][$bin] += $duration;
            $valid += $duration; $samples++;
        }
        $dominant = null;
        foreach ($sectors as &$sector) {
            $sector['percent'] = $valid ? round($sector['seconds']/$valid*100,2) : null;
            $sector['binPercent'] = array_map(function($s) use ($valid) { return $valid ? round($s/$valid*100,2) : 0; }, $sector['bins']);
            if ($sector['seconds']>0 && ($dominant===null || $sector['seconds']>$dominant['seconds'])) { $dominant = $sector; }
        }
        unset($sector);
        return ['start'=>$start,'end'=>$end,'unit'=>'km/h','calmThreshold'=>self::CALM,
            'bins'=>self::BINS,'sectors'=>$sectors,'dominant'=>$dominant,'validSeconds'=>$valid,
            'calmSeconds'=>$calm,'calmPercent'=>$valid?round($calm/$valid*100,1):null,
            'missingDirectionSeconds'=>$missingDir,'samples'=>$samples,
            'coverage'=>$end>$start?round(min(100,$valid/($end-$start)*100),1):0];
    }

    static function file_rows($file, $first, $last, &$records) {
        $fp = @fopen($file,'rb'); if (!$fp) { return false; }
        $header = fgetcsv($fp,0,';','"','');
        if (!$header) { fclose($fp); return false; }
        $header[0] = preg_replace('/^\xEF\xBB\xBF/','',$header[0]);
        while (($values=fgetcsv($fp,0,';','"','')) !== false) {
            if (count($values)!==count($header)) { continue; }
            $record = self::record(array_combine($header,$values));
            if ($record && $record['ts']>=$first-120 && $record['ts']<=$last) { $records[$record['ts']]=$record; }
        }
        fclose($fp); return true;
    }

    static function respond($request) {
        $station = (string)$request->get_param('station');
        if (!in_array($station,['turaszowka','lesniowka'],true)) { return new WP_Error('station','Nieznana stacja',['status'=>400]); }
        $now = time(); $tz = new DateTimeZone('Europe/Warsaw');
        $today = (new DateTimeImmutable('now',$tz))->setTime(0,0)->getTimestamp();
        $range = (string)$request->get_param('range');
        if (!in_array($range,['6h','24h','7d','day'],true)) { $range='24h'; }
        $end=$now; $start=$now-($range==='6h'?21600:($range==='7d'?604800:86400));
        if ($range==='day') {
            $date = (string)$request->get_param('date');
            $dt = DateTimeImmutable::createFromFormat('!Y-m-d',$date,$tz);
            if (!$dt || $dt->format('Y-m-d')!==$date || $dt->getTimestamp()>$now) { return new WP_Error('date','Nieprawidłowy dzień',['status'=>400]); }
            $start=$dt->getTimestamp(); $end=min($now,$dt->modify('+1 day')->getTimestamp());
        }
        $requestedStart=self::number($request->get_param('start')); $requestedEnd=self::number($request->get_param('end'));
        if ($requestedStart!==null && $requestedEnd!==null) {
            if ($requestedEnd<=$requestedStart || $requestedEnd-$requestedStart>604800 || $requestedStart<946684800 || $requestedStart>$now) {
                return new WP_Error('period','Nieprawidłowy okres',['status'=>400]);
            }
            $start=(int)$requestedStart; $end=(int)min($now,$requestedEnd);
        }
        $key='mk_wind_v1_'.md5($station.'|'.floor($start/60).'|'.floor($end/60));
        $cached=get_transient($key); if (is_array($cached)) { return rest_ensure_response($cached); }
        $first=min($start,$today); $last=max($end,$now);
        $months=[];
        foreach ([[$start,$end],[$today,$now]] as $interval) {
            $cursor=(new DateTimeImmutable('@'.$interval[0]))->setTimezone($tz)->modify('first day of this month')->setTime(0,0);
            while ($cursor->getTimestamp()<=$interval[1]) { $months[$cursor->format('m_Y')]=true; $cursor=$cursor->modify('+1 month'); }
        }
        $names=[$station.'.csv']; foreach (array_keys($months) as $month) { $names[]=$month.'_'.$station.'.csv'; }
        $suffix='/stacja.meteo-krosno.pl/pwsWD/ecowitt/'.($station==='lesniowka'?'lesniowka/':'');
        $base=dirname(rtrim(ABSPATH,'/\\')).$suffix;
        if (!is_dir($base)) { $base=dirname(rtrim(ABSPATH,'/\\'),2).$suffix; }
        $records=[]; $sources=[];
        foreach ($names as $name) {
            $file=$base.$name;
            if (is_readable($file)) { if (self::file_rows($file,$first,$last,$records)) { $sources[]=$name; } }
            elseif (!is_dir($base)) {
                // Ten sam publiczny katalog archiwum; adres nie pochodzi z żądania użytkownika.
                $url='http://stacja.meteo-krosno.pl/pwsWD/ecowitt/'.($station==='lesniowka'?'lesniowka/':'').$name;
                $response=wp_remote_get($url,['timeout'=>12,'redirection'=>0,'limit_response_size'=>33554432]);
                if (!is_wp_error($response) && wp_remote_retrieve_response_code($response)===200) {
                    $body=wp_remote_retrieve_body($response); $fp=fopen('php://temp','w+');fwrite($fp,$body);rewind($fp);
                    $header=fgetcsv($fp,0,';','"','');
                    if ($header) {
                        $header[0]=preg_replace('/^\xEF\xBB\xBF/','',$header[0]);
                        while (($values=fgetcsv($fp,0,';','"',''))!==false) {
                            if (count($values)!==count($header)) { continue; }
                            $record=self::record(array_combine($header,$values));
                            if ($record && $record['ts']>=$first-120 && $record['ts']<=$last) { $records[$record['ts']]=$record; }
                        }
                        $sources[]=$name;
                    }
                    fclose($fp);
                }
            }
        }
        ksort($records,SORT_NUMERIC); $rows=array_values($records);
        $data=['station'=>$station,'serverTime'=>$now,'sources'=>$sources,
            'today'=>self::summarize($rows,$today,$now),'period'=>self::summarize($rows,$start,$end),
            'method'=>'Czas reprezentowany przez próbki (maks. 120 s na zapis). Cisza <1,8 km/h; braki kierunku nie są ciszą. Udziały obejmują ciszę.'];
        set_transient($key,$data,35); return rest_ensure_response($data);
    }
}
add_action('rest_api_init',function() {
    register_rest_route('mk-stations/v1','/wind-rose',[
        'methods'=>'GET','callback'=>['MK_Station_Wind_Analytics','respond'],'permission_callback'=>'__return_true'
    ]);
});

// Odświeżenie wersji publicznych zasobów dla obu niezależnych dashboardów.
add_action('template_redirect', function() {
    $path = parse_url($_SERVER['REQUEST_URI'] ?? '', PHP_URL_PATH);
    if (!in_array(rtrim((string)$path,'/'), ['/stacja-turaszowka','/stacja-lesniowka'],true)) { return; }
    ob_start(function($html) {
        return preg_replace('~(mk-station-dashboards/assets/dashboard\.(?:js|css)\?v=)[^"\s<>]+~','$1station-instruments-20261023',$html);
    });
}, -100);
