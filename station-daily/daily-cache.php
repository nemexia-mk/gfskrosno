<?php
// MK daily CSV/cache v1.0.0. Weather data only; no credentials are persisted.
function mksd_stamp($value) {
    if (!is_string($value) || !preg_match('/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/D', $value)) return null;
    $d = DateTimeImmutable::createFromFormat('!Y-m-d H:i:s', $value, new DateTimeZone('UTC'));
    return $d && $d->format('Y-m-d H:i:s') === $value ? $d->getTimestamp() : null;
}
function mksd_headers() {
    return array_merge(['czas_UTC','tempf','humidity','baromrelin','windspeedmph','windgustmph','maxdailygust','winddir','dailyrainin','monthlyrainin','rainratein','solarradiation','uv'], array_map(fn($i)=>'tf_ch'.$i, range(1,8)), array_map(fn($i)=>'temp'.$i.'f',range(1,8)),array_map(fn($i)=>'humidity'.$i,range(1,8)));
}
function mksd_csv_atomic($path, $header, $rows) {
    $tmp = $path.'.tmp-'.bin2hex(random_bytes(6));
    $fp = @fopen($tmp, 'xb'); if (!$fp) return false;
    $ok = fputcsv($fp, $header, ';', '"', '') !== false;
    foreach ($rows as $row) $ok = $ok && fputcsv($fp, $row, ';', '"', '') !== false;
    $ok = $ok && fflush($fp); fclose($fp);
    if (!$ok || !@rename($tmp, $path)) { @unlink($tmp); return false; }
    return true;
}
function mksd_row($data) {
    $row = [];
    foreach (mksd_headers() as $key) {
        $value = $key === 'czas_UTC' ? ($data['dateutc'] ?? $data['czas_UTC'] ?? '') : ($data[$key] ?? '');
        $row[] = is_scalar($value) && ($key === 'czas_UTC' || is_numeric((string)$value)) ? (string)$value : '';
    }
    return $row;
}
function mksd_daily_record($data, $directory, $suffix, $now = null) {
    $now = $now ?? time(); $stamp = mksd_stamp($data['dateutc'] ?? null);
    if (!in_array($suffix,['turaszowka','lesniowka'],true) || $stamp === null || $stamp > $now+120 || $now-$stamp > 900) return false;
    $tz = new DateTimeZone('Europe/Warsaw'); $local = (new DateTimeImmutable('@'.$stamp))->setTimezone($tz);
    $today = (new DateTimeImmutable('@'.$now))->setTimezone($tz)->format('Y-m-d');
    if ($local->format('Y-m-d') !== $today) return false;
    $directory = rtrim($directory,'/'); $path = $directory.'/dzis_'.$suffix.'.csv';
    $lock = @fopen($path.'.lock','c'); if (!$lock || !flock($lock,LOCK_EX)) { if ($lock) fclose($lock); return false; }
    try {
        $header = mksd_headers(); $byTime = []; $firstDay = null;
        if (($fp = @fopen($path,'rb'))) {
            $h = fgetcsv($fp,0,';','"','');
            if ($h === $header) while (($r=fgetcsv($fp,0,';','"','')) !== false) {
                $ts=mksd_stamp($r[0]??null); if ($ts===null || count($r)!==count($header)) continue;
                $day=(new DateTimeImmutable('@'.$ts))->setTimezone($tz)->format('Y-m-d');
                if ($firstDay===null) $firstDay=$day;
                if ($day===$today && $ts<=$now+120) $byTime[$ts]=$r;
            }
            fclose($fp);
        }
        // On installation seed today's history once from the existing monthly archive.
        // Rotation after midnight starts a new daily file; monthly archives are never changed.
        if ($firstDay === null) {
            $month = $directory.'/'.$local->format('m_Y').'_'.$suffix.'.csv';
            if (($fp=@fopen($month,'rb'))) {
                if (flock($fp,LOCK_SH)) {
                    $h=fgetcsv($fp,0,';','"','');
                    if (is_array($h)) {
                        $h[0]=preg_replace('/^\xEF\xBB\xBF/','',$h[0]); $map=array_flip($h);
                        while (($r=fgetcsv($fp,0,';','"',''))!==false) {
                            $ts=mksd_stamp($r[$map['czas_UTC']??-1]??null); if ($ts===null || $ts>$now+120) continue;
                            if ((new DateTimeImmutable('@'.$ts))->setTimezone($tz)->format('Y-m-d')!==$today) continue;
                            $d=[]; if(isset($map['dane_json'])) $d=json_decode($r[$map['dane_json']]??'',true)?:[];
                            if (!$d) foreach($map as $key=>$i) $d[$key]=$r[$i]??'';
                            $d['dateutc']=gmdate('Y-m-d H:i:s',$ts); $byTime[$ts]=mksd_row($d);
                        }
                    }
                    flock($fp,LOCK_UN);
                }
                fclose($fp);
            }
        }
        $byTime[$stamp]=mksd_row($data); ksort($byTime,SORT_NUMERIC);
        // A delayed upload cannot replace the newest observation with an older one.
        $rows=array_values($byTime); $latest=end($rows);
        $ok=mksd_csv_atomic($path,$header,$rows);
        return $ok && mksd_csv_atomic($directory.'/ostatni_'.$suffix.'.csv',$header,[$latest]);
    } finally { flock($lock,LOCK_UN); fclose($lock); }
}
function mksd_cache_path($id,$range,$date) {
    $base=rtrim(sys_get_temp_dir(),'/').'/mk-daily-reader-'.substr(hash('sha256',ABSPATH),0,12);
    if(!is_dir($base) && !@mkdir($base,0700,true) && !is_dir($base)) return null;
    return $base.'/'.hash('sha256','v1|'.$id.'|'.$range.'|'.$date).'.json';
}
function mksd_reader_cache($id,$range,$date,$daily,$now,$start) {
    if (!$daily || !is_readable($daily)) return null;
    $fp=@fopen($daily,'rb'); if(!$fp) return null;
    $h=fgetcsv($fp,0,';','"','');$r=fgetcsv($fp,0,';','"','');fclose($fp);
    $ts=mksd_stamp($r[0]??null); if($h!==mksd_headers() || $ts===null) return null;
    $today=$now->format('Y-m-d');$tz=new DateTimeZone('Europe/Warsaw');
    if((new DateTimeImmutable('@'.$ts))->setTimezone($tz)->format('Y-m-d')!==$today)return null;
    $path=mksd_cache_path($id,$range,$date); if(!$path || !is_readable($path))return null;
    $cache=json_decode(file_get_contents($path),true);
    if(!is_array($cache) || !is_array($cache['points']??null) || !is_array($cache['monthTemps']??null) || !is_array($cache['sources']??null) || ($cache['day']??'')!==$today || $now->getTimestamp()-($cache['created']??0)>600 || ($cache['start']??PHP_INT_MAX)>$start->getTimestamp())return null;
    return $cache;
}
function mksd_store_reader($id,$range,$date,$now,$start,$points,$monthTemps,$sources) {
    $path=mksd_cache_path($id,$range,$date);if(!$path)return;
    $data=json_encode(['day'=>$now->format('Y-m-d'),'created'=>$now->getTimestamp(),'start'=>$start->getTimestamp(),'points'=>$points,'monthTemps'=>$monthTemps,'sources'=>$sources],JSON_INVALID_UTF8_SUBSTITUTE);
    if($data===false)return;$tmp=$path.'.tmp-'.bin2hex(random_bytes(5));
    if(file_put_contents($tmp,$data,LOCK_EX)!==false)@rename($tmp,$path);else @unlink($tmp);
}
