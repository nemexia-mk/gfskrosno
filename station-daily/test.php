<?php
require __DIR__.'/daily-cache.php';
define('ABSPATH',sys_get_temp_dir().'/mks-test-'.bin2hex(random_bytes(6)).'/');
function ck($value,$message){if(!$value)throw new Exception($message);}
function rows($file){$fp=fopen($file,'rb');$out=[];while(($r=fgetcsv($fp,0,';','"',''))!==false)$out[]=$r;fclose($fp);return $out;}
$dir=sys_get_temp_dir().'/mks-daily-tests-'.bin2hex(random_bytes(6));mkdir($dir,0700);
$now=strtotime('2026-10-30 10:00:00 UTC');
$d=['dateutc'=>'2026-10-30 10:00:00','tempf'=>'45.5','humidity'=>'99','humidity1'=>'98','tf_ch1'=>'44.6','PASSKEY'=>'must-not-be-saved'];
ck(mksd_daily_record($d,$dir,'lesniowka',$now),'First record');
$r=rows($dir.'/dzis_lesniowka.csv');ck(count($r)===2,'One row and header');ck(!str_contains(file_get_contents($dir.'/dzis_lesniowka.csv'),'must-not-be-saved'),'No secrets in weather CSV');
ck(mksd_daily_record($d,$dir,'lesniowka',$now),'Duplicate request');ck(count(rows($dir.'/dzis_lesniowka.csv'))===2,'Timestamp deduplication');
$older=$d;$older['dateutc']='2026-10-30 09:59:00';$older['humidity1']='97';mksd_daily_record($older,$dir,'lesniowka',$now);
ck(count(rows($dir.'/dzis_lesniowka.csv'))===3,'Older valid row retained');ck(rows($dir.'/ostatni_lesniowka.csv')[1][0]===$d['dateutc'],'Latest cannot move backwards');
ck(mksd_stamp("2026-10-30\0 10:00:00")===null,'NUL timestamp rejected');ck(mksd_stamp('2026-02-30 10:00:00')===null,'Impossible date rejected');
$future=$d;$future['dateutc']='2026-10-30 11:00:00';ck(!mksd_daily_record($future,$dir,'lesniowka',$now),'Future upload rejected');
// Midnight Warsaw is 23 UTC in winter and 22 UTC in summer.
$next=$d;$next['dateutc']='2026-10-30 23:00:01';$ts=strtotime($next['dateutc'].' UTC');mksd_daily_record($next,$dir,'lesniowka',$ts);
ck(count(rows($dir.'/dzis_lesniowka.csv'))===2,'Rotation at local midnight');
ck(!mksd_daily_record($d,$dir,'lesniowka',$ts),'Previous-day upload cannot rotate back');
file_put_contents($dir.'/10_2026_lesniowka.csv','unchanged-monthly-archive');$before=hash_file('sha256',$dir.'/10_2026_lesniowka.csv');mksd_daily_record($next,$dir,'lesniowka',$ts);ck(hash_file('sha256',$dir.'/10_2026_lesniowka.csv')===$before,'Monthly archive unchanged');
$nowDate=(new DateTimeImmutable('@'.$ts))->setTimezone(new DateTimeZone('Europe/Warsaw'));$start=$nowDate->modify('-6 hours');
mksd_store_reader('lesniowka','6h','',$nowDate,$start,[123=>['ts'=>123]],[123=>4.5],['monthly.csv']);
$cache=mksd_reader_cache('lesniowka','6h','',$dir.'/dzis_lesniowka.csv',$nowDate,$start);ck(is_array($cache)&&$cache['monthTemps'][123]===4.5,'Archive cache retains monthly data');
ck(mksd_reader_cache('lesniowka','6h','',$dir.'/dzis_lesniowka.csv',$nowDate->modify('+601 seconds'),$start)===null,'Cache expiry');
ck(mksd_reader_cache('lesniowka','6h','',$dir.'/dzis_lesniowka.csv',$nowDate->modify('+1 day'),$start)===null,'Day-boundary cache invalidation');
// Initial installation seeds valid records, skipping a corrupt timestamp row.
$seed=$dir.'/seed';mkdir($seed);$fp=fopen($seed.'/10_2026_turaszowka.csv','wb');fputcsv($fp,['czas_UTC','tempf','humidity'],';','"','');fputcsv($fp,["\0\0\0",'0','0'],';','"','');fputcsv($fp,['2026-10-30 09:58:00','45','90'],';','"','');fclose($fp);
ck(mksd_daily_record($d,$seed,'turaszowka',$now),'Seed accepted');ck(count(rows($seed.'/dzis_turaszowka.csv'))===3,'Valid earlier history seeded');
// The mirror copies only a changed, saved monthly CSV, never raw POST credentials/data.
$mirror=$dir.'/mirror';mkdir($mirror);$nowLive=time();$utcLive=gmdate('Y-m-d H:i:s',$nowLive);$monthLive=(new DateTimeImmutable('@'.$nowLive))->setTimezone(new DateTimeZone('Europe/Warsaw'))->format('m_Y');
$child=$mirror.'/receiver-fixture.php';
$code="<?php\nrequire ".var_export(__DIR__.'/daily-cache.php',true).";\n";
$code.='$_SERVER["REQUEST_METHOD"]="POST"; $_POST=["humidity"=>"1","PASSKEY"=>"not-weather-data"];' . "\n";
$code.='mksd_register_monthly_mirror(__DIR__,"turaszowka");'."\n";
$code.='$fp=fopen(__DIR__."/'. $monthLive.'_turaszowka.csv","wb");fputcsv($fp,["czas_UTC","tempf","humidity"],";",chr(34),"");fputcsv($fp,['.var_export($utcLive,true).',"46.4","99"],";",chr(34),"");fclose($fp);';
file_put_contents($child,$code);exec(escapeshellarg(PHP_BINARY).' '.escapeshellarg($child),$output,$exit);ck($exit===0,'Receiver fixture exit');
$mirrored=rows($mirror.'/dzis_turaszowka.csv');ck($mirrored[1][2]==='99','Only accepted saved measurements mirrored');
ck(!str_contains(file_get_contents($mirror.'/dzis_turaszowka.csv'),'not-weather-data'),'Raw POST is never mirrored');
$small=$dir.'/small-cache';$smallBody=file_get_contents($mirror.'/dzis_turaszowka.csv');
ck(mksd_refresh_small_csv('http://stacja.meteo-krosno.pl/pwsWD/ecowitt/dzis_turaszowka.csv',$small,30,fn($u)=>$smallBody,$nowLive),'Small standalone CSV refresh');
ck(!mksd_refresh_small_csv('http://other.example/dzis_turaszowka.csv',$small,30,fn($u)=>$smallBody,$nowLive),'No arbitrary remote fetches');
$badPath=$dir.'/bad-cache';ck(!mksd_refresh_small_csv('http://stacja.meteo-krosno.pl/pwsWD/ecowitt/dzis_turaszowka.csv',$badPath,30,fn($u)=>'<html>Error</html>',$nowLive),'Error page is never cached as weather');
ck(!is_file($badPath),'Invalid fetch cannot replace prior weather');
echo "Daily CSV: rotation, timestamps, duplicate uploads, out-of-order uploads, secret exclusion, archive preservation, cache and corrupt rows passed.\n";
