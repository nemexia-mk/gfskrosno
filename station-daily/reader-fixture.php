<?php
namespace MKStations;
use DateTimeImmutable;
use DateTimeZone;
defined('ABSPATH') || exit;

$stations=require __DIR__.'/config.php';
$id=$stationRequest;
if(!is_string($id)||!isset($stations[$id])) return new \WP_Error('station','Nieznana stacja',['status'=>404]);
$c=$stations[$id];$tz=new DateTimeZone('Europe/Warsaw');$utc=new DateTimeZone('UTC');$now=new DateTimeImmutable('now',$tz);
$range=in_array($rangeRequest,['6h','24h','7d','day'],true)?$rangeRequest:'24h';
$end=$now;
if($range==='day'){
 $start=DateTimeImmutable::createFromFormat('!Y-m-d',$dateRequest,$tz);
 if(!$start || $start->format('Y-m-d')!==$dateRequest || $start>$now)return new \WP_Error('date','Nieprawidłowa data',['status'=>400]);
 $end=$start->modify('+1 day');if($end>$now)$end=$now;
}else{$start=$now->modify($range==='7d'?'-7 days':($range==='6h'?'-6 hours':'-24 hours'));}

$rainStart=$now->setTimezone($utc)->setTime(6,0);if($rainStart>$now)$rainStart=$rainStart->modify('-1 day');
$rainYesterdayStart=$rainStart->modify('-1 day');
$day=$now->setTime(0,0);$earliest=$start<$day?$start:$day;
function num($v){if(!is_scalar($v)||$v==='')return null;$v=str_replace(',','.',(string)$v);return is_numeric($v)&&is_finite((float)$v)?(float)$v:null;}
function ftemp($v){$n=num($v);return $n===null?null:round(($n-32)*5/9,2);}
function mul($v,$m){$n=num($v);return $n===null?null:round($n*$m,3);}
function stamp($s){if(!is_string($s)||!preg_match('/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/D',$s))return null;$d=DateTimeImmutable::createFromFormat('!Y-m-d H:i:s',$s,new DateTimeZone('UTC'));return $d&&$d->format('Y-m-d H:i:s')===$s?$d->getTimestamp():null;}
function observation($d,$c){
 $t=stamp($d['dateutc']??null);if($t===null)return null;
 $o=['ts'=>$t,'temp'=>ftemp($d[$c['temperature']]??null),'humidity'=>num($d[$c['humidity']]??null),'pressure'=>mul($d['baromrelin']??null,33.8638866667),'wind'=>mul($d['windspeedmph']??null,1.609344),'gust'=>mul($d['windgustmph']??null,1.609344),'maxdailygust'=>mul($d['maxdailygust']??null,1.609344),'direction'=>num($d['winddir']??null),'rain'=>mul($d['dailyrainin']??null,25.4),'monthlyRain'=>mul($d['monthlyrainin']??null,25.4),'rainrate'=>mul($d['rainratein']??null,25.4),'solar'=>num($d['solarradiation']??null),'uv'=>num($d['uv']??null),'sensors'=>[]];
 foreach($c['sensors'] as $k=>$label)$o['sensors'][]=['key'=>$k,'label'=>$label,'value'=>ftemp($d[$k]??null)];
 foreach($c['sensors'] as $key=>$label)$o['sensor_'.$key]=ftemp($d[$key]??null);
 $o['dew']=null;if($o['temp']!==null&&$o['humidity']!==null&&$o['humidity']>0&&$o['humidity']<=100&&$o['temp']>-100&&$o['temp']<100){$g=log($o['humidity']/100)+17.62*$o['temp']/(243.12+$o['temp']);$o['dew']=round(243.12*$g/(17.62-$g),1);}
 return $o;
}
$monthStart=$now->modify('first day of this month')->setTime(0,0);$monthTemps=[];
$points=[];$sources=[];$warnings=[];$csvCount=0;
// Czytamy strumieniowo tylko miesiace potrzebne dla wybranego okresu.
$months=[];$m=$start->modify('first day of this month')->setTime(0,0);
while($m<=$end){$months[]=$m->format('m_Y').'_'.$c['suffix'].'.csv';$m=$m->modify('first day of next month');}
$months[]=$now->format('m_Y').'_'.$c['suffix'].'.csv';$months[]=$rainYesterdayStart->setTimezone($tz)->format('m_Y').'_'.$c['suffix'].'.csv';$months=array_values(array_unique($months));
// Stary plik jest dodatkowym zrodlem z okresu przed rotacja miesieczna.
$files=$months;
// MK daily reader v1: archive parsing is cached; today's small CSV supplies new points.
require_once __DIR__.'/daily-cache.php';
$mksDailyBase='dzis_'.$c['suffix'].'.csv';
$mksDailyPath=remote_file($c['dir'].'/'.$mksDailyBase,30);
$mksReaderCache=\mksd_reader_cache($id,$range,$dateRequest??'', $mksDailyPath,$now,$start);
if($mksReaderCache){$points=$mksReaderCache['points'];$monthTemps=$mksReaderCache['monthTemps'];$sources=$mksReaderCache['sources'];$files=[];}
if($mksDailyPath)$files[]=$mksDailyBase;

foreach($files as $base){
 $path=remote_file($c['dir'].'/'.$base,60);if(!$path)continue;
 $fp=@fopen($path,'rb');if(!$fp){$warnings[]='Nie można odczytać '.$base;continue;}
 if(!flock($fp,LOCK_SH)){fclose($fp);continue;}
 $h=fgetcsv($fp,0,';','"','');if(!$h){flock($fp,LOCK_UN);fclose($fp);continue;}
 $h[0]=preg_replace('/^\xEF\xBB\xBF/','',$h[0]);$map=array_flip($h);$sources[]=$base;
 while(($r=fgetcsv($fp,0,';','"',''))!==false){
  $ts=isset($map['czas_UTC'])?stamp($r[$map['czas_UTC']]??null):null;
  if($ts===null||$ts>$now->getTimestamp()+120)continue;
  if($ts>=$monthStart->getTimestamp()&&$ts<=$now->getTimestamp()){
   $mt=isset($map[$c['temperature']])?ftemp($r[$map[$c['temperature']]]??null):null;
   if(!isset($map[$c['temperature']])&&isset($map['dane_json'])){$md=json_decode($r[$map['dane_json']]??'',true);if(is_array($md))$mt=ftemp($md[$c['temperature']]??null);}
   $monthTemps[$ts]=$mt;
  }

  if(!(($ts>=$start->getTimestamp()&&$ts<=$end->getTimestamp())||$ts>=$rainYesterdayStart->getTimestamp()))continue;
  $d=[];
  if(isset($map['dane_json'])){$decoded=json_decode($r[$map['dane_json']]??'',true);if(is_array($decoded))$d=$decoded;}
  if(!$d){foreach($map as $k=>$i)$d[$k]=$r[$i]??'';$d['dateutc']=$r[$map['czas_UTC']];}
  $o=observation($d,$c);if($o){$points[$ts]=$o;$csvCount++;}
 }
 flock($fp,LOCK_UN);fclose($fp);
}
if(!$mksReaderCache)\mksd_store_reader($id,$range,$dateRequest??'', $now,$start,$points,$monthTemps,$sources);
$sources=array_values(array_unique($sources));
ksort($points,SORT_NUMERIC);$records=array_values($points);
$latest=$records?end($records):null;
$stats=['min'=>null,'max'=>null,'gust'=>null,'rain'=>null,'mean'=>null,'samples'=>0,'coverage'=>0,'first'=>null,'last'=>null,'groundMin'=>null,'groundMax'=>null,'solarMax'=>null,'rainYesterday'=>null,'rainYesterdayComplete'=>false,'rainYesterdayStart'=>$rainYesterdayStart->getTimestamp(),'rainYesterdayEnd'=>$rainStart->getTimestamp()];$sum=0;
foreach($records as $o){if($o['ts']<$day->getTimestamp()||$o['ts']>$now->getTimestamp()+120)continue;$stats['samples']++;if($stats['first']===null)$stats['first']=$o['ts'];$stats['last']=$o['ts'];
 $ground=$o['sensor_tf_ch3']??null;
 if($id==='turaszowka'&&$ground!==null){foreach(['groundMin','groundMax'] as $gk)if($stats[$gk]===null||($gk==='groundMin'?$ground<$stats[$gk]['value']:$ground>$stats[$gk]['value']))$stats[$gk]=['value'=>$ground,'ts'=>$o['ts']];}
 if($o['solar']!==null&&($stats['solarMax']===null||$o['solar']>$stats['solarMax']['value']))$stats['solarMax']=['value'=>$o['solar'],'ts'=>$o['ts']];
 if($o['temp']!==null){$sum+=$o['temp'];$temps[]=$o['temp'];foreach(['min','max'] as $k)if($stats[$k]===null||($k==='min'?$o['temp']<$stats[$k]['value']:$o['temp']>$stats[$k]['value']))$stats[$k]=['value'=>$o['temp'],'ts'=>$o['ts']];}
 if($o['gust']!==null&&($stats['gust']===null||$o['gust']>$stats['gust']['value']))$stats['gust']=['value'=>$o['gust'],'ts'=>$o['ts']];
}
if(isset($temps))$stats['mean']=round($sum/count($temps),2);
if($latest&&$latest['ts']>=$rainStart->getTimestamp()&&$latest['ts']<=$now->getTimestamp()+120)$stats['rain']=$latest['rain'];
if($latest&&$latest['ts']>=$day->getTimestamp()&&$latest['solar']!==null&&($stats['solarMax']===null||$latest['solar']>$stats['solarMax']['value']))$stats['solarMax']=['value'=>$latest['solar'],'ts'=>$latest['ts']];
// Licznik bramek zeruje sie o 06 UTC; odczyt koncowy zamknietej doby.
$yesterdayFirst=null;$yesterdayLast=null;
foreach($records as $o){if($o['ts']<$rainYesterdayStart->getTimestamp()||$o['ts']>=$rainStart->getTimestamp()||$o['rain']===null)continue;if($yesterdayFirst===null)$yesterdayFirst=$o['ts'];$yesterdayLast=$o['ts'];$stats['rainYesterday']=$stats['rainYesterday']===null?$o['rain']:max($stats['rainYesterday'],$o['rain']);}
$stats['rainYesterdayComplete']=$yesterdayFirst!==null&&$yesterdayFirst-$rainYesterdayStart->getTimestamp()<=300&&$rainStart->getTimestamp()-$yesterdayLast<=300;

// Pokrycie: minuty z co najmniej jedna paczka / minuty od lokalnej polnocy.
$minutes=[];foreach($records as $o)if($o['ts']>=$day->getTimestamp()&&$o['ts']<=$now->getTimestamp())$minutes[(int)floor($o['ts']/60)]=true;
$stats['coverage']=min(100,round(count($minutes)/max(1,ceil(($now->getTimestamp()-$day->getTimestamp())/60))*100,1));
// Agregacja 5/30 min: srednia temperatury, wilgotnosci i cisnienia,
// maksimum porywu i ostatni licznik opadu. Luka nie jest wypelniana zerami.
$requestedStep=in_array($resolutionRequest,[60,300,900,3600],true)?$resolutionRequest:0;
$step=$requestedStep?:($range==='7d'?1800:($range==='6h'?60:300));
if($range==='7d')$step=max(300,$step);
$buckets=[];$periodTemps=[];$periodMinutes=[];$periodGust=null;$periodCount=0;
foreach($records as $o){if($o['ts']<$start->getTimestamp()||$o['ts']>$end->getTimestamp())continue;$periodCount++;$periodMinutes[(int)floor($o['ts']/60)]=true;if($o['temp']!==null)$periodTemps[]=$o['temp'];if($o['gust']!==null&&($periodGust===null||$o['gust']>$periodGust))$periodGust=$o['gust'];$b=(int)(floor($o['ts']/$step)*$step);$buckets[$b][]=$o;}
$chart=[];foreach($buckets as $b=>$obs){$p=['ts'=>$b];foreach(array_merge(['temp','humidity','pressure','wind','dew','rainrate','solar','uv'],array_map(function($k){return 'sensor_'.$k;},array_keys($c['sensors']))) as $k){$vals=array_column($obs,$k);$vals=array_values(array_filter($vals,function($v){return $v!==null;}));$p[$k]=$vals?round(array_sum($vals)/count($vals),2):null;}$gs=array_values(array_filter(array_column($obs,'gust'),function($v){return $v!==null;}));$p['gust']=$gs?max($gs):null;$last=end($obs);$p['rain']=$last['rain'];$tv=array_values(array_filter(array_column($obs,'temp'),function($v){return $v!==null;}));$p['low']=$tv?min($tv):null;$p['high']=$tv?max($tv):null;$p['samples']=count($obs);$chart[]=$p;}
if(!$sources)$warnings[]='Brak archiwum CSV dla wybranego okresu. Historia pojawi się po rozpoczęciu zapisu.';
$age=$latest?$now->getTimestamp()-$latest['ts']:null;

$tempNorms=[1=>-2.2,2=>-0.9,3=>3.1,4=>9.0,5=>13.6,6=>17.0,7=>18.7,8=>18.5,9=>13.7,10=>9.0,11=>4.2,12=>-0.9];
$precipNorms=[1=>37.7,2=>38.6,3=>38.3,4=>53.0,5=>95.2,6=>92.5,7=>105.6,8=>75.0,9=>76.6,10=>61.9,11=>43.9,12=>39.6];
$sunshineNorms=[1=>62.6,2=>72.1,3=>125.2,4=>181.6,5=>224.6,6=>229.2,7=>246.1,8=>243.1,9=>164.0,10=>120.0,11=>69.1,12=>51.4];
$frostNorms=[1=>9.5,2=>11.1,3=>12.6,4=>4.4,5=>0.4,6=>0.0,7=>0.0,8=>0.0,9=>3.2,10=>3.2,11=>7.9,12=>10.9];
$monthsNames=[1=>'Styczeń',2=>'Luty',3=>'Marzec',4=>'Kwiecień',5=>'Maj',6=>'Czerwiec',7=>'Lipiec',8=>'Sierpień',9=>'Wrzesień',10=>'Październik',11=>'Listopad',12=>'Grudzień'];
$monthNum=(int)$now->format('n');$monthTotal=0;$monthCount=0;$monthMins=[];$frostDates=[];$firstMonth=null;
foreach($monthTemps as $ts=>$mt){if($mt===null)continue;$monthTotal+=$mt;$monthCount++;$monthMins[(int)floor($ts/60)]=true;if($firstMonth===null||$ts<$firstMonth)$firstMonth=$ts;if($mt<0)$frostDates[(new DateTimeImmutable('@'.$ts))->setTimezone($tz)->format('Y-m-d')]=true;}
$monthMean=$monthCount?round($monthTotal/$monthCount,2):null;$monthRain=$latest&&$latest['ts']>=$monthStart->getTimestamp()?$latest['monthlyRain']:null;
$monthly=['label'=>$monthsNames[$monthNum].' '.$now->format('Y'),'tempMean'=>$monthMean,'tempNorm'=>$tempNorms[$monthNum],'tempAnomaly'=>$monthMean===null?null:round($monthMean-$tempNorms[$monthNum],2),'rain'=>$monthRain,'rainNorm'=>$precipNorms[$monthNum],'rainPercent'=>$monthRain===null?null:round($monthRain/$precipNorms[$monthNum]*100,1),'frostDays'=>$monthCount?count($frostDates):null,'frostNorm'=>$frostNorms[$monthNum],'sunshine'=>null,'sunshinePercent'=>null,'sunshineThrough'=>null,'sunshineDays'=>0,'sunshineNorm'=>$sunshineNorms[$monthNum],'first'=>$firstMonth,'coverage'=>min(100,round(count($monthMins)/max(1,ceil(($now->getTimestamp()-$monthStart->getTimestamp())/60))*100,1))];

// Dobowe uslonecznienie Krosna IMGW: suma godzin z kolumny SUN HR (16).
$sunPath=remote_file('http://stacja.meteo-krosno.pl/zbiorczy.csv',900);
if($sunPath&&($sunFp=@fopen($sunPath,'rb'))){
 if(flock($sunFp,LOCK_SH)){
  $sunDays=[];
  while(($sr=fgetcsv($sunFp,0,';','"',''))!==false){
   if(count($sr)<18||(int)$sr[2] !== (int)$now->format('Y')||(int)$sr[3]!==$monthNum)continue;
   $sd=(int)$sr[4];if(!checkdate($monthNum,$sd,(int)$sr[2]))continue;
   $dateKey=sprintf('%04d-%02d-%02d',(int)$sr[2],$monthNum,$sd);
   if($dateKey>=$now->format('Y-m-d'))continue;
   $sun=num($sr[16]??null);$flag=trim($sr[17]??'');
   if($sun===null||$sun<0||$sun>24||in_array($flag,['8','9'],true))continue;
   $sunDays[$dateKey]=$sun;
  }
  flock($sunFp,LOCK_UN);
  if($sunDays){ksort($sunDays);$monthly['sunshine']=round(array_sum($sunDays),1);$monthly['sunshinePercent']=round($monthly['sunshine']/$monthly['sunshineNorm']*100,1);$monthly['sunshineThrough']=array_key_last($sunDays);$monthly['sunshineDays']=count($sunDays);}
 }
 fclose($sunFp);
}
return ['monthly'=>$monthly,'refreshRequired'=>!empty($GLOBALS['mks_refresh_required']),'dayStart'=>$day->getTimestamp(),'sourceMode'=>$GLOBALS['mks_source_mode']??'csv','windRecords'=>array_map(function($o){return [$o['ts'],$o['wind'],$o['direction']];},$records),'station'=>$id,'name'=>$c['name'],'serverTime'=>$now->getTimestamp(),'day'=>$day->format('Y-m-d'),'range'=>$range,'step'=>$step,'start'=>$start->getTimestamp(),'end'=>$end->getTimestamp(),'period'=>['min'=>$periodTemps?min($periodTemps):null,'max'=>$periodTemps?max($periodTemps):null,'mean'=>$periodTemps?round(array_sum($periodTemps)/count($periodTemps),2):null,'gust'=>$periodGust,'samples'=>$periodCount,'coverage'=>min(100,round(count($periodMinutes)/max(1,ceil(($end->getTimestamp()-$start->getTimestamp())/60))*100,1))],'latest'=>$latest,'age'=>$age,'status'=>$age===null?'missing':($age< -120?'clock':($age<=180?'live':($age<=900?'delayed':'offline'))),'stats'=>$stats,'chart'=>$chart,'sources'=>$sources,'warnings'=>$warnings];
