<?php
function add_action(...$args) {}
define('ABSPATH','/tmp/meteo-test/wordpress/');
require __DIR__.'/../mu-plugins/mk-station-wind-analytics.php';
function check($value,$label){if(!$value){throw new Exception($label);}}
$r=MK_Station_Wind_Analytics::record(['czas_UTC'=>'2026-10-23 10:00:00','windspeedmph'=>'10,0','winddir'=>'0']);
check(abs($r['wind']-16.09344)<0.00001,'mph conversion');check($r['direction']===0.0,'north zero');
$m=MK_Station_Wind_Analytics::record(['czas_UTC'=>'2026-10-23 10:00:00','windspeedmph'=>'','winddir'=>'']);check($m['wind']===null&&$m['direction']===null,'missing not calm');
$rows=[['ts'=>0,'wind'=>10,'direction'=>350],['ts'=>60,'wind'=>20,'direction'=>10],['ts'=>120,'wind'=>0,'direction'=>null],['ts'=>180,'wind'=>12,'direction'=>null],['ts'=>240,'wind'=>null,'direction'=>null]];
$s=MK_Station_Wind_Analytics::summarize($rows,0,300);
check($s['validSeconds']===180.0,'coverage excludes missing');check($s['calmSeconds']===60.0,'calm');check($s['missingDirectionSeconds']===60.0,'missing direction');check($s['dominant']['code']==='N','circular north');check(abs($s['coverage']-60)<.01,'coverage');
$s=MK_Station_Wind_Analytics::summarize([['ts'=>0,'wind'=>10,'direction'=>360],['ts'=>600,'wind'=>10,'direction'=>180]],0,660);check($s['validSeconds']===180.0,'cap gaps');
$s=MK_Station_Wind_Analytics::summarize([$m],$m['ts'],$m['ts']+60);check($s['dominant']===null&&$s['calmPercent']===null,'no fabricated calm');
$tmp=tempnam(sys_get_temp_dir(),'wind');file_put_contents($tmp,"\xEF\xBB\xBFczas_UTC;windspeedmph;winddir\n2026-10-23 10:00:00;2,5;360\nmalformed;0\n");$records=[];MK_Station_Wind_Analytics::file_rows($tmp,0,PHP_INT_MAX,$records);unlink($tmp);check(count($records)===1,'CSV BOM and malformed rows');
echo "PASS wind units, nulls, north, calm, coverage, gaps, CSV\n";
