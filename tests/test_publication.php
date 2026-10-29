<?php
define('ABSPATH',__DIR__);$hooks=[];
function add_action($name,$fn,$p=10){$GLOBALS['hooks'][$name][$p][]=$fn;}
function add_filter($name,$fn,$p=10){add_action($name,$fn,$p);}
function is_email($v){return filter_var($v,FILTER_VALIDATE_EMAIL);}
function esc_html($v){return htmlspecialchars($v,ENT_QUOTES,'UTF-8');}
function esc_url($v){return htmlspecialchars($v,ENT_QUOTES,'UTF-8');}
function home_url($p='/'){return 'https://meteo-krosno.pl'.$p;}
function content_url($p){return home_url('/wp-content'.$p);}
require __DIR__.'/../site/mu-plugins/mk-publication.php';
function ck($v,$message){if(!$v)throw new RuntimeException($message);}
ck(!mkp_privacy_ready(),'An incomplete identity must not enable measurement or publish a final policy');
$html='<html><head><title>Test</title><script async src="https://www.googletagmanager.com/gtag/js?id=UA-1"></script><script>gtag("config","G-1");</script><script type="application/ld+json">{"@type":"WebPage"}</script><script src="/assets/app.js"></script></head><body><img src="http://meteo-krosno.pl/wp-content/uploads/a.jpg"><p>Treść</p></body></html>';
$out=mkp_clean_html($html);
ck(strpos($out,'googletagmanager.com')===false&&strpos($out,'gtag(')===false,'Remove all legacy analytics loaders and config');
ck(strpos($out,'application/ld+json')!==false&&strpos($out,'/assets/app.js')!==false,'Preserve schema and application');
ck(strpos($out,'http://meteo-krosno.pl/wp-content/uploads/a.jpg')!==false,'Preserve original upload protocols');
ck(substr_count($out,'id="mk-privacy-root"')===1,'One consent UI');
ck(mkp_clean_html($out)===$out,'HTML sanitizing is idempotent');
ck(mkp_https_images('http://other.example/x.jpg')==='http://other.example/x.jpg','Do not invent HTTPS support on other hosts');
$_GET=['mk_station_frame'=>'lesniowka'];$frame=mkp_clean_html($html);ck(strpos($frame,'googletagmanager.com')===false&&strpos($frame,'mk-privacy-root')===false,'Embedded dashboard does not independently track or duplicate consent');$_GET=[];
$now=1793290000;$inv=['sensors'=>[['sensor'=>'outdoor.humidity','value'=>55,'time'=>$now],['sensor'=>'tf_ch10.humidity','value'=>50,'time'=>$now],['sensor'=>'tf_ch1.humidity','value'=>98,'time'=>$now]]];
$cloud=mkp_humidity(['sensors'=>[['sensor'=>'temp_and_humidity_ch1.temperature','value'=>45.3,'time'=>$now],['sensor'=>'temp_and_humidity_ch1.humidity','value'=>97,'time'=>$now]]],$now);
ck(mkp_humidity(['sensors'=>[['sensor'=>'temp_and_humidity_ch1.temperature','value'=>45.3,'time'=>$now]]],$now)===null,'Group names containing humidity must not turn their temperature leaf into humidity');ck($cloud['value']===97.0&&$cloud['sensor']==='temp_and_humidity_ch1','Cloud API channel name differs from the CSV field name');
$h=mkp_humidity($inv,$now);ck($h['value']===98.0,'Humidity comes from TF CH1, not outdoor or CH10');
foreach ([['value'=>null,'time'=>$now],['value'=>101,'time'=>$now],['value'=>98,'time'=>$now-901],['value'=>98,'time'=>$now+61]] as $bad){ck(mkp_humidity(['sensors'=>[array_merge(['sensor'=>'tf_ch1.humidity'],$bad)]],$now)===null,'Missing/invalid/stale/future readings are excluded');}
$data=['stations'=>[['id'=>'turaszowka','available'=>true,'temp'=>8,'humidity'=>99],['id'=>'lesniowka','available'=>true,'fallback'=>false,'temp'=>7.6,'humidity'=>null,'timestamps'=>['temp'=>$now]]]];
$fixed=mkp_correct_live($data,$h,$now);ck($fixed['stations'][0]===$data['stations'][0],'Turaszowka remains unchanged');ck($fixed['stations'][1]['humidity']===98.0&&$fixed['stations'][1]['timestamps']['humidity']===$now,'API includes actual humidity and timestamp');ck(abs($fixed['stations'][1]['dew']-7.3)<.11,'Dew point follows the corrected humidity');
ck(mkp_correct_live($data,null,$now)===$data,'An absent real measurement never becomes an invented number');
echo "Publication: legacy tracking removed, consent gated, secure images and correct TF CH1 humidity/timestamps OK\n";
