<?php
define('ABSPATH',__DIR__);$filters=[];
function add_action(...$a){}function add_filter($name,$fn){global $filters;$filters[$name]=$fn;}
function esc_html($s){return htmlspecialchars($s,ENT_QUOTES,'UTF-8');}function esc_attr($s){return esc_html($s);}function esc_url($s){return esc_html($s);}
function ck($v,$label){if(!$v)throw new RuntimeException($label);}
require __DIR__.'/../public-app/mk-android-release.php';
$base='<html><head><title>App</title></head><body><header><summary>O serwisie</summary><div></div></header><main id="main"><section class="mk-seo-content"><h1>Fallback</h1></section></main><footer>Footer</footer><script src="https://meteo-krosno.pl/wp-content/themes/meteo-krosno-modern/assets/app.js?v=123" defer></script><script src="consent-20261029.js"></script></body></html>';
$_SERVER['REQUEST_URI']='/aplikacja/';$page=mkandroid_html($base);
ck(substr_count($page,'<h1>')===1,'single H1');ck(strpos($page,'<h1>Aplikacja Meteo Krosno</h1>')!==false,'actual page');ck(strpos($page,'/assets/app.js')===false,'SPA cannot overwrite page');ck(strpos($page,'consent-20261029.js')!==false,'consent retained');ck(strpos($page,'Meteo-Krosno-1.0.0.apk')!==false,'download');ck(strpos($page,'Informacje o prywatności')!==false,'privacy');ck(isset($filters['mkseo_routes']([])['aplikacja']),'route in sitemap');ck(mkandroid_html($page)===$page,'idempotent');
$_SERVER['REQUEST_URI']='/';$home=mkandroid_html($base);ck(strpos($home,'id="mk-android-promo"')!==false,'home download button');ck(strpos($home,'/assets/app.js')!==false,'home bootstrap retained');
$_SERVER['REQUEST_URI']='/prognoza/';$other=mkandroid_html($base);ck(strpos($other,'id="mk-android-promo"')===false,'no extra home promo elsewhere');ck(strpos($other,'/assets/app.js')!==false,'forecast bootstrap retained');echo "PASS: dedicated static page, download links, sitemap route, privacy, home/menu links and other page bootstrap\n";
