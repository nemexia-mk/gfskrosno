<?php
define('ABSPATH',__DIR__);function add_action(...$args){}
require __DIR__.'/../site/mu-plugins/mk-home-loading.php';
function ck($condition){if(!$condition)throw new RuntimeException('Home loading check failed');}
$base='<html><head><title>Pogoda</title></head><body><header>Menu</header><main id="main"><section class="mk-seo-content"><h1>Pogoda w Krośnie</h1><table><tr><td>7,4</td></tr></table></section></main></body></html>';
$patched=mk_home_loading_html($base);
ck(strpos($patched,'mk-home-loading-style')<strpos($patched,'<title>'));
ck(substr_count($patched,'id="mk-home-loader"')===1);
ck(strpos($patched,'<h1>Pogoda w Krośnie</h1><table>')!==false);
ck(mk_home_loading_html($patched)===$patched);
ck(mk_home_loading_html('<html><head></head><body>Other</body></html>')==='<html><head></head><body>Other</body></html>');
echo "PASS: early loader, unchanged basic content, idempotency and other templates\n";
