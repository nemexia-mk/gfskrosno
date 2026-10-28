<?php
/* Contract tests for public HTML, unpublished content and sitemap boundaries. */
define('ABSPATH', __DIR__);
$hooks = []; $options = []; $updated = []; $fixture_posts = []; $queried = null; $is404 = false; $singular = false;
class WP_Post { public $ID = 1; public $post_status = 'publish'; public $post_name = 'sample'; public $post_title = 'Tytuł'; public $post_type = 'post'; public $post_content = ''; public $post_excerpt = ''; public $post_password = ''; }
function add_action($name, $fn, $priority = 10) { global $hooks; $hooks[$name][$priority][] = $fn; }
function add_filter($name, $fn, $priority = 10) { add_action($name, $fn, $priority); }
function home_url($p = '/') { return 'https://meteo-krosno.pl' . $p; }
function get_queried_object() { global $queried; return $queried; }
function is_404() { global $is404; return $is404; }
function is_search() { return false; } function is_author() { return false; } function is_date() { return false; } function is_attachment() { return false; }
function is_singular($type = null) { global $singular, $queried; return $singular && (!$type || $queried->post_type === $type); }
function esc_html($s) { return htmlspecialchars($s, ENT_QUOTES, 'UTF-8'); }
function esc_attr($s) { return esc_html($s); } function esc_url($s) { return esc_html($s); }
function get_the_title($p) { return $p->post_title; }
function wp_strip_all_tags($s) { return strip_tags(preg_replace('~<(script|style)\b[^>]*>.*?</\1>~is', '', $s)); } function strip_shortcodes($s) { return $s; }
function wp_trim_words($s, $n, $more = '') { $a = preg_split('/\s+/', $s); return implode(' ', array_slice($a, 0, $n)) . (count($a) > $n ? $more : ''); }
function get_permalink($p) { return home_url('/2026/10/26/' . $p->post_name . '/'); }
function get_the_post_thumbnail_url($p, $size) { return false; } function get_site_icon_url($size) { return home_url('/icon.png'); }
function get_post_time($format, $gmt, $post) { return '2026-10-26T10:00:00+00:00'; }
function get_post_modified_time($format, $gmt, $post) { return '2026-10-27T10:00:00+00:00'; }
function wp_json_encode($s, $options = 0) { return json_encode($s, $options); }
function get_template_directory() { return sys_get_temp_dir() . '/missing-mk-theme'; }
function get_posts($args) { global $fixture_posts; return $fixture_posts; }
function apply_filters($name, $value) {
    if (!empty($GLOBALS['buffering_filter'])) { ob_start(); echo $value; return ob_get_clean(); }
    return $value;
}
function wp_kses_post($content) { return strip_tags($content, '<h2><p><a><table><tr><td>'); }
function get_option($key) { global $options; return $options[$key] ?? false; }
function add_option($key, $value, $unused = '', $autoload = false) { global $options; if (!isset($options[$key])) { $options[$key] = $value; } }
function get_page_by_path($path, $output, $type) { global $sample; return $sample; }
function wp_update_post($data, $error = false) { global $sample, $updated; $updated[] = $data; $sample->post_status = $data['post_status']; return $sample->ID; }
function is_wp_error($value) { return false; }
require __DIR__ . '/../site/mu-plugins/mk-public-seo.php';
function check($condition, $message) { if (!$condition) { throw new RuntimeException($message); } }

$fixture = '<!doctype html><html><head><title>Meteo Krosno</title><meta name="robots" content="noindex"><meta name="robots" content="index,follow"><meta name="description" content="old"><link rel="canonical" href="https://wrong.example"><script type="application/ld+json">{"old":1}</script><script src="app.js" defer></script></head><body><header>Menu</header><main id="main" class="container"></main><noscript><main><h1>Meteo Krosno</h1></main></noscript><footer>Stopka</footer></body></html>';
$titles = [];
foreach (mkseo_routes() as $path => $route) {
    $_SERVER['REQUEST_URI'] = '/' . ($path ? $path . '/' : ''); $_GET = [];
    $out = mkseo_html($fixture);
    check(substr_count($out, '<title>') === 1, 'Exactly one title: ' . $path);
    check(substr_count($out, 'name="robots"') === 1 && strpos($out, 'content="index,follow,max-image-preview:large"') !== false, 'Consistent index: ' . $path);
    check(substr_count($out, '<h1>') === 1 && strpos($out, esc_html($route[1])) !== false, 'One meaningful H1: ' . $path);
    check(strpos($out, esc_attr($route[2])) !== false, 'Description: ' . $path);
    check(strpos($out, 'wrong.example') === false && substr_count($out, 'rel="canonical"') === 1, 'Canonical replacement');
    check(strpos($out, 'app.js') !== false && strpos($out, '<header>Menu</header>') !== false, 'Interactive app and layout preserved');
    check(strpos($out, '<noscript>') === false && substr_count($out, '<main') === 1, 'No duplicate fallback main');
    preg_match('~<script type="application/ld\+json">(.*?)</script>~s', $out, $json);
    check(isset($json[1]) && json_decode($json[1], true)['@context'] === 'https://schema.org', 'Valid JSON-LD');
    $titles[] = $route[0];
}
check(count(array_unique($titles)) === count($titles), 'Titles are unique');
$_SERVER['REQUEST_URI'] = '/klimat-miesieczny/';
$native = str_replace('<main id="main" class="container"></main>', '<main class="report"><h1>Październik 2026</h1><p>Opad 63,9 mm</p></main>', $fixture);
$out = mkseo_html($native);
check(strpos($out, 'Podsumowanie klimatyczne miesiąca – Październik 2026') !== false && strpos($out, 'Opad 63,9 mm') !== false, 'Native report data preserved and heading clarified');
$_SERVER['REQUEST_URI'] = '/audyt-nieistniejaca/'; $is404 = true;
$out = mkseo_html($fixture);
check(strpos($out, 'content="noindex,follow"') !== false, '404 is noindex');
check(strpos($out, 'rel="canonical"') === false, '404 has no fake canonical');
check(strpos($out, 'application/ld+json') === false, '404 has no indexed page schema');
$is404 = false;

$published = new WP_Post(); $published->post_name = 'podsumowanie-wrzesien-2026';
$published->post_content = '<h1>Wyniki</h1><p>Rzeczywista treść &amp; pomiary.</p><script>alert(1)</script>';
$draft = clone $published; $draft->ID = 2; $draft->post_status = 'draft'; $draft->post_name = 'tajny-szkic';
$test = clone $published; $test->ID = 3; $test->post_name = 'podsumowanie-testowe';
$fixture_posts = [$published, $draft, $test];
check(mkseo_article_html($draft) === '', 'Draft body never rendered');
check(mkseo_article_html($test) === '', 'Test body never rendered');
$queried = $published; $singular = true; $_SERVER['REQUEST_URI'] = '/2026/10/26/podsumowanie-wrzesien-2026/';
$out = mkseo_html($fixture);
check(substr_count($out, '<h1>') === 1 && strpos($out, '<h2>Wyniki</h2>') !== false, 'Article content preserved with single H1');
check(strpos($out, 'alert(1)') === false, 'Scripts excluded from fallback');
check(strpos($out, 'datePublished') !== false, 'Article dates in schema');
// A shortcode-like filter can use buffering, as WordPress blocks actually do.
$GLOBALS['buffering_filter'] = true;
$GLOBALS['mkseo_prepared_articles'][$published->ID] = mkseo_article_html($published);
ob_start(); ob_start('mkseo_html'); echo $fixture; ob_end_flush(); $buffered = ob_get_clean();
check(strpos($buffered, '<h2>Wyniki</h2>') !== false, 'Prepared article supports shortcode output buffering');
unset($GLOBALS['mkseo_prepared_articles'], $GLOBALS['buffering_filter']);
$xml = mkseo_sitemap_xml('post');
check(strpos($xml, 'podsumowanie-wrzesien-2026') !== false && strpos($xml, 'tajny-szkic') === false && strpos($xml, 'podsumowanie-testowe') === false, 'Only public editorial posts in sitemap');
$xml = mkseo_sitemap_xml('page');
check(strpos($xml, '/prognoza/') !== false && strpos($xml, '/analizator/') !== false && strpos($xml, 'przykladowa-strona') === false && strpos($xml, 'szukac-dostepne') === false, 'Curated pages in sitemap');
$_GET = ['mk_api' => 'gfs']; check(mkseo_internal(), 'API excluded from HTML transformations');
$_GET = ['mk_station_frame' => 'turaszowka']; check(mkseo_internal(), 'Embedded station excluded from transformations');
define('OBJECT', 'OBJECT'); $sample = new WP_Post(); $sample->post_type = 'page'; $sample->post_name = 'przykladowa-strona';
$migration = $hooks['init'][20][0]; $migration(); $migration();
check($sample->post_status === 'draft' && count($updated) === 1, 'Draft migration is idempotent');
check($options['mkseo_sample_previous_status_v1']['status'] === 'publish', 'Previous status preserved for rollback');
echo "Public SEO contracts passed: HTML, metadata, drafts, schema and sitemaps.\n";
