<?php
/**
 * Plugin Name: Meteo Krosno – prywatność i porządki publikacji
 * Version: 1.0.0
 */
if (!defined('ABSPATH')) exit;
const MKP_VERSION = '20261029c';
// Public identity must be supplied by the site operator, never inferred from a login.
const MKP_ADMIN_NAME = '';
const MKP_CONTACT_EMAIL = '';
function mkp_privacy_ready() { return MKP_ADMIN_NAME !== '' && is_email(MKP_CONTACT_EMAIL); }
function mkp_https_images($content) {
    // Only this site's uploads. External image protocols need separate verification.
    return preg_replace('~http://(?:www\.)?meteo-krosno\.pl/wp-content/uploads/~i', 'https://meteo-krosno.pl/wp-content/uploads/', $content);
}
add_filter('the_content', 'mkp_https_images', 999);
add_filter('wp_get_attachment_url', 'mkp_https_images', 999);
function mkp_privacy_content() {
    $file = __DIR__ . '/mk-publication-policy.php';
    $html = is_readable($file) ? require($file) : '';
    return str_replace(['{{ADMIN}}', '{{EMAIL}}'], [esc_html(MKP_ADMIN_NAME ?: '[Do uzupełnienia: tożsamość administratora]'), esc_html(MKP_CONTACT_EMAIL ?: '[Do uzupełnienia: publiczny adres e-mail]')], $html);
}
add_filter('mkseo_routes', function ($routes) {
    if (mkp_privacy_ready()) $routes['polityka-prywatnosci'] = ['Polityka prywatności i cookies | Meteo Krosno', 'Polityka prywatności', 'Informacje o danych osobowych, cookies, dobrowolnej analityce i ustawieniach prywatności w Meteo Krosno.', 'Dowiedz się, jakie dane są potrzebne do działania serwisu i jak możesz zarządzać zgodą na statystyki.'];
    return $routes;
});

/** One-time content migration with private rollback records and original revisions. */
add_action('init', function () {
    if (get_option('mkp_migrated_' . MKP_VERSION)) return;
    $backup = get_option('mkp_backup_' . MKP_VERSION, ['test' => null, 'images' => []]);
    $test = get_page_by_path('testowy-wpis-prognozy', OBJECT, 'post');
    if ($test && $test->post_status !== 'trash') {
        $backup['test'] = ['id' => $test->ID, 'status' => $test->post_status];
        update_option('mkp_backup_' . MKP_VERSION, $backup, false);
        if (!wp_trash_post($test->ID)) return;
    }
    foreach (get_posts(['post_type' => ['post','page'], 'post_status' => 'publish', 'posts_per_page' => -1]) as $post) {
        $updated = mkp_https_images($post->post_content);
        if ($updated === $post->post_content) continue;
        $backup['images'][$post->ID] = $post->post_content;
        update_option('mkp_backup_' . MKP_VERSION, $backup, false);
        wp_save_post_revision($post->ID);
        $result = wp_update_post(['ID' => $post->ID, 'post_content' => wp_slash($updated)], true);
        if (is_wp_error($result)) return;
    }
    update_option('mkp_backup_' . MKP_VERSION, $backup, false);
    update_option('mkp_migrated_' . MKP_VERSION, 1, false);
}, 21);
add_action('init', function () {
    $key = 'mkp_policy_' . MKP_VERSION . (mkp_privacy_ready() ? '_ready' : '_draft');
    if (get_option($key)) return;
    $page = get_page_by_path('polityka-prywatnosci', OBJECT, 'page');
    if ($page && !get_option('mkp_previous_policy_v1')) add_option('mkp_previous_policy_v1', ['id'=>$page->ID, 'status'=>$page->post_status, 'content'=>$page->post_content], '', false);
    $id = wp_insert_post(wp_slash(['ID' => $page ? $page->ID : 0, 'post_type' => 'page', 'post_status' => mkp_privacy_ready() ? 'publish' : 'draft', 'post_name' => 'polityka-prywatnosci', 'post_title' => 'Polityka prywatności', 'post_content' => mkp_privacy_content(), 'comment_status' => 'closed', 'ping_status' => 'closed']), true);
    if (!is_wp_error($id)) update_option($key, $id, false);
}, 22);
add_action('init', function () {
    if (!get_option('mkp_migrated_' . MKP_VERSION) || get_option('mkp_cache_flushed_' . MKP_VERSION)) return;
    if (function_exists('w3tc_flush_all')) {
        w3tc_flush_all();
        update_option('mkp_cache_flushed_' . MKP_VERSION, 1, false);
    }
}, 99);

function mkp_humidity($inventory, $now = null) {
    $now = $now ?? time();
    foreach (($inventory['sensors'] ?? []) as $row) {
        $sensor = trim(strtolower(preg_replace('/[^a-zA-Z0-9]+/', '_', $row['sensor'] ?? '')), '_');
        if (!preg_match('/(?:^|_)(temp_and_humidity_ch1|tf_ch1)(?:_|$)/', $sensor, $group) || !preg_match('/(?:^|_)humidity$/', $sensor)) continue;
        $value = str_replace(',', '.', (string) ($row['value'] ?? ''));
        $at = (int) ($row['time'] ?? 0);
        if (is_numeric($value) && is_finite((float)$value) && $value >= 0 && $value <= 100 && $at > 0 && $at <= $now + 60 && $now - $at <= 900) return ['value'=>(float)$value,'at'=>$at,'sensor'=>$group[1]];
    }
    return null;
}
function mkp_correct_live($data, $humidity, $now = null) {
    $now = $now ?? time();
    if (!$humidity) return $data;
    if (!isset($data['stations']) || !is_array($data['stations'])) return $data;
    foreach ($data['stations'] as &$station) {
        if (($station['id'] ?? '') !== 'lesniowka' || empty($station['available']) || !empty($station['fallback'])) continue;
        $station['humidity'] = $humidity['value'];
        $station['humiditySensor'] = $humidity['sensor'] ?? 'tf_ch1';
        $station['timestamps']['humidity'] = $humidity['at'];
        $temperature = $station['temp'] ?? null;
        $at = (int) ($station['timestamps']['temp'] ?? 0);
        if (is_numeric($temperature) && $temperature > -90 && $temperature < 60 && abs($now-$at)<=900 && abs($at-$humidity['at'])<=900 && $humidity['value']>0) {
            $gamma = log($humidity['value']/100) + 17.62*$temperature/(243.12+$temperature);
            $station['dew'] = round(243.12*$gamma/(17.62-$gamma),1);
            $station['timestamps']['dew'] = min($at,$humidity['at']);
        } else { $station['dew'] = null; $station['timestamps']['dew'] = null; }
    }
    unset($station);
    return $data;
}
add_action('template_redirect', function () {
    if (($_GET['mk_api'] ?? '') !== 'live') return;
    ob_start(function ($body) {
        $data = json_decode($body, true);
        if (!is_array($data)) return $body;
        $humidity = mkp_humidity(get_transient('mk_eco_inventory_lesniowka') ?: []);
        if (!$humidity) {
            $humidity = get_transient('mkp_public_humidity_v1');
            if (!$humidity || time()-$humidity['at']>900) {
                // The same public sensor API used by the station dashboard; no credentials.
                $response = wp_remote_get(home_url('/wp-content/plugins/mk-station-dashboards/station-api.php?station=lesniowka&range=6h&resolution=60'), ['timeout'=>4,'redirection'=>0]);
                if (!is_wp_error($response) && wp_remote_retrieve_response_code($response)===200) {
                    $station = json_decode(wp_remote_retrieve_body($response),true);
                    $point = $station['latest'] ?? [];
                    $humidity = mkp_humidity(['sensors'=>[['sensor'=>'tf_ch1.humidity','value'=>$point['humidity']??null,'time'=>$point['ts']??0]]]);
                    if ($humidity) set_transient('mkp_public_humidity_v1',$humidity,30);
                }
            }
        }
        if (!$humidity) return $body;
        return wp_json_encode(mkp_correct_live($data,$humidity), JSON_UNESCAPED_UNICODE|JSON_INVALID_UTF8_SUBSTITUTE);
    });
}, -60);

function mkp_clean_html($html) {
    if (stripos($html,'<html')===false) return $html;
    $html = mkp_https_images($html);
    // Remove legacy measurement loaders/configuration before any browser receives them.
    $html = preg_replace_callback('~<script\b[^>]*>.*?</script>~is', function ($m) {
        return preg_match('~googletagmanager\.com|google-analytics\.com|\bgtag\s*\(|GoogleAnalyticsObject~i',$m[0]) ? '' : $m[0];
    },$html);
    $html = preg_replace('~<noscript>\s*<iframe[^>]*googletagmanager\.com.*?</noscript>~is','',$html);
    $html = preg_replace('~<link\b[^>]*(?:googletagmanager\.com|google-analytics\.com)[^>]*>~i','',$html);
    if (isset($_GET['mk_station_frame'])) return $html;
    if (strpos($html,'id="mk-privacy-root"')!==false) return $html;
    $base = content_url('/mu-plugins/mk-publication-assets/');
    $html = str_ireplace('</head>', '<link rel="stylesheet" href="'.esc_url($base.'consent-20261029.css').'"><script defer src="'.esc_url($base.'consent-20261029.js').'" data-mk-analytics-ready="'.(mkp_privacy_ready()?'1':'0').'"></script></head>', $html);
    $policy = mkp_privacy_ready() ? '<a href="'.esc_url(home_url('/polityka-prywatnosci/')).'">Polityka prywatności</a>' : '<button type="button" data-mk-privacy-info>Informacje o prywatności</button>';
    $ui = '<div id="mk-privacy-root"><div class="mk-privacy-links">'.$policy.'<button type="button" data-mk-consent-open>Ustawienia prywatności</button></div><section id="mk-consent-banner" role="dialog" aria-labelledby="mk-consent-title" aria-describedby="mk-consent-description" hidden><h2 id="mk-consent-title">Twoja prywatność</h2><p id="mk-consent-description">Serwis działa bez zgody na analitykę. Za Twoją zgodą Google Analytics pomoże nam zrozumieć, jak korzystasz ze strony. Możesz odmówić lub zmienić wybór w ustawieniach prywatności.</p><div class="mk-consent-actions"><button type="button" data-mk-consent-deny>Tylko niezbędne</button><button type="button" data-mk-consent-accept'.(mkp_privacy_ready()?'':' disabled').'>Zgadzam się na analitykę</button><button type="button" data-mk-consent-close>Zamknij bez zgody</button></div>'.(mkp_privacy_ready()?$policy:'<p>Analityka pozostaje wyłączona do zakończenia konfiguracji informacji o prywatności.</p>').'<p role="status" id="mk-consent-status"></p></section><section id="mk-privacy-info" role="dialog" aria-labelledby="mk-privacy-info-title" hidden><h2 id="mk-privacy-info-title">Informacje o prywatności</h2><p>Do działania serwisu i ochrony przed nadużyciami hosting przetwarza dane techniczne połączenia, m.in. adres IP, czas i adres żądania. Twój wybór dotyczący analityki zapisujemy w przeglądarce na 180 dni. Google Analytics nie jest obecnie uruchamiany. Mapy i biblioteki dostarczane przez zewnętrzne serwisy mogą otrzymywać dane techniczne połączenia.</p><button type="button" data-mk-privacy-close>Zamknij</button></section></div>';
    return str_ireplace('</body>',$ui.'</body>',$html);
}
add_action('template_redirect', function () {
    if (is_admin() || is_feed() || isset($_GET['mk_api']) || isset($_GET['mk_station_api']) || isset($_GET['mk_station_data'])) return;
    ob_start('mkp_clean_html');
}, -2000);
add_action('template_redirect', function () {
    if (($_GET['mk_api'] ?? '') !== 'publication-status') return;
    nocache_headers();header('Content-Type: application/json; charset=UTF-8');header('X-Robots-Tag: noindex, nofollow');
    $test=get_page_by_path('testowy-wpis-prognozy',OBJECT,'post');
    $policy=get_page_by_path('polityka-prywatnosci',OBJECT,'page');
    $remaining=false;
    foreach(get_posts(['post_type'=>['post','page'],'post_status'=>'publish','posts_per_page'=>-1]) as $post) if(mkp_https_images($post->post_content)!==$post->post_content){$remaining=true;break;}
    echo wp_json_encode(['migrationComplete'=>(bool)get_option('mkp_migrated_'.MKP_VERSION),'testRemoved'=>!$test||$test->post_status==='trash','oldImageReferencesRemaining'=>$remaining,'cacheFlushed'=>(bool)get_option('mkp_cache_flushed_'.MKP_VERSION),'privacyPublished'=>$policy&&$policy->post_status==='publish','privacyDraftReady'=>$policy&&$policy->post_status==='draft','analyticsEnabled'=>mkp_privacy_ready()]);exit;
},-2500);
