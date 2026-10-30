<?php
/** Plugin Name: Meteo Krosno — płynny start pulpitu
 * Version: 20261030a
 */
defined('ABSPATH') || exit;
function mk_home_loading_html($html) {
    if (strpos($html, 'mk-home-loading-style') !== false || !preg_match('~<main\b[^>]*\bid=["\']main["\'][^>]*>~i', $html)) return $html;
    $head = '<style id="mk-home-loading-style">' . file_get_contents(__DIR__.'/mk-home-loading.css') . '</style>';
    $head .= '<script data-no-optimize="1">' . file_get_contents(__DIR__.'/mk-home-loading.js') . '</script>';
    $html = preg_replace_callback('~<head\b[^>]*>~i', fn($m) => $m[0].$head, $html, 1);
    $loader = '<section id="mk-home-loader" aria-label="Ładowanie strony głównej"><div class="mk-load-card"><span class="mk-load-orbit" aria-hidden="true"></span><p class="mk-load-brand">Meteo Krosno</p><p role="status" aria-live="polite">Ładowanie pogody…</p><p class="mk-load-note">Za chwilę zobaczysz pomiary i prognozę.</p><button type="button" id="mk-load-basic">Pokaż podstawową wersję</button></div></section>';
    return preg_replace_callback('~<main\b[^>]*\bid=["\']main["\'][^>]*>~i', fn($m) => $loader.$m[0], $html, 1);
}
add_action('template_redirect', function () {
    if (is_admin() || is_feed() || is_404() || isset($_GET['mk_api'])) return;
    $path=trim((string)parse_url($_SERVER['REQUEST_URI'] ?? '/', PHP_URL_PATH), '/');
    if (!in_array($path, ['', 'strona-glowna'], true)) return;
    // Outer buffer runs after the existing SEO fallback has been rendered.
    ob_start('mk_home_loading_html');
}, -1000);
