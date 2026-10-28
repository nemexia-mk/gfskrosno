<?php
/** Screen-only palette for the existing monthly and annual climate reports. */
if (!defined('ABSPATH')) { exit; }
$mk_climate_path = rtrim((string) parse_url($_SERVER['REQUEST_URI'] ?? '', PHP_URL_PATH), '/');
if (in_array($mk_climate_path, ['/klimat-miesieczny', '/klimat-roczny'], true)) {
    ob_start(static function ($html) {
        if (stripos($html, '</head>') === false || strpos($html, 'id="mk-climate-palette"') !== false) { return $html; }
        $url = 'https://meteo-krosno.pl/wp-content/themes/meteo-krosno-modern/assets/climate-panels-20261028.css';
        $link = '<link id="mk-climate-palette" rel="stylesheet" href="' . $url . '" media="screen">';
        return preg_replace('~</head>~i', $link . '</head>', $html, 1);
    });
}
unset($mk_climate_path);
